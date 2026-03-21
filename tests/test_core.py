"""Tests for unit_jit."""

from dataclasses import dataclass
from typing import cast

import numpy as np
import pytest
from pint import Quantity

from unit_jit import unit_jit, ureg

# ── Shared decorated functions (all compiled together on first call) ────────────


@unit_jit
def _div(d: Quantity, t: Quantity) -> Quantity:
    return cast("Quantity", d / t)


@unit_jit
def _strip_mag(x: Quantity) -> float:
    return x.magnitude  # stripped → plain float in fast zone


@unit_jit
def _velocity_loop(n: int) -> Quantity:
    """Return velocity as Quantity — unit_jit wraps the result back."""
    v = 0.0 * ureg.cm / ureg.s
    for _ in range(n):
        d = 10 * ureg.cm
        t = 2 * ureg.s
        v = d / t
    return v


@dataclass
class _Params:
    alpha: Quantity
    delta: Quantity


class _Model:
    def __init__(self, params: _Params) -> None:
        self.params = params

    @unit_jit
    def rate(self, mrna: Quantity) -> Quantity:
        """alpha - delta * mrna."""
        return cast("Quantity", self.params.alpha - self.params.delta * mrna)

    @unit_jit
    def simulate(self, n: int) -> np.ndarray:
        """Decay loop — inner call to rate() stays in fast zone."""
        mrna: Quantity = self.params.alpha / self.params.delta
        out = np.empty(n)
        dt = 0.1 * ureg.s
        for i in range(n):
            mrna = cast("Quantity", mrna + self.rate(mrna) * dt)
            out[i] = mrna.to_base_units().magnitude
        return out


# ── Correctness ────────────────────────────────────────────────────────────────


def test_returns_quantity():
    result = _div(10 * ureg.m, 2 * ureg.s)
    assert isinstance(result, Quantity)


def test_magnitude_matches_pint():
    _div(10 * ureg.m, 2 * ureg.s)  # warm-up
    result = _div(10 * ureg.m, 2 * ureg.s)
    expected = (10 * ureg.m / (2 * ureg.s)).to_base_units().magnitude
    assert abs(result.to_base_units().magnitude - expected) < 1e-12


def test_unit_invariant():
    """m/s and cm/s give the same SI result."""
    _div(1 * ureg.m, 1 * ureg.s)  # warm-up
    r1 = _div(10 * ureg.m, 2 * ureg.s)
    r2 = _div(1000 * ureg.cm, 200 * ureg.cs)
    assert abs(r1.to_base_units().magnitude - r2.to_base_units().magnitude) < 1e-12


def test_magnitude_stripped():
    """A function that calls .magnitude returns a plain float in fast zone."""
    _strip_mag(1 * ureg.m)  # warm-up
    result = _strip_mag(5 * ureg.m)
    assert isinstance(result, float | int)


def test_quantity_return_wrapped():
    """Returning a Quantity directly is wrapped back by unit_jit."""
    _velocity_loop(1)  # warm-up
    result = _velocity_loop(10)
    assert isinstance(result, Quantity)
    assert abs(result.to_base_units().magnitude - 0.05) < 1e-12  # 10 cm / 2 s = 0.05 m/s


# ── Dimension check ────────────────────────────────────────────────────────────


def test_dimension_mismatch_raises():
    """Second call with wrong dimension raises TypeError."""
    _div(10 * ureg.m, 2 * ureg.s)  # warm-up: arg1=[length], arg2=[time]
    with pytest.raises(TypeError, match="dimensions"):
        _div(10 * ureg.m, 2 * ureg.m)  # arg2 is [length], expected [time]


# ── Class with snapshot ────────────────────────────────────────────────────────


def test_class_method_result():
    params = _Params(
        alpha=0.1 * ureg.mol / ureg.L / ureg.s,
        delta=0.01 / ureg.s,
    )
    model = _Model(params)
    mrna = 10 * ureg.mol / ureg.L
    result = model.rate(mrna)
    assert isinstance(result, Quantity)
    expected = (params.alpha - params.delta * mrna).to_base_units().magnitude
    assert abs(result.to_base_units().magnitude - expected) < 1e-12


def test_simulate_shape_and_finite():
    params = _Params(
        alpha=0.1 * ureg.mol / ureg.L / ureg.s,
        delta=0.01 / ureg.s,
    )
    model = _Model(params)
    model.simulate(10)  # warm-up
    out = model.simulate(50)
    assert out.shape == (50,)
    assert np.all(np.isfinite(out))


def test_simulate_matches_pint_baseline():
    """Fast simulate gives same trajectory as plain Pint."""
    params = _Params(
        alpha=0.1 * ureg.mol / ureg.L / ureg.s,
        delta=0.01 / ureg.s,
    )
    model = _Model(params)

    # Pint baseline (no fast zone)
    def _simulate_pint(n: int) -> np.ndarray:
        mrna = params.alpha / params.delta
        out = np.empty(n)
        dt = 0.1 * ureg.s
        for i in range(n):
            mrna = mrna + (params.alpha - params.delta * mrna) * dt
            out[i] = mrna.to_base_units().magnitude
        return out

    model.simulate(10)  # warm-up
    fast_out = model.simulate(20)
    pint_out = _simulate_pint(20)
    np.testing.assert_allclose(fast_out, pint_out, rtol=1e-10)
