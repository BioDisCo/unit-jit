"""Compatibility tests for pintrs as a drop-in pint replacement.

Covers: type detection, CST rewriting of unit literals, registry capture,
dimension checking, NumPy array quantities, class/method decoration with
dataclass params, NamedTuple param snapshots, and inner-call fast-zone
propagation.

All module-level names use a _pt_ / _Pt prefix to avoid qualname collisions
with identically-named functions in other test modules (qualnames are the key
in the global _return_registry dict).
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pytest

pintrs = pytest.importorskip("pintrs")

from unit_jit import unit_jit  # noqa: E402

ureg = pintrs.UnitRegistry()


# --- scalar functions ---


@unit_jit
def _pt_div(d: pintrs.Quantity, t: pintrs.Quantity) -> pintrs.Quantity:
    return d / t


@unit_jit
def _pt_velocity_loop(n: int) -> pintrs.Quantity:
    v = 0.0 * ureg.cm / ureg.s
    for _ in range(n):
        d = 10.0 * ureg.cm
        t = 2.0 * ureg.s
        v = d / t
    return v


# --- numpy functions ---


@unit_jit
def _pt_vec_div(d: pintrs.Quantity, t: pintrs.Quantity) -> pintrs.Quantity:
    return d / t  # type: ignore[return-value]


@unit_jit
def _pt_weighted_sum(vals: pintrs.Quantity, weights: np.ndarray) -> pintrs.Quantity:
    return (vals * weights).sum()  # type: ignore[return-value]


@unit_jit
def _pt_l2_norm(v: pintrs.Quantity) -> float:
    mags = v.magnitude
    return float(np.sqrt(np.dot(mags, mags)))


# --- class with dataclass params and inner method call ---


@dataclass
class _PtParams:
    alpha: pintrs.Quantity
    delta: pintrs.Quantity


class _PtModel:
    def __init__(self, params: _PtParams) -> None:
        self.params = params

    @unit_jit
    def rate(self, mrna: pintrs.Quantity) -> pintrs.Quantity:
        return self.params.alpha - self.params.delta * mrna  # type: ignore[return-value]

    @unit_jit
    def simulate(self, n: int) -> np.ndarray:
        mrna: pintrs.Quantity = self.params.alpha / self.params.delta
        out = np.empty(n)
        dt = 0.1 * ureg.s
        for i in range(n):
            mrna = mrna + self.rate(mrna) * dt  # type: ignore[assignment]
            out[i] = mrna.to_base_units().magnitude
        return out


# --- NamedTuple params ---


class _PtNTParams(NamedTuple):
    alpha: pintrs.Quantity
    delta: pintrs.Quantity


class _PtNTModel:
    def __init__(self, params: _PtNTParams) -> None:
        self.params = params

    @unit_jit
    def run(self, x: pintrs.Quantity) -> pintrs.Quantity:
        return self.params.alpha * x - self.params.delta * x  # type: ignore[return-value]


# === scalar tests ===


def test_returns_pintrs_quantity():
    result = _pt_div(10 * ureg.m, 2 * ureg.s)
    assert isinstance(result, pintrs.Quantity)


def test_magnitude_correct():
    _pt_div(10 * ureg.m, 2 * ureg.s)  # warm-up
    result = _pt_div(10 * ureg.m, 2 * ureg.s)
    expected = (10 * ureg.m / (2 * ureg.s)).to_base_units().magnitude
    assert abs(result.to_base_units().magnitude - expected) < 1e-12


def test_unit_invariant():
    """cm/s and m/s inputs give the same SI result."""
    _pt_div(1 * ureg.m, 1 * ureg.s)  # warm-up
    r1 = _pt_div(10 * ureg.m, 2 * ureg.s)
    r2 = _pt_div(1000 * ureg.cm, 200 * ureg.cs)
    assert abs(r1.to_base_units().magnitude - r2.to_base_units().magnitude) < 1e-12


def test_unit_literal_rewriting():
    """ureg.UNIT literals inside the body are rewritten to SI floats."""
    _pt_velocity_loop(1)  # warm-up
    result = _pt_velocity_loop(5)
    assert isinstance(result, pintrs.Quantity)
    assert abs(result.to_base_units().magnitude - 0.05) < 1e-12  # 10 cm / 2 s = 0.05 m/s


def test_registry_captured_from_args():
    """Result is interoperable with other pintrs quantities from the same registry."""
    _pt_div(10 * ureg.m, 2 * ureg.s)  # warm-up
    result = _pt_div(10 * ureg.m, 2 * ureg.s)
    combined = result + 1 * ureg.m / ureg.s
    assert isinstance(combined, pintrs.Quantity)


def test_dimension_mismatch_raises():
    _pt_div(10 * ureg.m, 2 * ureg.s)  # warm-up: arg0=[length], arg1=[time]
    with pytest.raises(TypeError, match="dimensions"):
        _pt_div(10 * ureg.m, 2 * ureg.m)  # arg1 is [length], expected [time]


# === numpy tests ===


def test_vec_div_returns_quantity():
    v = np.array([3.0, 4.0, 5.0]) * ureg.m
    result = _pt_vec_div(v, 2.0 * ureg.s)
    assert isinstance(result, pintrs.Quantity)


def test_vec_div_shape_and_values():
    v = np.array([3.0, 4.0]) * ureg.m
    t = 2.0 * ureg.s
    _pt_vec_div(v, t)  # warm-up
    result = _pt_vec_div(v, t)
    expected = (v / t).to_base_units().magnitude
    assert result.to_base_units().magnitude.shape == (2,)
    np.testing.assert_allclose(result.to_base_units().magnitude, expected)


def test_vec_div_non_si_input():
    """Non-SI inputs are converted correctly: 300 cm / 2000 ms = 1.5 m/s."""
    v = np.array([300.0, 600.0]) * ureg.cm
    t = 2000.0 * ureg.ms
    _pt_vec_div(v, t)  # warm-up
    result = _pt_vec_div(v, t)
    np.testing.assert_allclose(result.to_base_units().magnitude, [1.5, 3.0], rtol=1e-12)


def test_weighted_sum_value():
    vals = np.array([1.0, 2.0, 3.0]) * ureg.m
    w = np.array([0.5, 0.3, 0.2])
    _pt_weighted_sum(vals, w)  # warm-up
    result = _pt_weighted_sum(vals, w)
    expected = float(np.dot(vals.to_base_units().magnitude, w))
    assert isinstance(result, pintrs.Quantity)
    assert abs(result.to_base_units().magnitude - expected) < 1e-12


def test_l2_norm_returns_float():
    v = np.array([3.0, 4.0]) * ureg.m
    _pt_l2_norm(v)  # warm-up
    result = _pt_l2_norm(v)
    assert isinstance(result, float)
    assert abs(result - 5.0) < 1e-12


def test_l2_norm_non_si_units():
    """Norm of input in cm equals SI magnitude norm (metres)."""
    v = np.array([300.0, 400.0]) * ureg.cm  # 3 m, 4 m
    _pt_l2_norm(v)  # warm-up
    assert abs(_pt_l2_norm(v) - 5.0) < 1e-12


# === class with inner method call ===


def test_model_rate_correct():
    params = _PtParams(
        alpha=0.1 * ureg.mol / ureg.L / ureg.s,
        delta=0.01 / ureg.s,
    )
    model = _PtModel(params)
    mrna = 10 * ureg.mol / ureg.L
    result = model.rate(mrna)
    assert isinstance(result, pintrs.Quantity)
    expected = (params.alpha - params.delta * mrna).to_base_units().magnitude
    assert abs(result.to_base_units().magnitude - expected) < 1e-12


def test_model_simulate_shape_and_finite():
    params = _PtParams(
        alpha=0.1 * ureg.mol / ureg.L / ureg.s,
        delta=0.01 / ureg.s,
    )
    model = _PtModel(params)
    model.simulate(5)  # warm-up
    out = model.simulate(30)
    assert out.shape == (30,)
    assert np.all(np.isfinite(out))


def test_model_simulate_matches_plain_pintrs():
    """Fast simulate gives the same trajectory as plain pintrs arithmetic."""
    params = _PtParams(
        alpha=0.1 * ureg.mol / ureg.L / ureg.s,
        delta=0.01 / ureg.s,
    )

    def _plain(n: int) -> np.ndarray:
        mrna = params.alpha / params.delta
        out = np.empty(n)
        dt = 0.1 * ureg.s
        for i in range(n):
            mrna = mrna + (params.alpha - params.delta * mrna) * dt
            out[i] = mrna.to_base_units().magnitude
        return out

    model = _PtModel(params)
    model.simulate(5)  # warm-up
    np.testing.assert_allclose(model.simulate(20), _plain(20), rtol=1e-10)


# === NamedTuple params snapshot ===


def test_namedtuple_params_correct():
    alpha = 2.0 / ureg.s
    delta = 0.5 / ureg.s
    x = 3.0 * ureg.mol / ureg.L
    model = _PtNTModel(_PtNTParams(alpha=alpha, delta=delta))
    model.run(x)  # warm-up
    result = model.run(x)
    expected = (alpha * x - delta * x).to_base_units().magnitude
    assert isinstance(result, pintrs.Quantity)
    assert abs(result.to_base_units().magnitude - expected) < 1e-12
