"""Tests for unit_jit."""

from dataclasses import dataclass
from typing import cast

import numpy as np
import pytest
from pint import Quantity, UnitRegistry

from unit_jit import unit_jit

ureg = UnitRegistry()

# Shared decorated functions (all compiled together on first call)


@unit_jit
def _div(d: Quantity, t: Quantity) -> Quantity:
    return cast("Quantity", d / t)


@unit_jit
def _strip_mag(x: Quantity) -> float:
    return x.magnitude  # stripped to plain float in fast zone


@unit_jit
def _velocity_loop(n: int) -> Quantity:
    """Return velocity as Quantity; unit_jit wraps the result back."""
    v = 0.0 * ureg.cm / ureg.s
    for _ in range(n):
        d = 10 * ureg.cm
        t = 2 * ureg.s
        v = d / t
    return v


@unit_jit
def _div_kw(d: Quantity, *, t: Quantity) -> Quantity:
    return cast("Quantity", d / t)


@unit_jit
def _add_matching_constant(d: Quantity) -> Quantity:
    result = d + 1 * ureg.m  # constant added in the middle of the body
    return result


@unit_jit
def _mul_constant(d: Quantity) -> Quantity:
    result = d * (ureg.m * 2)  # constant multiplied in the middle of the body
    return result


@unit_jit
def _add_constant_in_branch(d: Quantity, flag: bool) -> Quantity:
    if flag:
        result = d + 1 * ureg.m  # [length] + [length]
    else:
        result = d + 2 * ureg.m  # [length] + [length]
    return result


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
        """Decay loop; inner call to rate() stays in fast zone."""
        mrna: Quantity = self.params.alpha / self.params.delta
        out = np.empty(n)
        dt = 0.1 * ureg.s
        for i in range(n):
            mrna = cast("Quantity", mrna + self.rate(mrna) * dt)
            out[i] = mrna.to_base_units().magnitude
        return out


# Correctness


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


# Dimension check


def test_dimension_mismatch_raises():
    """Second call with wrong positional dimension raises TypeError."""
    _div(10 * ureg.m, 2 * ureg.s)  # warm-up: arg1=[length], arg2=[time]
    with pytest.raises(TypeError, match="dimensions"):
        _div(10 * ureg.m, 2 * ureg.m)  # arg2 is [length], expected [time]


def test_kwarg_dimension_mismatch_raises():
    """Second call with wrong keyword argument dimension raises TypeError."""
    _div_kw(10 * ureg.m, t=2 * ureg.s)  # warm-up
    with pytest.raises(TypeError, match="dimensions"):
        _div_kw(10 * ureg.m, t=2 * ureg.m)  # t is [length], expected [time]


def test_body_add_matching_constant():
    """Adding a same-dimension constant in the body succeeds on first call."""
    result = _add_matching_constant(5 * ureg.m)
    assert isinstance(result, Quantity)
    expected = (6 * ureg.m).to_base_units().magnitude
    assert abs(result.to_base_units().magnitude - expected) < 1e-12


def test_body_add_mismatched_constant_raises():
    """Adding a constant with wrong dimension raises TypeError on the first call (inferrer)."""

    @unit_jit
    def _bad_add(d: Quantity, t: Quantity) -> Quantity:
        return cast("Quantity", d / t + 1 * ureg.m)  # [velocity] + [length]

    with pytest.raises(TypeError):
        _bad_add(10 * ureg.m, 2 * ureg.s)


def test_body_mul_matching_constant():
    """Multiplying by a unit constant in the body succeeds on first call."""
    result = _mul_constant(3 * ureg.m)
    assert isinstance(result, Quantity)
    expected = (3 * ureg.m * 2 * ureg.m).to_base_units().magnitude
    assert abs(result.to_base_units().magnitude - expected) < 1e-12


def test_body_mul_then_add_mismatched_raises():
    """Multiplying by a unit constant then adding the original (wrong dim) raises TypeError."""

    @unit_jit
    def _bad_mul_add(d: Quantity) -> Quantity:
        return cast("Quantity", d * (2 * ureg.m) + d)  # [length^2] + [length]

    with pytest.raises(TypeError):
        _bad_mul_add(3 * ureg.m)


def test_body_if_branch_matching_constant():
    """Adding a matching constant in both if/else branches succeeds on first call."""
    result = _add_constant_in_branch(5 * ureg.m, flag=True)
    assert isinstance(result, Quantity)
    assert abs(result.to_base_units().magnitude - 6.0) < 1e-12
    result2 = _add_constant_in_branch(5 * ureg.m, flag=False)
    assert abs(result2.to_base_units().magnitude - 7.0) < 1e-12


def test_body_if_branch_mismatched_constant_raises():
    """A mismatched constant in one branch raises TypeError on first call."""

    @unit_jit
    def _bad_branch(d: Quantity, t: Quantity, flag: bool) -> Quantity:
        if flag:
            result = d / t + 1 * ureg.m  # [velocity] + [length]
        else:
            result = d / t
        return result

    with pytest.raises(TypeError):
        _bad_branch(10 * ureg.m, 2 * ureg.s, True)


def test_body_if_branch_inconsistent_return_raises():
    """Branches returning different dimensions are caught by the inferrer."""

    @unit_jit
    def _inconsistent_return(d: Quantity, t: Quantity, flag: bool) -> Quantity:
        if flag:
            return cast("Quantity", d / t)  # [velocity]
        return d  # [length]

    with pytest.raises(TypeError):
        _inconsistent_return(10 * ureg.m, 2 * ureg.s, True)


def test_self_attr_dimension_mismatch_in_body_raises():
    """Method body that adds self.alpha (rate) and self.volume (volume) is caught at inference."""

    class _BadModel:
        def __init__(self, alpha: Quantity, volume: Quantity) -> None:
            self.alpha = alpha
            self.volume = volume

        @unit_jit
        def run(self, x: Quantity) -> Quantity:
            return cast("Quantity", self.alpha + self.volume)  # [1/s] + [volume]

    model = _BadModel(0.5 / ureg.s, 1.0 * ureg.L)
    with pytest.raises(TypeError):
        model.run(1.0 * ureg.mol / ureg.L)


def test_namedtuple_params_body_dimension_mismatch_raises():
    """NamedTuple params of incompatible dimensions are caught at inference."""

    class _BadNTModel:
        def __init__(self, params: _NTParams) -> None:
            self.params = params

        @unit_jit
        def run(self, x: Quantity) -> Quantity:
            # alpha is 1/s, x is mol/L: alpha*x is mol/L/s; params.delta is also 1/s
            # adding mol/L/s and 1/s is a dimension error
            return cast("Quantity", self.params.alpha * x + self.params.delta)

    model = _BadNTModel(_NTParams(alpha=2.0 / ureg.s, delta=0.5 / ureg.s))
    with pytest.raises(TypeError):
        model.run(3.0 * ureg.mol / ureg.L)


def test_while_loop_body_dimension_error_raises():
    """Accumulating with wrong-dimension increment inside a while loop is caught at inference."""

    class _BadLoop:
        def __init__(self, volume: Quantity) -> None:
            self.volume = volume

        @unit_jit
        def run(self, max_time: Quantity, rate: Quantity) -> Quantity:
            time = 0.0 * ureg.s
            while time < max_time:
                time = cast("Quantity", time + self.volume)  # [time] + [volume]
            return time

    model = _BadLoop(1.0 * ureg.L)
    with pytest.raises(TypeError):
        model.run(10.0 * ureg.s, 1.0 / ureg.s)


# Class with snapshot


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


# While loop with Quantity accumulation and self Quantity attribute


class _WhileAccumulator:
    """Minimal Gillespie-style model: Quantity attribute on self, while loop, time accumulation."""

    def __init__(self, volume: Quantity) -> None:
        self.volume = volume  # [length^3]

    @unit_jit
    def run(self, max_time: Quantity, rate: Quantity) -> Quantity:
        """Accumulate time in a while loop; tau uses self.volume directly.

        rate: [1 / (volume * time)], tau = 1 / (rate * self.volume): [time]
        """
        tau = 1.0 / (rate * self.volume)
        time = 0.0 * ureg.s
        while time < max_time:
            time = cast("Quantity", time + tau)
        return time


def test_while_loop_quantity_accumulation_result():
    """While loop correctly accumulates a Quantity: result is in [max_time, max_time + tau)."""
    # rate = 0.5 / (L * s), volume = 1 L => tau ~ 2 s
    max_time = 10.0 * ureg.s
    rate = 0.5 / ureg.L / ureg.s
    volume = 1.0 * ureg.L
    model = _WhileAccumulator(volume)
    result = model.run(max_time, rate)
    tau = 1.0 / (rate * volume)
    assert isinstance(result, Quantity)
    assert result.to(ureg.s).magnitude >= max_time.to(ureg.s).magnitude - 1e-10
    assert result.to(ureg.s).magnitude < (max_time + tau).to(ureg.s).magnitude + 1e-10


def test_while_loop_self_volume_attribute():
    """self.volume Quantity attribute is read correctly: result in [max_time, max_time + tau)."""
    # rate = 1 / (L * s), volume = 2 L => tau ~ 0.5 s
    max_time = 1.0 * ureg.s
    rate = 1.0 / ureg.L / ureg.s
    volume = 2.0 * ureg.L
    model = _WhileAccumulator(volume)
    model.run(max_time, rate)  # warm-up
    result = model.run(max_time, rate)
    tau = 1.0 / (rate * volume)
    assert isinstance(result, Quantity)
    assert result.to(ureg.s).magnitude >= max_time.to(ureg.s).magnitude - 1e-10
    assert result.to(ureg.s).magnitude < (max_time + tau).to(ureg.s).magnitude + 1e-10


def test_while_loop_matches_pint_baseline():
    """Fast path gives the same result as plain Pint."""
    volume = 1.5 * ureg.L
    rate = 2.0 / ureg.L / ureg.s  # tau = 1/(2/s * 1.5) = 1/3 s
    max_time = 5.0 * ureg.s

    def _pint_run(max_time: Quantity, rate: Quantity, volume: Quantity) -> Quantity:
        tau = 1.0 / (rate * volume)
        time = 0.0 * ureg.s
        while time < max_time:
            time = cast("Quantity", time + tau)
        return time

    model = _WhileAccumulator(volume)
    model.run(max_time, rate)  # warm-up
    fast_result = model.run(max_time, rate)
    pint_result = _pint_run(max_time, rate, volume)
    assert fast_result.to(ureg.s).magnitude == pytest.approx(pint_result.to(ureg.s).magnitude)


# Custom registry


def test_custom_registry_result_uses_same_registry():
    """Results wrapped by unit_jit must belong to the user's registry, not unit_jit's internal."""
    from pint import UnitRegistry

    my_ureg = UnitRegistry()

    @unit_jit
    def _scale(x: my_ureg.Quantity, factor: float) -> my_ureg.Quantity:  # type: ignore[name-defined]
        return cast("Quantity", x * factor)

    x = 3.0 * my_ureg.meter
    _scale(x, 2.0)  # warm-up
    result = _scale(x, 2.0)

    assert isinstance(result, Quantity)
    assert result._REGISTRY is my_ureg  # noqa: SLF001
    assert abs(result.to_base_units().magnitude - 6.0) < 1e-12


def test_custom_registry_interop_with_other_quantities():
    """Result from unit_jit can be combined with Quantities from the same registry."""
    from pint import UnitRegistry

    my_ureg = UnitRegistry()

    @unit_jit
    def _double(x: my_ureg.Quantity) -> my_ureg.Quantity:  # type: ignore[name-defined]
        return cast("Quantity", x * 2.0)

    x = 1.0 * my_ureg.second
    _double(x)  # warm-up
    result = _double(x)

    # This would raise "Cannot operate with Quantity of different registries" before the fix.
    combined = result + 1.0 * my_ureg.second
    assert abs(combined.to_base_units().magnitude - 3.0) < 1e-12


# NamedTuple params snapshot


from typing import NamedTuple  # noqa: E402


class _NTParams(NamedTuple):
    alpha: Quantity
    delta: Quantity


class _ModelWithNamedTupleParams:
    def __init__(self, params: _NTParams) -> None:
        self.params = params

    @unit_jit
    def run(self, x: Quantity) -> Quantity:
        # alpha and delta both in 1/s, x in mol/L: each term is mol/L/s
        return cast("Quantity", self.params.alpha * x - self.params.delta * x)


def test_namedtuple_params_result_correct():
    """NamedTuple params with Quantity fields are snapshotted to SI floats at boundary."""
    alpha = 2.0 / ureg.s
    delta = 0.5 / ureg.s
    x = 3.0 * ureg.mol / ureg.L
    params = _NTParams(alpha=alpha, delta=delta)
    model = _ModelWithNamedTupleParams(params)
    model.run(x)  # warm-up
    result = model.run(x)
    expected = alpha * x - delta * x
    assert isinstance(result, Quantity)
    assert abs(result.to_base_units().magnitude - expected.to_base_units().magnitude) < 1e-12


def test_namedtuple_params_jit_active():
    """JIT is active for a method that accesses NamedTuple params."""
    import unit_jit as _uj

    alpha = 2.0 / ureg.s
    delta = 0.5 / ureg.s
    params = _NTParams(alpha=alpha, delta=delta)
    model = _ModelWithNamedTupleParams(params)
    model.run(3.0 * ureg.mol / ureg.L)
    qualname = model.run.__qualname__
    assert qualname in _uj._return_units
    assert qualname not in _uj._jit_disabled
