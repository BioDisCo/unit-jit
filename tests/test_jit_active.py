"""Verify that JIT speedup conditions are met after first call.

These tests check the internal state to confirm that:
- unit inference succeeded (return units cached, JIT enabled), or
- inference correctly failed and the function is marked as JIT-disabled.
"""

import pytest
from pint import Quantity, UnitRegistry

import unit_jit as _uj
from unit_jit import get_rewritten_source, unit_jit

ureg = UnitRegistry()


def _jit_active(func: object) -> bool:
    """Return True if JIT is active for func (inference succeeded, fast path used)."""
    qualname = getattr(func, "__qualname__", None)
    return (
        qualname is not None and qualname in _uj._return_units and qualname not in _uj._jit_disabled
    )


def _jit_disabled(func: object) -> bool:
    """Return True if JIT was disabled for func (inference failed)."""
    qualname = getattr(func, "__qualname__", None)
    return qualname is not None and qualname in _uj._jit_disabled


# Basic scalar function


@unit_jit
def _scalar(d: Quantity, t: Quantity) -> Quantity:
    return d / t


def test_scalar_jit_active():
    _scalar(10 * ureg.m, 2 * ureg.s)
    assert _jit_active(_scalar)


def test_scalar_module_compiled():
    _scalar(10 * ureg.m, 2 * ureg.s)
    assert _scalar.__module__ in _uj._compiled


def test_scalar_source_rewritten():
    _scalar(10 * ureg.m, 2 * ureg.s)
    # get_rewritten_source raises if not compiled; result differs from original
    src = get_rewritten_source(_scalar)
    assert src is not None


# Function returning dimensionless (None unit_info) is still JIT-active


@unit_jit
def _dimensionless(x: Quantity) -> float:
    return x.to_base_units().magnitude


def test_dimensionless_jit_active():
    _dimensionless(3 * ureg.m)
    assert _jit_active(_dimensionless)


@unit_jit
def _to_minutes_magnitude(t: Quantity) -> float:
    return t.to(ureg.min).magnitude


@unit_jit
def _to_dimensionless_magnitude(x: Quantity, y: Quantity) -> float:
    return (x / y).to(ureg.dimensionless).magnitude


@unit_jit
def _to_inverse_minutes_magnitude(rate: Quantity) -> float:
    return rate.to(1 / ureg.min).magnitude


@unit_jit
def _inverse_of_to_inverse_minutes_magnitude(rate: Quantity) -> float:
    return 1.0 / rate.to(1 / ureg.min).magnitude


def _plain_to_minutes_magnitude(t: Quantity) -> float:
    return t.to(ureg.min).magnitude


def _plain_to_dimensionless_magnitude(x: Quantity, y: Quantity) -> float:
    return (x / y).to(ureg.dimensionless).magnitude


def _plain_to_inverse_minutes_magnitude(rate: Quantity) -> float:
    return rate.to(1 / ureg.min).magnitude


def _plain_inverse_of_to_inverse_minutes_magnitude(rate: Quantity) -> float:
    return 1.0 / rate.to(1 / ureg.min).magnitude


def test_to_minutes_magnitude_matches_plain_pint():
    value = 120 * ureg.s
    assert _to_minutes_magnitude(value) == pytest.approx(_plain_to_minutes_magnitude(value))


def test_to_dimensionless_magnitude_matches_plain_pint():
    x = 6 * ureg.m
    y = 3 * ureg.m
    assert _to_dimensionless_magnitude(x, y) == pytest.approx(
        _plain_to_dimensionless_magnitude(x, y)
    )


def test_to_inverse_minutes_magnitude_matches_plain_pint():
    rate = 120 / ureg.s
    assert _to_inverse_minutes_magnitude(rate) == pytest.approx(
        _plain_to_inverse_minutes_magnitude(rate)
    )


def test_inverse_of_to_inverse_minutes_magnitude_matches_plain_pint():
    rate = 120 / ureg.s
    assert _inverse_of_to_inverse_minutes_magnitude(rate) == pytest.approx(
        _plain_inverse_of_to_inverse_minutes_magnitude(rate)
    )


# Class methods


@unit_jit
class _JitActiveModel:
    def rate(self, x: Quantity) -> Quantity:
        return x * 2.0


def test_class_method_jit_active():
    m = _JitActiveModel()
    m.rate(1 * ureg.m)
    assert _jit_active(m.rate)


# input_args: inference triggered at decoration time


def test_input_args_jit_active_before_explicit_call():
    @unit_jit(input_args=(ureg.m, ureg.s))
    def _input_args_fn(d: Quantity, t: Quantity) -> Quantity:
        return d / t

    # No explicit call: inference ran during decoration via input_args.
    assert _jit_active(_input_args_fn)


def test_input_args_result_correct():
    @unit_jit(input_args=(ureg.m, ureg.s))
    def _input_args_result(d: Quantity, t: Quantity) -> Quantity:
        return d / t

    result = _input_args_result(6 * ureg.m, 3 * ureg.s)
    assert isinstance(result, Quantity)
    assert abs(result.to_base_units().magnitude - 2.0) < 1e-12


def test_input_args_wrong_dimension_raises():
    @unit_jit(input_args=(ureg.m, ureg.s))
    def _input_args_dim(d: Quantity, t: Quantity) -> Quantity:
        return d / t

    with pytest.raises(TypeError):
        _input_args_dim(10 * ureg.m, 2 * ureg.m)  # m/m: wrong dimension for t


# __init_subclass__ automatic decoration


class _AutoJitBase:
    """Base class that applies @unit_jit to reaction_rates in subclasses transparently."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if "reaction_rates" in cls.__dict__:
            cls.reaction_rates = unit_jit(cls.reaction_rates)


class _UserModel(_AutoJitBase):
    def __init__(self, delta: Quantity, gamma: Quantity) -> None:
        self.delta = delta
        self.gamma = gamma

    # No @unit_jit here: applied automatically by _AutoJitBase.__init_subclass__
    def reaction_rates(self, state: list[Quantity]) -> list[Quantity]:
        return [self.delta * state[0], self.gamma]


def test_init_subclass_result_correct():
    """Auto-decorated reaction_rates returns list[Quantity] without explicit @unit_jit."""
    model = _UserModel(0.5 / ureg.s, 0.1 * ureg.mol / ureg.L / ureg.s)
    state = [10.0 * ureg.mol / ureg.L]
    result = model.reaction_rates(state)
    assert isinstance(result, list)
    assert all(isinstance(r, Quantity) for r in result)
    assert (
        abs(
            result[0].to_base_units().magnitude
            - (0.5 / ureg.s * state[0]).to_base_units().magnitude
        )
        < 1e-12
    )
    assert (
        abs(
            result[1].to_base_units().magnitude
            - (0.1 * ureg.mol / ureg.L / ureg.s).to_base_units().magnitude
        )
        < 1e-12
    )


def test_init_subclass_jit_active_after_first_call():
    """After the first call, __init_subclass__-decorated reaction_rates is JIT-compiled."""
    model = _UserModel(0.5 / ureg.s, 0.1 * ureg.mol / ureg.L / ureg.s)
    model.reaction_rates([10.0 * ureg.mol / ureg.L])
    assert _jit_active(model.reaction_rates)


def test_init_subclass_fast_path_matches_pint():
    """Warm call result matches the plain-Pint baseline."""
    delta = 0.5 / ureg.s
    gamma = 0.1 * ureg.mol / ureg.L / ureg.s
    state = [10.0 * ureg.mol / ureg.L]
    model = _UserModel(delta, gamma)
    model.reaction_rates(state)  # warm-up
    result = model.reaction_rates(state)
    assert (
        abs(result[0].to_base_units().magnitude - (delta * state[0]).to_base_units().magnitude)
        < 1e-12
    )
    assert abs(result[1].to_base_units().magnitude - gamma.to_base_units().magnitude) < 1e-12


# JIT disabled when inference fails (source not inspectable)


def test_jit_disabled_on_inference_failure(caplog):
    import logging

    # Build a function whose source cannot be retrieved by inspect.getsource.
    globs: dict = {}
    exec(  # noqa: S102
        "from unit_jit import unit_jit\n"
        "from pint import Quantity, UnitRegistry\n"
        "ureg = UnitRegistry()\n"
        "@unit_jit\n"
        "def _no_src(x: Quantity) -> Quantity:\n"
        "    return x * 2.0\n",
        globs,
    )
    f = globs["_no_src"]
    with caplog.at_level(logging.WARNING, logger="unit_jit"):
        f(1 * ureg.m)
    assert _jit_disabled(f)
    assert any("unit inference failed" in r.message for r in caplog.records)
