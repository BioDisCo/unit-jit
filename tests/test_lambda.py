"""Tests for lambda inference and default-arg binding."""

import math

import pytest
from pint import UnitRegistry

from unit_jit._inferrer import infer_return_units

ureg = UnitRegistry()


# ---------------------------------------------------------------------------
# Lambda inference + default-arg binding
# ---------------------------------------------------------------------------


def test_lambda_return_unit_inferred_from_defaults():
    # step is a time; the m/s default and the dimensionless ramp give a m/s result.
    speed = 0.25 * ureg.m / ureg.s
    t_ramp = 30 * ureg.us
    cb = lambda step, s=speed, t=t_ramp: s * (1 - math.exp(-((step / t) ** 2)))  # noqa: E731

    inferred, reg = infer_return_units(cb, (1 * ureg.s,), {}, {})
    assert reg is not None
    assert inferred.dimensionality == (ureg.m / ureg.s).dimensionality


def test_lambda_transcendental_wrong_unit_raises():
    # t_ramp in metres: step / t is s/m, fed to exp -> dimensional error.
    speed = 0.25 * ureg.m / ureg.s
    t_bad = 30 * ureg.m
    cb = lambda step, s=speed, t=t_bad: s * (1 - math.exp(-((step / t) ** 2)))  # noqa: E731

    with pytest.raises(TypeError, match="dimensionless argument"):
        infer_return_units(cb, (1 * ureg.s,), {}, {})


def test_default_binding_for_regular_function():
    def f(x, scale=2 * ureg.s):
        return x / scale

    inferred, _ = infer_return_units(f, (1 * ureg.m,), {}, {})
    assert inferred.dimensionality == (ureg.m / ureg.s).dimensionality


# ---------------------------------------------------------------------------
# Stored-lambda inference: lambda assigned to a local variable and called
# within a @unit_jit function body (_LambdaNode + _infer_stored_lambda path)
# ---------------------------------------------------------------------------


def test_stored_lambda_positional_args_correct_units():
    # All positional: both args carry the same dimensionality -> addition is valid.
    def f(a, b):
        fn = lambda x, y: x + y  # noqa: E731
        return fn(a, b)

    inferred, _ = infer_return_units(f, (1 * ureg.m, 1 * ureg.m), {}, {})
    assert inferred.dimensionality == ureg.m.dimensionality


def test_stored_lambda_positional_args_mismatched_units_raises():
    # Adding metres to metres/second inside the lambda must raise at inference time.
    def f(a, b):
        fn = lambda x, y: x + y  # noqa: E731
        return fn(a, b)

    with pytest.raises(TypeError):
        infer_return_units(f, (1 * ureg.m / ureg.s, 1 * ureg.m), {}, {})


def test_stored_lambda_outer_scope_default():
    # Default b=y is resolved from the outer function's env at the call site.
    def f(x, y):
        fn = lambda a, b=y: a + b  # noqa: E731
        return fn(x)

    inferred, _ = infer_return_units(f, (1 * ureg.m, 1 * ureg.m), {}, {})
    assert inferred.dimensionality == ureg.m.dimensionality


def test_stored_lambda_outer_scope_default_wrong_unit_raises():
    # Default b=y has wrong dimensionality relative to positional a -> raises.
    def f(x, y):
        fn = lambda a, b=y: a + b  # noqa: E731
        return fn(x)

    with pytest.raises(TypeError):
        infer_return_units(f, (1 * ureg.m / ureg.s, 1 * ureg.m), {}, {})


def test_stored_lambda_transcendental_dimensional_arg_raises():
    # Transcendental inside a stored lambda: dimensional arg must be caught.
    def f(t):
        fn = lambda x: math.exp(x)  # noqa: E731
        return fn(t)

    with pytest.raises(TypeError, match="dimensionless argument"):
        infer_return_units(f, (1 * ureg.s,), {}, {})

