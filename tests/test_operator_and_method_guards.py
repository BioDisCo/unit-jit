"""Operator semantics and actual quantity-method dispatch regression tests."""

from types import SimpleNamespace

import numpy as np
import pytest
from _jit_state import expect_execution
from pint import UnitRegistry

from unit_jit import trace_execution, unit_jit

ureg = UnitRegistry()


def floor(a, b):
    return a // b


def floor_inplace(a, b):
    a //= b
    return a


def same(a, b):
    return a is b


def different(a, b):
    return a is not b


def identity_chain(a, b):
    return a is b is a


def equal(a, b):
    return a == b


@pytest.fixture(params=["pint", "pintrs"])
def reg(request):
    return pytest.importorskip(request.param).UnitRegistry()


@pytest.mark.parametrize("operation", [floor, floor_inplace])
@pytest.mark.parametrize(
    "case", ["scalar_rhs", "scalar_lhs", "compatible", "incompatible", "dimensionless", "zero"]
)
@pytest.mark.parametrize("array", [False, True])
def test_quantity_floor_division_retains_backend_semantics(reg, operation, case, array):
    def arguments():
        value = np.array([5.0, 9.0]) if array else 5.0
        a = reg.Quantity(value, "cm")
        b = {
            "scalar_rhs": 2,
            "scalar_lhs": reg.Quantity(value, "cm"),
            "compatible": reg.Quantity(2, "mm"),
            "incompatible": reg.Quantity(2, "s"),
            "dimensionless": 2,
            "zero": reg.Quantity(0, "cm"),
        }[case]
        if case == "scalar_lhs":
            a = 2
        if case == "dimensionless":
            a = reg.Quantity(value, "percent")
        return a, b

    fast = unit_jit(operation)
    # Backends differ on supported operators; preserve both results and errors.
    with np.errstate(divide="ignore", invalid="ignore"):
        try:
            expected = operation(*arguments())
        except Exception as original:
            with trace_execution() as trace, pytest.raises(type(original)):
                fast(*arguments())
            assert [call.path for call in trace.calls] == ["fallback"]
        else:
            with expect_execution(fast, "fallback"):
                actual = fast(*arguments())
            if hasattr(expected, "units"):
                np.testing.assert_allclose(actual.to(expected.units).magnitude, expected.magnitude)
            else:
                assert not hasattr(actual, "units")
                np.testing.assert_equal(actual, expected)


@pytest.mark.parametrize("operation", [floor, floor_inplace])
@pytest.mark.parametrize("numba", [False, True])
def test_plain_floor_division_stays_fast(operation, numba):
    if numba:
        pytest.importorskip("numba")
    fast = unit_jit(operation, use_numba=numba)
    with expect_execution(fast):
        actual = fast(np.array([5, 9]), 2)
    np.testing.assert_equal(actual, [2, 4])


def counted_floor(counter, a, b):
    counter.calls += 1
    return a // b


def test_floor_error_is_not_retried_after_side_effects():
    counter = SimpleNamespace(calls=0)
    fast = unit_jit(counted_floor)
    with trace_execution() as trace, pytest.raises(Exception):
        fast(counter, 5 * ureg.cm, 2)
    assert [call.path for call in trace.calls] == ["fallback"]
    assert counter.calls == 1


@pytest.mark.parametrize("operation", [same, different, identity_chain])
@pytest.mark.parametrize("aliased", [False, True])
@pytest.mark.parametrize("array", [False, True])
def test_identity_uses_original_quantities(reg, operation, aliased, array):
    magnitude = np.array([200.0]) if array else 200.0
    a = reg.Quantity(magnitude, "cm")
    b = a if aliased else reg.Quantity(magnitude, "cm")
    fast = unit_jit(operation)
    with expect_execution(fast, "fallback"):
        assert fast(a, b) is operation(a, b)


def test_identity_of_input_containers_is_preserved():
    value = [2 * ureg.cm]
    fast = unit_jit(same)
    with expect_execution(fast, "fallback"):
        assert fast(value, value) is True


def test_equality_still_strips_quantities(reg):
    fast = unit_jit(equal)
    with expect_execution(fast):
        assert fast(reg.Quantity(2.0, "m"), reg.Quantity(200.0, "cm"))


def total(x):
    return x.sum()


def derived_total(x):
    return (x * 2).sum()


def total_axis(x):
    return x.sum(axis=0)


@pytest.mark.parametrize("operation", [total, derived_total, total_axis])
@pytest.mark.parametrize("hook", ["sum", "_numpy_method_wrap"])
@pytest.mark.parametrize("warm", [False, True])
def test_quantity_class_override_is_used_before_and_after_compilation(
    monkeypatch, operation, hook, warm
):
    value = ureg.Quantity(np.array([100.0, 200.0]), "cm")
    fast = unit_jit(operation)
    if warm:
        with expect_execution(fast):
            fast(value)
    calls = []

    def replacement(self, *args, **kwargs):
        calls.append(self)
        return 7 * ureg.s

    with monkeypatch.context() as patch:
        patch.setattr(ureg.Quantity, hook, replacement, raising=False)
        with expect_execution(fast, "fallback"):
            assert fast(value) == 7 * ureg.s
        assert len(calls) == 1
    if warm:
        # Removing the override restores eligibility of the existing plan.
        with expect_execution(fast):
            assert fast(value) == operation(value)


@pytest.mark.parametrize("new_instance", [False, True])
def test_quantity_instance_override_invalidates_dispatch(new_instance):
    first = ureg.Quantity(np.array([100.0, 200.0]), "cm")
    fast = unit_jit(total)
    with expect_execution(fast):
        assert fast(first) == 3 * ureg.m
    value = ureg.Quantity(np.array([100.0, 200.0]), "cm") if new_instance else first
    value.sum = lambda: 7 * ureg.s
    with expect_execution(fast, "fallback"):
        assert fast(value) == 7 * ureg.s
    del value.sum
    with expect_execution(fast):
        assert fast(value) == 3 * ureg.m


@pytest.mark.parametrize("warm", [False, True])
def test_custom_quantity_class_is_not_treated_as_standard(warm):
    class CustomQuantity(ureg.Quantity):
        def sum(self):
            return 7 * ureg.s

    fast = unit_jit(total)
    if warm:
        with expect_execution(fast):
            fast(ureg.Quantity(np.array([100.0, 200.0]), "cm"))
    value = CustomQuantity(np.array([100.0, 200.0]), "cm")
    with expect_execution(fast, "fallback"):
        assert fast(value) == total(value) == 7 * ureg.s


@pytest.mark.parametrize("warm", [False, True])
def test_magnitude_method_override_retains_original_scale(warm):
    class CustomArray(np.ndarray):
        def sum(self, *args, **kwargs):
            return 7.0

    fast = unit_jit(total)
    if warm:
        with expect_execution(fast):
            fast(ureg.Quantity(np.array([100.0, 200.0]), "cm"))
    value = ureg.Quantity(np.array([100.0, 200.0]).view(CustomArray), "cm")
    assert type(value.magnitude) is CustomArray
    with expect_execution(fast, "fallback"):
        assert fast(value) == total(value) == 7 * ureg.cm


@pytest.mark.parametrize("operation", [total, derived_total])
def test_standard_quantity_methods_stay_fast_across_values_and_scales(reg, operation):
    fast = unit_jit(operation)
    for magnitude, units in [([100.0, 200.0], "cm"), ([4.0, 6.0], "m")]:
        value = reg.Quantity(np.array(magnitude), units)
        with expect_execution(fast):
            actual = fast(value)
        expected = operation(value)
        np.testing.assert_allclose(actual.to(expected.units).magnitude, expected.magnitude)


def sequence_total(values):
    return values[0].sum()


@pytest.mark.parametrize("container", [list, tuple])
def test_nested_quantity_override_uses_fallback(container):
    values = container(
        [
            ureg.Quantity(np.array([100.0, 200.0]), "cm"),
            ureg.Quantity(np.array([200.0, 300.0]), "cm"),
        ]
    )
    fast = unit_jit(sequence_total)
    with expect_execution(fast):
        assert fast(values) == 3 * ureg.m
    values[0].sum = lambda: 7 * ureg.s
    with expect_execution(fast, "fallback"):
        assert fast(values) == 7 * ureg.s
    del values[0].sum
    with expect_execution(fast):
        assert fast(values) == 3 * ureg.m


def test_override_can_be_observed_inside_a_branch():
    def choose(a, b):
        if a is b:
            return a * 2
        return a * 3

    value = 200 * ureg.cm
    fast = unit_jit(choose)
    with expect_execution(fast, "fallback"):
        assert fast(value, value) == choose(value, value) == 4 * ureg.m


@pytest.mark.parametrize("operation", [total, derived_total])
def test_numba_plan_respects_later_quantity_method_override(monkeypatch, operation):
    pytest.importorskip("numba")
    value = ureg.Quantity(np.array([100.0, 200.0]), "cm")
    fast = unit_jit(operation, use_numba=True)
    with expect_execution(fast):
        assert fast(value) == operation(value)
    monkeypatch.setattr(ureg.Quantity, "sum", lambda self: 7 * ureg.s, raising=False)
    with expect_execution(fast, "fallback"):
        assert fast(value) == operation(value) == 7 * ureg.s
