"""Differential regressions and liveness checks for scope and call contracts."""

from types import SimpleNamespace

import numpy as np
import pytest
from _jit_state import expect_execution
from pint import UnitRegistry

from unit_jit import trace_execution, unit_jit

ureg = UnitRegistry()


def helper(x):
    return x * 2


def parameter_helper(helper, x):
    return helper(x)


def assigned_helper(replacement, x):
    helper = replacement
    return helper(x)


def lambda_helper(x):
    helper = lambda value: 7  # noqa: E731, F841 — exercise a local lambda binding
    return helper(x)


def unbound_helper(x):
    result = helper(x)  # noqa: F823 — Python must report the local binding error
    helper = lambda value: 7  # noqa: E731, F841 — exercise a local lambda binding
    return result


def registry_parameter(ureg):
    return ureg.m


def registry_local(value):
    ureg = value
    return ureg.m


def registry_comprehension(values):
    return [ureg.m for ureg in values]


def module_parameter(np, x):
    return np.sum(x)


def builtin_parameter(min, x):
    return min(x)


def registry_conversion(ureg, x):
    return x.to(ureg.m).magnitude


def quantity_conversion(x):
    return x.to(2 * ureg.cm).magnitude


@pytest.mark.parametrize("operation", [parameter_helper, assigned_helper])
def test_local_callable_binding_wins_over_global(operation):
    fast = unit_jit(operation)
    args = (lambda value: 7, 3 * ureg.m)
    with expect_execution(fast, "fallback"):
        assert fast(*args) == operation(*args) == 7


def test_local_lambda_wins_and_stays_fast():
    fast = unit_jit(lambda_helper)
    with expect_execution(fast):
        assert fast(3 * ureg.m) == 7


def test_later_assignment_is_still_a_local_binding():
    fast = unit_jit(unbound_helper)
    with trace_execution() as trace, pytest.raises(UnboundLocalError):
        fast(3 * ureg.m)
    assert [call.path for call in trace.calls] == ["fallback"]


@pytest.mark.parametrize("operation", [registry_parameter, registry_local])
def test_local_registry_name_is_not_rewritten(operation):
    fast = unit_jit(operation)
    value = SimpleNamespace(m=42 * ureg.m)
    with expect_execution(fast):
        assert fast(value) == operation(value) == 42 * ureg.m


def test_comprehension_binding_is_not_rewritten():
    fast = unit_jit(registry_comprehension)
    values = [SimpleNamespace(m=42 * ureg.m), SimpleNamespace(m=23 * ureg.m)]
    with expect_execution(fast):
        assert fast(values) == registry_comprehension(values)


@pytest.mark.parametrize(
    "operation,value",
    [
        (module_parameter, SimpleNamespace(sum=lambda x: 7)),
        (builtin_parameter, lambda x: 7),
    ],
)
def test_local_module_and_builtin_names_do_not_use_global_contracts(operation, value):
    fast = unit_jit(operation)
    with expect_execution(fast, "fallback"):
        assert fast(value, 3 * ureg.m) == operation(value, 3 * ureg.m) == 7


def test_conversion_target_obeys_local_binding():
    fast = unit_jit(registry_conversion)
    args = (SimpleNamespace(m=ureg.cm), 2 * ureg.m)
    with expect_execution(fast, "fallback"):
        assert fast(*args) == registry_conversion(*args) == 200


def test_quantity_conversion_target_uses_units_not_magnitude():
    fast = unit_jit(quantity_conversion)
    with expect_execution(fast):
        assert fast(2 * ureg.m) == quantity_conversion(2 * ureg.m) == 200


def minimum(a, b):
    return min(a, b)


def maximum(a, b):
    return max(a, b)


def minimum_three(a, b):
    return min(a, a, b)


def sum_positional(values):
    return sum(values, 1)


def sum_keyword(values):
    return sum(values, start=1)


@pytest.mark.parametrize("operation", [minimum, maximum, minimum_three])
def test_all_selector_operands_are_checked(operation):
    fast = unit_jit(operation)
    args = (1 * ureg.m, 2 * ureg.s)
    with pytest.raises(Exception) as original:
        operation(*args)
    with trace_execution() as trace, pytest.raises(type(original.value)):
        fast(*args)
    assert [call.path for call in trace.calls] == ["fallback"]


@pytest.mark.parametrize("operation", [minimum, maximum, minimum_three])
def test_compatible_selector_operands_stay_fast(operation):
    fast = unit_jit(operation)
    args = (1 * ureg.m, 200 * ureg.cm)
    with expect_execution(fast):
        assert fast(*args) == operation(*args)


@pytest.mark.parametrize("operation", [sum_positional, sum_keyword])
def test_sum_start_does_not_erase_backend_error(operation):
    fast = unit_jit(operation)
    values = [1 * ureg.m]
    with pytest.raises(Exception) as original:
        operation(values)
    with trace_execution() as trace, pytest.raises(type(original.value)):
        fast(values)
    assert [call.path for call in trace.calls] == ["fallback"]


def sum_initial(x):
    return np.sum(x, initial=2)


def method_initial(x):
    return x.sum(initial=2)


def diff_prepend(x):
    return np.diff(x, prepend=2)


def diff_append(x):
    return np.diff(x, append=2)


def sum_dtype(x):
    return np.sum(x, dtype=int)


def method_dtype(x):
    return x.sum(dtype=int)


def sum_dtype_positional(x):
    return np.sum(x, None, int)


def diff_prepend_positional(x):
    return np.diff(x, 1, -1, 2)


def min_initial(x):
    return np.min(x, initial=2)


@pytest.fixture(params=["pint", "pintrs"])
def reg(request):
    return pytest.importorskip(request.param).UnitRegistry()


@pytest.mark.parametrize(
    "operation",
    [
        sum_initial,
        method_initial,
        diff_prepend,
        diff_append,
        sum_dtype,
        method_dtype,
        min_initial,
        sum_dtype_positional,
        diff_prepend_positional,
    ],
)
@pytest.mark.parametrize("units", ["m", "cm"])
def test_scale_sensitive_options_keep_backend_semantics(reg, operation, units):
    value = reg.Quantity(np.array([1.0, 2.0]), units)
    fast = unit_jit(operation)
    # Optional backends may reject an option; preserve that error as well.
    try:
        expected = operation(value)
    except Exception as original:
        with trace_execution() as trace, pytest.raises(type(original)):
            fast(value)
        assert [call.path for call in trace.calls] == ["fallback"]
    else:
        with expect_execution(fast, "fallback"):
            actual = fast(value)
        np.testing.assert_allclose(actual.to(expected.units).magnitude, expected.magnitude)


def sum_axis(x):
    return np.sum(x, 0, keepdims=True)


def sum_keyword_data(x):
    return np.sum(a=x, axis=0, keepdims=True)


def method_axis(x):
    return x.sum(axis=0, keepdims=True)


def diff_controls(x):
    return np.diff(x, n=1, axis=0)


@pytest.mark.parametrize("operation", [sum_axis, sum_keyword_data, method_axis, diff_controls])
def test_bound_data_and_scale_independent_controls_stay_fast(reg, operation):
    fast = unit_jit(operation)
    value = reg.Quantity(np.array([100.0, 200.0]), "cm")
    try:
        expected = operation(value)
    except TypeError:
        with trace_execution() as trace, pytest.raises(TypeError):
            fast(value)
        assert [call.path for call in trace.calls] == ["fallback"]
    else:
        with expect_execution(fast):
            actual = fast(value)
        np.testing.assert_allclose(actual.to(expected.units).magnitude, expected.magnitude)


def numba_sum_axis(x):
    return np.sum(x, axis=0)


def numba_diff_order(x):
    return np.diff(x, n=1)


@pytest.mark.parametrize("operation", [numba_sum_axis, numba_diff_order])
def test_numba_supported_controls_stay_fast(reg, operation):
    pytest.importorskip("numba")
    fast = unit_jit(operation, use_numba=True)
    value = reg.Quantity(np.array([100.0, 200.0]), "cm")
    with expect_execution(fast):
        actual = fast(value)
    expected = operation(value)
    np.testing.assert_allclose(actual.to(expected.units).magnitude, expected.magnitude)


def min_callback(values, key):
    return min(values, key=key)


def test_callback_is_not_run_during_inference():
    seen = []

    def key(value):
        seen.append(value)
        return -value.magnitude

    values = [1 * ureg.cm, 2 * ureg.cm]
    fast = unit_jit(min_callback)
    with expect_execution(fast, "fallback"):
        assert fast(values, key) == values[1]
    assert seen == values


def invalid_keyword(x):
    return np.sum(x, unknown_option=1)


def test_unbound_options_preserve_call_error():
    fast = unit_jit(invalid_keyword)
    value = np.array([1.0]) * ureg.m
    with trace_execution() as trace, pytest.raises(TypeError):
        fast(value)
    assert [call.path for call in trace.calls] == ["fallback"]


def square_root(x):
    return np.sqrt(x)


def test_rebound_known_implementation_is_not_trusted(monkeypatch):
    monkeypatch.setattr(np, "sqrt", lambda x: 7)
    fast = unit_jit(square_root)
    with expect_execution(fast, "fallback"):
        assert fast(4 * ureg.m**2) == 7


def helper_name_collision(_unit_jit_rescale_to_magnitude, x):
    return x.to(ureg.cm).magnitude


def test_conversion_rewrite_does_not_inject_shadowable_names():
    fast = unit_jit(helper_name_collision)
    with expect_execution(fast):
        assert fast(lambda *args: 7, 2 * ureg.m) == 200


def default_min(values, default):
    return min(values, default=default)


def test_selector_default_is_not_confused_with_data_operand():
    fast = unit_jit(default_min)
    for values in ([1 * ureg.m], []):
        with expect_execution(fast, "fallback"):
            assert fast(values, 2 * ureg.s) == default_min(values, 2 * ureg.s)


def array_copy(x):
    return np.copy(x)


def array_sum(x):
    return np.sum(x)


@pytest.mark.parametrize("operation", [array_copy, array_sum])
def test_plain_sequences_keep_numeric_library_output_representation(operation):
    fast = unit_jit(operation)
    with expect_execution(fast):
        actual = fast([1.0, 2.0])
    expected = operation([1.0, 2.0])
    assert type(actual) is type(expected)
    np.testing.assert_equal(actual, expected)


@pytest.mark.parametrize("operation", [array_copy, array_sum])
def test_quantity_sequence_coercion_uses_backend(operation):
    fast = unit_jit(operation)
    values = [1 * ureg.m, 2 * ureg.m]
    try:
        expected = operation(values)
    except Exception as original:
        with trace_execution() as trace, pytest.raises(type(original)):
            fast(values)
        assert [call.path for call in trace.calls] == ["fallback"]
    else:
        with expect_execution(fast, "fallback"):
            actual = fast(values)
        np.testing.assert_equal(actual, expected)


def sum_output(x, out):
    return np.sum(x, out=out)


def test_output_storage_is_not_written_in_si_units():
    fast = unit_jit(sum_output)
    value = np.array([100.0, 200.0]) * ureg.cm
    expected_out = np.array(0.0)
    actual_out = np.array(0.0)
    expected = sum_output(value, expected_out)
    with expect_execution(fast, "fallback"):
        actual = fast(value, actual_out)
    np.testing.assert_equal(actual_out, expected_out)
    assert actual.to(expected.units).magnitude == expected.magnitude


def test_captured_callable_is_not_resolved_as_global():
    def make(helper):
        def invoke(x):
            return helper(x)

        return invoke

    original = make(lambda x: 7)
    fast = unit_jit(original)
    with expect_execution(fast, "fallback"):
        assert fast(3 * ureg.m) == original(3 * ureg.m) == 7


def test_captured_registry_name_is_not_rewritten_as_global():
    def make(ureg):
        def invoke(x):
            return ureg.m * x

        return invoke

    original = make(SimpleNamespace(m=42))
    fast = unit_jit(original)
    with expect_execution(fast, "fallback"):
        assert fast(2) == original(2) == 84
