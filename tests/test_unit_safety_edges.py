"""Boundary, indexing, mutation and fallback checks against the original backend."""

from types import SimpleNamespace

import numpy as np
import pytest
from _jit_state import expect_execution
from pint import DimensionalityError, UnitRegistry

import unit_jit as runtime
from unit_jit import is_jit_disabled, trace_execution, unit_jit

ureg = UnitRegistry()
gain = 2


@pytest.fixture(autouse=True)
def fresh_inference(monkeypatch):
    prefix = f"{__name__}::"
    monkeypatch.setattr(
        runtime, "_states", {k: v for k, v in runtime._states.items() if not k.startswith(prefix)}
    )


@pytest.fixture(params=["pint", "pintrs"])
def backend_registry(request):
    return pytest.importorskip(request.param).UnitRegistry()


@pytest.fixture(params=[False, True], ids=["python", "numba"])
def jit(request):
    if request.param:
        pytest.importorskip("numba")
    return lambda func: unit_jit(func, use_numba=request.param)


def assert_same(actual, expected):
    if hasattr(expected, "units"):
        assert hasattr(actual, "units")
        assert actual.dimensionality == expected.dimensionality
        np.testing.assert_allclose(actual.to(expected.units).magnitude, expected.magnitude)
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for item, reference in zip(actual, expected, strict=True):
            assert_same(item, reference)
    else:
        assert not hasattr(actual, "units")
        np.testing.assert_equal(actual, expected)


def write_element(x, value):
    x[0] = value
    return x


def write_slice(x, value):
    x[::2] = value
    return x


def add_inplace(x, value):
    x += value
    return x


@pytest.mark.parametrize("operation", [write_element, write_slice, add_inplace])
@pytest.mark.parametrize("units", ["m", "cm"])
def test_array_argument_mutations_match_backend(backend_registry, jit, operation, units):
    reg = backend_registry
    actual = reg.Quantity(np.array([100.0, 200.0, 300.0]), units)
    expected = reg.Quantity(np.array([100.0, 200.0, 300.0]), units)
    value = reg.Quantity(3, "m")
    fast = jit(operation)
    with expect_execution(fast, "fallback"):
        result = fast(actual, value)
    reference = operation(expected, value)
    assert_same(actual, expected)
    assert_same(result, reference)


def write_alias(x, alias, value):
    x[0] = value
    return alias


def test_array_views_and_aliases_preserve_effects(backend_registry, jit):
    reg = backend_registry
    actual = reg.Quantity(np.array([100.0, 200.0, 300.0]), "cm")
    expected = reg.Quantity(np.array([100.0, 200.0, 300.0]), "cm")
    view, ref_view = actual[::2], expected[::2]
    value = reg.Quantity(3, "m")
    result = jit(write_alias)(view, view, value)
    reference = write_alias(ref_view, ref_view, value)
    assert_same(actual, expected)
    assert_same(result, reference)
    assert result is view


def test_scalar_warmup_does_not_authorize_array_inplace_mutation():
    fast = unit_jit(add_inplace)
    with expect_execution(fast):
        assert_same(fast(1 * ureg.m, 2 * ureg.m), 3 * ureg.m)
    actual = np.array([100.0, 200.0]) * ureg.cm
    expected = actual.copy()
    with expect_execution(fast, "fallback"):
        result = fast(actual, 3 * ureg.m)
    reference = add_inplace(expected, 3 * ureg.m)
    assert_same(actual, expected)
    assert_same(result, reference)


def read_index(x, a, b):
    return x[int(a + b)]


def write_index(x, a, b):
    x[int(a + b)] = 99
    return x


def read_slice(x, a, b):
    return x[: int(a + b)]


def write_index_slice(x, a, b):
    x[: int(a + b)] = 99
    return x


@pytest.mark.parametrize("operation", [read_index, write_index, read_slice, write_index_slice])
def test_index_expressions_do_not_erase_errors(backend_registry, jit, operation):
    reg = backend_registry
    args = (np.array([10, 20, 30]), reg.Quantity(1, "m"), reg.Quantity(1, "s"))
    with pytest.raises(Exception) as reference:
        operation(*args)
    actual = args[0].copy()
    with pytest.raises((TypeError, type(reference.value))):
        jit(operation)(actual, *args[1:])
    np.testing.assert_array_equal(actual, args[0])


def array_ratio(x, t):
    return x / t


def test_readonly_arrays_still_compile(backend_registry, jit):
    reg = backend_registry
    x = reg.Quantity(np.array([100.0, 200.0]), "cm")
    t = reg.Quantity(2, "s")
    fast = jit(array_ratio)
    with expect_execution(fast):
        assert_same(fast(x, t), array_ratio(x, t))


def list_identity(values):
    return values


def sum_values(values):
    return sum(values)


def nested_values(x, count):
    return [[x] * count, []]


def filtered_values(x, count):
    return [x for i in range(count) if i > 0]


@pytest.mark.parametrize("operation", [list_identity, sum_values])
@pytest.mark.parametrize("empty_first", [False, True])
def test_empty_and_nonempty_inputs_preserve_result_type(operation, empty_first):
    fast = unit_jit(operation)
    cases = [[], [1 * ureg.m], [2 * ureg.m, 3 * ureg.m]]
    if not empty_first:
        cases.reverse()
    for values in cases:
        assert_same(fast(values), operation(values))


@pytest.mark.parametrize("operation", [nested_values, filtered_values])
def test_nested_or_filtered_outputs_can_be_empty(operation):
    fast = unit_jit(operation)
    for count in (3, 0, 1, 5):
        args = (2 * ureg.cm, count)
        assert_same(fast(*args), operation(*args))


def global_helper(x):
    return x * 2


def call_global(x):
    return global_helper(x)


def global_gain(x):
    return x * gain


@pytest.mark.parametrize("which", ["helper", "constant"])
def test_rebound_globals_invalidate_plan(monkeypatch, which):
    operation = call_global if which == "helper" else global_gain
    fast = unit_jit(operation)
    with expect_execution(fast):
        assert_same(fast(3 * ureg.m), 6 * ureg.m)
    if which == "helper":
        monkeypatch.setitem(globals(), "global_helper", lambda x: x / ureg.s)
    else:
        monkeypatch.setitem(globals(), "gain", 2 * ureg.s)
    with expect_execution(fast, "fallback"):
        assert_same(fast(3 * ureg.m), operation(3 * ureg.m))


class Model:
    def rate(self, x):
        return x * 2

    def run(self, x):
        return self.rate(x)


def test_instance_override_uses_original_dispatch():
    model = Model()
    fast = unit_jit(Model.run)
    with expect_execution(fast):
        assert_same(fast(model, 3 * ureg.m), 6 * ureg.m)
    model.rate = lambda x: x / ureg.s
    with expect_execution(fast, "fallback"):
        assert_same(fast(model, 3 * ureg.m), model.run(3 * ureg.m))


def test_redefinitions_do_not_reuse_another_functions_plan():
    def make_function(scale):
        def compute(x, factor=scale):
            return x * factor

        return unit_jit(compute)

    old = make_function(2)
    assert_same(old(3 * ureg.m), 6 * ureg.m)
    new = make_function(2 * ureg.s)
    assert_same(new(3 * ureg.m), 6 * ureg.m * ureg.s)
    assert_same(old(3 * ureg.m), 6 * ureg.m)


def absolute_value(x):
    return abs(x)


def rounded_value(x):
    return round(x)


@pytest.mark.parametrize(
    "units,value", [("degC", -2.0), ("dB", -3.0), ("percent", 12.5), ("cm", 12.5)]
)
@pytest.mark.parametrize("operation", [absolute_value, rounded_value])
def test_scale_sensitive_operations_match_pint(operation, units, value):
    quantity = ureg.Quantity(value, units)
    assert_same(unit_jit(operation)(quantity), operation(quantity))


def update_pair(pair, divisor):
    pair.x *= 2
    pair.y *= 3
    return pair.x / pair.y / divisor


@pytest.mark.parametrize("divisor", [1, 0])
def test_restoration_failure_does_not_skip_other_fields(monkeypatch, divisor):
    fast = unit_jit(update_pair)
    with expect_execution(fast):
        fast(SimpleNamespace(x=1 * ureg.m, y=2 * ureg.s), 1)
    pair = SimpleNamespace(x=1 * ureg.m, y=2 * ureg.s)
    quantity = ureg.Quantity

    def construct(value, units):
        if units == ureg.m:
            raise RuntimeError("injected restoration failure")
        return quantity(value, units)

    monkeypatch.setattr(ureg, "Quantity", construct)
    with pytest.raises(ExceptionGroup, match="restore") as caught:
        fast(pair, divisor)
    assert str(caught.value.exceptions[0]) == "injected restoration failure"
    assert_same(pair.y, 6 * ureg.s)
    assert not runtime._in_fast_zone()
    if divisor == 0:
        assert isinstance(caught.value.__context__, ZeroDivisionError)


def positional_and_keyword(d, /, *, t=2 * ureg.s):
    return d / t


def variadic(d, *times):
    return d / times[0]


def keyword_variadic(d, **options):
    return d / options["t"]


def nested_default(d):
    return positional_and_keyword(d)


@pytest.mark.parametrize(
    "operation,args,kwargs",
    [
        (positional_and_keyword, (6 * ureg.m,), {}),
        (positional_and_keyword, (6 * ureg.m,), {"t": 3 * ureg.s}),
        (variadic, (6 * ureg.m, 3 * ureg.s), {}),
        (keyword_variadic, (6 * ureg.m,), {"t": 3 * ureg.s}),
        (nested_default, (6 * ureg.m,), {}),
    ],
)
def test_binding_combinations_match_pint(operation, args, kwargs):
    fast = unit_jit(operation)
    assert_same(fast(*args, **kwargs), operation(*args, **kwargs))
    assert_same(fast(*args, **kwargs), operation(*args, **kwargs))


def test_keyword_only_default_is_guarded():
    fast = unit_jit(positional_and_keyword)
    fast(6 * ureg.m)
    with pytest.raises(TypeError):
        fast(6 * ureg.m, t=3 * ureg.m)


def test_positional_only_constraint_is_preserved():
    with pytest.raises(TypeError):
        unit_jit(positional_and_keyword)(d=6 * ureg.m)


def test_variadic_dimensions_are_guarded():
    fast = unit_jit(variadic)
    fast(6 * ureg.m, 2 * ureg.s)
    with pytest.raises(TypeError):
        fast(6 * ureg.m, 2 * ureg.m)


def bounded_index(x, i):
    return x[i]


def test_numba_preserves_index_bounds_errors():
    pytest.importorskip("numba")
    fast = unit_jit(bounded_index, use_numba=True)
    with pytest.raises(IndexError):
        fast(np.array([1.0, 2.0]), 2)


@pytest.mark.parametrize("omitted_kind", ["Subscript", "Call", "BinaryOperation"])
def test_expression_audit_rejects_incomplete_inference(monkeypatch, omitted_kind):
    """A future handler forgetting any subtree must fail closed."""
    import libcst as cst

    from unit_jit._inferrer import _UnitInferrer
    from unit_jit._values import PLAIN

    original = _UnitInferrer._expr_impl

    def incomplete(self, node):
        if isinstance(node, getattr(cst, omitted_kind)):
            return PLAIN
        return original(self, node)

    monkeypatch.setattr(_UnitInferrer, "_expr_impl", incomplete)
    fast = unit_jit(read_index)
    with trace_execution() as trace, pytest.raises(DimensionalityError):
        fast(np.arange(5), 1 * ureg.m, 1 * ureg.s)
    assert [call.path for call in trace.calls] == ["fallback"]
    assert is_jit_disabled(fast)


def test_checked_index_expression_stays_fast(jit):
    fast = jit(read_index)
    with expect_execution(fast):
        assert fast(np.arange(5), 1, 1) == 2


def keyword_sum(x):
    return np.sum(a=x)


def test_quantity_keyword_operand_keeps_units():
    fast = unit_jit(keyword_sum)
    value = ureg.Quantity(np.arange(4), "cm")
    assert_same(fast(value), keyword_sum(value))


def magnitude_norm(x):
    return float(np.sqrt(np.dot(x.magnitude, x.magnitude)))


def test_bare_magnitude_retains_backend_scale(backend_registry, jit):
    value = backend_registry.Quantity(np.array([300.0, 400.0]), "cm")
    fast = jit(magnitude_norm)
    with expect_execution(fast, "fallback"):
        assert fast(value) == magnitude_norm(value) == 500.0
    assert is_jit_disabled(fast)


def read_lower_bound(x, a, b):
    return x[int(a + b) :]


def read_step(x, a, b):
    return x[:: int(a + b)]


def read_multiple_indices(x, a, b):
    return x[0, int(a + b)]


@pytest.mark.parametrize("operation", [read_lower_bound, read_step, read_multiple_indices])
def test_all_slice_and_index_positions_are_checked(backend_registry, jit, operation):
    reg = backend_registry
    args = (np.arange(9).reshape(3, 3), reg.Quantity(1, "m"), reg.Quantity(1, "s"))
    with pytest.raises(Exception) as reference:
        operation(*args)
    with pytest.raises((TypeError, type(reference.value))):
        jit(operation)(*args)
