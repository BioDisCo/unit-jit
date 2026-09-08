"""Require actual stripped execution, alongside numerical/unit correctness."""

from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import pytest

from unit_jit import QuantitySnapshot, is_jit_active, trace_execution, unit_jit

Pair = namedtuple("Pair", "x t")


def assert_paths(trace, paths):
    assert [call.path for call in trace.calls] == paths
    assert all(call.finished and call.exception is None for call in trace.calls)


def divide(x, t):
    return x / t


def indexed(x, t):
    return x[int(0 + 1)] / t


def reduction(x, t):
    return x.sum() / t


def sequence_sum(x, t):
    return sum(x) / t


def fields(x, t):
    return x.x / x.t / t


def branches(x, t):
    if t > 0:
        result = x / t
    else:
        result = x / (t - 1)
    return result


def loop(x, t):
    result = x
    for i in range(t):
        result = result + x
    return result


def comprehension(x, t):
    return [x / t for i in range(t)]


def base_magnitude(x, t):
    return x.to_base_units().magnitude / t


@pytest.fixture(params=["pint", "pintrs"])
def reg(request):
    return pytest.importorskip(request.param).UnitRegistry()


@pytest.mark.parametrize("numba", [False, True], ids=["python", "numba"])
@pytest.mark.parametrize("operation", [divide, indexed, reduction, branches, loop, base_magnitude])
def test_numeric_hot_paths_really_strip(reg, numba, operation):
    if numba:
        pytest.importorskip("numba")
    array = operation in (indexed, reduction)
    magnitude = np.array([100.0, 200.0, 300.0]) if array else 150.0
    x = reg.Quantity(magnitude, "cm")
    fast = unit_jit(operation, use_numba=numba)
    # Verify both first-call and cached execution, including a compatible scale change.
    for value in (x, reg.Quantity(magnitude * 10, "mm")):
        expected = operation(value, 2)
        with trace_execution() as trace:
            actual = fast(value, 2)
        assert_paths(trace, ["fast"])
        np.testing.assert_allclose(trace.calls[0].arguments["x"], value.to_base_units().magnitude)
        if hasattr(expected, "units"):
            np.testing.assert_allclose(actual.to(expected.units).magnitude, expected.magnitude)
            np.testing.assert_allclose(trace.calls[0].result, expected.to_base_units().magnitude)
        else:
            np.testing.assert_allclose(actual, expected)
            np.testing.assert_allclose(trace.calls[0].result, expected)
        if numba:
            import unit_jit as runtime

            plan = runtime._states[runtime._state_key(fast)]
            assert plan.fast.nopython_signatures


@pytest.mark.parametrize("kind", ["list", "namedtuple", "object", "comprehension"])
def test_structured_hot_paths_really_strip(reg, kind):
    q = reg.Quantity(150.0, "cm")
    if kind == "list":
        operation, x, stripped = sequence_sum, [q, q], [1.5, 1.5]
    elif kind == "namedtuple":
        operation, x, stripped = fields, Pair(q, 2), [1.5, 2]
    elif kind == "object":
        operation, x, stripped = fields, SimpleNamespace(x=q, t=2), {"x": 1.5, "t": 2}
    else:
        operation, x, stripped = comprehension, q, 1.5
    fast = unit_jit(operation)
    with trace_execution() as trace:
        actual = fast(x, 2)
    assert_paths(trace, ["fast"])
    if kind == "object":
        assert trace.calls[0].arguments["x"] == stripped
        assert hasattr(x.x, "units")  # caller-visible fields restored afterward
    else:
        np.testing.assert_equal(trace.calls[0].arguments["x"], stripped)
    expected = operation(x, 2)
    for a, e in zip(
        actual if isinstance(actual, list) else [actual],
        expected if isinstance(expected, list) else [expected],
        strict=True,
    ):
        assert a.to(e.units).magnitude == pytest.approx(e.magnitude)


def bare_magnitude(x):
    return x.magnitude


def inplace(x, value):
    x *= value
    return x


def identity(values):
    return values


def test_fallback_really_executes_with_quantities(reg):
    fast = unit_jit(bare_magnitude)
    with trace_execution() as trace:
        result = fast(reg.Quantity(150, "cm"))
    assert_paths(trace, ["fallback"])
    assert isinstance(trace.calls[0].arguments["x"], QuantitySnapshot)
    assert trace.calls[0].arguments["x"].magnitude == 150
    assert result == 150


def test_active_plan_can_fall_back_for_one_call(reg):
    fast = unit_jit(inplace)
    with trace_execution() as trace:
        fast(reg.Quantity(1, "m"), 2)
    assert_paths(trace, ["fast"])
    assert is_jit_active(fast)
    x = reg.Quantity(np.array([100.0, 200.0]), "cm")
    expected = reg.Quantity(np.array([100.0, 200.0]), "cm")
    reference = inplace(expected, 2)
    with trace_execution() as trace:
        actual = fast(x, 2)
    assert_paths(trace, ["fallback"])
    assert isinstance(trace.calls[0].arguments["x"], QuantitySnapshot)
    np.testing.assert_allclose(x.magnitude, expected.magnitude)
    np.testing.assert_allclose(actual.to(reference.units).magnitude, reference.magnitude)
    assert is_jit_active(fast)  # A cached plan alone does not prove this call used it.
    with trace_execution() as trace:
        fast(reg.Quantity(3, "m"), 2)
    assert_paths(trace, ["fast"])


def test_nonempty_reduction_stays_fast_across_lengths(reg):
    fast = unit_jit(sequence_sum)
    for count in (2, 5, 0, 1):
        values = [reg.Quantity(100.0, "cm") for _ in range(count)]
        with trace_execution() as trace:
            actual = fast(values, 2)
        if count:
            assert_paths(trace, ["fast"])
            assert trace.calls[0].arguments["x"] == [1.0] * count
            assert actual.to("m").magnitude == count / 2
        else:
            assert_paths(trace, ["fallback"])
            assert actual == 0


@unit_jit
def inner(x, t):
    return x / t


def outer(x, t):
    return inner(x, t)


def with_default(x, /, *, t=2):
    return x / t


@pytest.mark.parametrize("operation", [outer, with_default])
def test_nested_calls_and_defaults_really_strip(reg, operation):
    fast = unit_jit(operation)
    args = (reg.Quantity(150.0, "cm"), 2) if operation is outer else (reg.Quantity(150.0, "cm"),)
    with trace_execution() as trace:
        actual = fast(*args)
    expected_paths = ["fast", "fast"] if operation is outer else ["fast"]
    assert_paths(trace, expected_paths)
    assert trace.calls[0].arguments == {"x": 1.5, "t": 2}
    assert trace.calls[0].result == 0.75
    assert actual.to("m").magnitude == 0.75


@pytest.mark.parametrize("backend", ["pint", "pintrs"])
@pytest.mark.parametrize("count", [0, 1, 4])
def test_registry_literal_loop_really_strips(monkeypatch, backend, count):
    # Reuse a module-global function so both backends follow the literal rewrite path.
    registry = pytest.importorskip(backend).UnitRegistry()
    monkeypatch.setitem(globals(), "literal_registry", registry)
    fast = unit_jit(literal_loop)
    with trace_execution() as trace:
        actual = fast(count)
    assert_paths(trace, ["fast"])
    assert trace.calls[0].arguments == {"count": count}
    assert trace.calls[0].result == pytest.approx(0.05 if count else 0)
    assert actual.to("m/s").magnitude == pytest.approx(0.05 if count else 0)


def literal_loop(count):
    value = 0 * literal_registry.cm / literal_registry.s
    for i in range(count):
        value = 10 * literal_registry.cm / (2 * literal_registry.s)
    return value


literal_registry = None


def distinct_registry_literals():
    return 1 * literal_registry.m / other_literal_registry.s


other_literal_registry = None


def test_unit_interning_does_not_merge_distinct_registry_origins(monkeypatch):
    from pint import UnitRegistry

    monkeypatch.setitem(globals(), "literal_registry", UnitRegistry())
    monkeypatch.setitem(globals(), "other_literal_registry", UnitRegistry())
    with pytest.raises(ValueError):
        distinct_registry_literals()
    fast = unit_jit(distinct_registry_literals)
    with trace_execution() as trace, pytest.raises(ValueError):
        fast()
    assert len(trace.calls) == 1
    assert trace.calls[0].path == "fallback"
    assert isinstance(trace.calls[0].exception, ValueError)
