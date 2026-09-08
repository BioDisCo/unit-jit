"""Behavioral tests for the public execution observation API."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import numpy as np
import pytest
from pint import UnitRegistry

from unit_jit import QuantitySnapshot, trace_execution, unit_jit

ureg = UnitRegistry()


@unit_jit
def divide(x, t):
    return x / t


@unit_jit
def fail_fallback(x):
    return x.magnitude / 0


@unit_jit
def update(obj):
    obj.x = obj.x * 2
    return obj.x


@unit_jit
def bare(x):
    return x.magnitude


@pytest.mark.parametrize("path", ["fast", "fallback"])
def test_trace_preserves_body_exception(path):
    with trace_execution() as trace:
        with pytest.raises(ZeroDivisionError) as error:
            if path == "fast":
                divide(2 * ureg.m, 0)
            else:
                fail_fallback(2 * ureg.m)
    assert len(trace.calls) == 1
    call = trace.calls[0]
    assert call.path == path
    assert call.exception is error.value
    assert call.finished
    assert call.result is None
    # An exceptional scope must not leak tracing into later calls.
    divide(2 * ureg.m, 2)
    assert len(trace.calls) == 1


def test_nested_scopes_receive_only_their_calls():
    with trace_execution() as outer:
        divide(2 * ureg.m, 2)
        with trace_execution() as inner:
            divide(4 * ureg.m, 2)
        divide(6 * ureg.m, 2)
    assert [call.result for call in outer.calls] == [1, 2, 3]
    assert [call.result for call in inner.calls] == [2]
    assert all(call.function is divide.__wrapped__ for call in outer.calls)


def test_trace_context_resets_when_scope_raises():
    with pytest.raises(RuntimeError), trace_execution() as trace:
        divide(2 * ureg.m, 2)
        raise RuntimeError("outside the traced body")
    divide(2 * ureg.m, 2)
    assert len(trace.calls) == 1


def test_concurrent_traces_are_isolated():
    barrier = Barrier(2)

    def run(value):
        with trace_execution() as trace:
            barrier.wait(timeout=10)
            divide(value * ureg.m, 2)
        return trace

    with ThreadPoolExecutor(max_workers=2) as pool:
        traces = list(pool.map(run, [2, 6]))
    assert [len(trace.calls) for trace in traces] == [1, 1]
    assert [trace.calls[0].result for trace in traces] == [1, 3]


def test_snapshots_survive_array_mutation_and_object_restoration():
    obj = SimpleNamespace(x=np.array([100.0, 200.0]) * ureg.cm)
    with trace_execution() as trace:
        result = update(obj)
    call = trace.calls[0]
    assert call.path == "fast"
    np.testing.assert_equal(call.arguments["obj"]["x"], [1, 2])
    np.testing.assert_equal(call.result, [2, 4])
    assert hasattr(obj.x, "units")
    result.magnitude[:] = 99
    np.testing.assert_equal(call.arguments["obj"]["x"], [1, 2])
    np.testing.assert_equal(call.result, [2, 4])


def test_fallback_quantity_and_array_snapshots_preserve_original_scale():
    value = np.array([100.0, 200.0]) * ureg.cm
    with trace_execution() as trace:
        result = bare(value)
    call = trace.calls[0]
    assert call.path == "fallback"
    assert isinstance(call.arguments["x"], QuantitySnapshot)
    assert call.arguments["x"].units == ureg.cm
    value.magnitude[:] = 0
    result[:] = 0
    np.testing.assert_equal(call.arguments["x"].magnitude, [100, 200])
    np.testing.assert_equal(call.result, [100, 200])


def test_no_snapshot_work_when_tracing_is_off(monkeypatch):
    import unit_jit._execution as execution

    def forbidden(*args):
        raise AssertionError("tracing unexpectedly enabled")

    monkeypatch.setattr(execution, "_snapshot", forbidden)
    assert divide(2 * ureg.m, 2) == 1 * ureg.m


def test_guard_failure_does_not_report_body_execution():
    divide(2 * ureg.m, 2)
    with trace_execution() as trace, pytest.raises(TypeError):
        divide(2 * ureg.s, 2)
    assert trace.calls == []


@unit_jit
def return_cycle(obj):
    return obj


def test_fallback_snapshots_preserve_cycles():
    obj = SimpleNamespace()
    obj.self = obj
    with trace_execution() as trace:
        assert return_cycle(obj) is obj
    call = trace.calls[0]
    assert call.path == "fallback"
    assert call.arguments["obj"]["self"] is call.arguments["obj"]
    assert call.result["self"] is call.result


@unit_jit
def inspect_entry(x, observe):
    observe()
    return x


def test_call_is_visible_while_its_body_is_running():
    with trace_execution() as trace:

        def observe():
            assert len(trace.calls) == 1
            call = trace.calls[0]
            assert call.path == "fallback"
            assert not call.finished
            assert call.exception is None
            assert call.arguments["x"].magnitude == 2

        result = inspect_entry(2 * ureg.cm, observe)
    assert result == 2 * ureg.cm
    assert trace.calls[0].finished
    assert isinstance(trace.calls[0].result, QuantitySnapshot)
