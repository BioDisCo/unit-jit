"""Test helpers: execution evidence plus aliases for predicate API tests."""

from contextlib import contextmanager

from unit_jit import QuantitySnapshot, trace_execution
from unit_jit import is_jit_active as jit_active
from unit_jit import is_jit_disabled as jit_disabled

__all__ = ["expect_execution", "jit_active", "jit_disabled"]


def _assert_stripped(value, seen=None):
    if seen is None:
        seen = set()
    assert not isinstance(value, QuantitySnapshot), "fast execution still contains a quantity"
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, dict):
        for item in value.values():
            _assert_stripped(item, seen)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_stripped(item, seen)


@contextmanager
def expect_execution(function, path="fast"):
    """Require real execution of this function, including every call in the scope."""
    original = function.__wrapped__
    with trace_execution() as trace:
        yield trace
    calls = [call for call in trace.calls if call.function is original]
    assert calls, f"{function.__qualname__} did not execute"
    assert all(call.path == path for call in calls), [
        (call.function.__qualname__, call.path) for call in calls
    ]
    assert all(call.finished and call.exception is None for call in calls)
    if path == "fast":
        # Include nested decorated calls, so a successful outer result cannot
        # conceal either fallback or unstripped inner arguments/results.
        assert all(call.path == "fast" for call in trace.calls)
        for call in trace.calls:
            _assert_stripped(call.arguments)
            _assert_stripped(call.result)
