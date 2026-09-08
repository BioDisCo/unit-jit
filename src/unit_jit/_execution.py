"""Opt-in observation of actual function-body execution."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from inspect import BoundArguments
from typing import Any, Literal

import numpy as np

from ._inferrer import _QUANTITY_TYPES


@dataclass(frozen=True)
class QuantitySnapshot:
    """A quantity's magnitude and units captured before later mutation."""

    magnitude: Any
    units: Any


@dataclass
class ExecutionCall:
    """One body invocation, appended at entry and completed on return or raise.

    Fast arguments/results are captured after stripping and before restoration or
    output wrapping. Fallback arguments/results retain quantity snapshots.
    Objects are represented by dictionaries of their instance attributes.
    """

    function: Callable[..., Any]
    path: Literal["fast", "fallback"]
    arguments: dict[str, Any]
    result: Any = None
    exception: BaseException | None = None
    finished: bool = False


@dataclass
class ExecutionTrace:
    """Calls in entry order, including nested decorated calls."""

    calls: list[ExecutionCall]


_traces: ContextVar[tuple[ExecutionTrace, ...]] = ContextVar("unit_jit_traces", default=())


def _snapshot(value: Any, memo: dict[int, Any] | None = None) -> Any:
    if memo is None:
        memo = {}
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, _QUANTITY_TYPES):
        result = QuantitySnapshot(_snapshot(value.magnitude, memo), value.units)
    elif isinstance(value, np.ndarray):
        result = np.asarray(value).copy()
    elif isinstance(value, dict):
        result = {}
        memo[id(value)] = result
        result.update((key, _snapshot(item, memo)) for key, item in value.items())
    elif isinstance(value, (list, tuple)):
        # Lists also represent tuples, avoiding reconstruction of arbitrary
        # user-defined tuple subclasses and preserving cycles through objects.
        result = []
        memo[id(value)] = result
        result.extend(_snapshot(item, memo) for item in value)
    elif hasattr(value, "__dict__") and not callable(value):
        result = {}
        memo[id(value)] = result
        result.update((key, _snapshot(item, memo)) for key, item in vars(value).items())
    else:
        return value
    memo[id(value)] = result
    return result


@contextmanager
def trace_execution() -> Iterator[ExecutionTrace]:
    """Capture actual JIT/fallback body calls in the current execution context.

    Nested trace scopes each receive calls made within their scope. New threads
    have independent contexts. Capturing arrays/objects costs memory and time;
    leave tracing off for benchmarks. No call is recorded if binding, inference
    or argument preparation fails before a body is entered.

    Snapshots copy numeric arrays and recursively capture containers/instance
    attributes; quantities become QuantitySnapshot objects. Other opaque values
    are retained by reference. Exceptions are recorded and re-raised unchanged.
    """
    trace = ExecutionTrace([])
    token = _traces.set((*_traces.get(), trace))
    try:
        yield trace
    finally:
        _traces.reset(token)


def _invoke(
    original: Callable[..., Any],
    implementation: Callable[..., Any],
    bound: BoundArguments,
    path: Literal["fast", "fallback"],
) -> Any:
    traces = _traces.get()
    if not traces:
        return implementation(*bound.args, **bound.kwargs)
    call = ExecutionCall(original, path, _snapshot(bound.arguments))
    for trace in traces:
        trace.calls.append(call)
    try:
        result = implementation(*bound.args, **bound.kwargs)
    except BaseException as exc:
        call.exception = exc
        call.finished = True
        raise
    call.result = _snapshot(result)
    call.finished = True
    return result
