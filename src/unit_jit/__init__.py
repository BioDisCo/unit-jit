"""JIT unit-stripping decorator for Pint-annotated Python.

All @unit_jit functions in the same module are rewritten together on the
first call to any of them. Pint Quantities are converted to SI floats at
the outermost boundary; inner @unit_jit calls within the fast zone skip
conversion entirely.

On the first call, unit inference runs abstract interpretation over the
function's CST, propagating Pint units symbolically through all branches.
Dimensional errors (e.g. adding meters to seconds) are caught at this
point. If inference fails (source unavailable, parse error), the original
function is marked as JIT-disabled and runs as plain Pint on every call.

Rewrites applied inside the fast zone:
  - x.magnitude         -> x
  - x.to_base_units()   -> x
  - cast("Quantity", x) -> x
  - ureg.UNIT           -> SI float (e.g. ureg.s -> 1.0, ureg.cm -> 0.01)
  - arithmetic unchanged (works identically for floats)

Quantity attributes on objects (e.g. self.params.alpha) are handled via
in-place stripping: at the outermost boundary, all Pint Quantity attrs are
replaced with their SI magnitudes directly on the original object.  After
the call the units are restored, re-wrapping the (now updated) float arrays.
This makes stateful simulations work correctly: mutations inside the JIT
loop go to the original object and are visible after the call.
"""

import inspect
import logging
import textwrap
import threading
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, overload

import libcst as cst
import numpy as np
from pint import UnitRegistry

from unit_jit._admission import CallableBinding
from unit_jit._dispatch import DispatchBinding
from unit_jit._execution import (
    ExecutionCall as ExecutionCall,
)
from unit_jit._execution import (
    ExecutionTrace as ExecutionTrace,
)
from unit_jit._execution import (
    QuantitySnapshot as QuantitySnapshot,
)
from unit_jit._execution import (
    _invoke,
)
from unit_jit._execution import (
    trace_execution as trace_execution,
)
from unit_jit._inferrer import (  # noqa: E402
    _QUANTITY_DISPATCH,
    _QUANTITY_TYPES,
    _REGISTRY_TYPES,
    _SENTINEL,
    _UNIT_TYPES,
    _UNKNOWN,  # noqa: F401 (re-exported for tests)
    SequenceValue,
    _Binding,
    _InferenceContext,
    _strip_decorators,
    _Unsupported,
    infer_return_units,
)
from unit_jit._schema import argument_schema, check_schema
from unit_jit._scope import LexicalBindings, conversion_scale
from unit_jit._values import PLAIN, AbstractValue, QuantityValue

_log = logging.getLogger(__name__)

_fast_zone = threading.local()
_registry: dict[str, list[Callable[..., Any]]] = defaultdict(list)
_compiled: dict[str, dict[Callable[..., Any], Callable[..., Any]]] = {}
_rewritten_src: dict[str, str] = {}  # qualname -> rewritten source
_use_numba: set[Callable[..., Any]] = set()  # functions for which numba.jit should be applied

# Sentinel used in the saved list to signal a NamedTuple restore (see _strip_inplace).
_NT_RESTORE: object = object()


def _in_fast_zone() -> bool:
    return getattr(_fast_zone, "active", False)


def _ureg_si_magnitude(ureg_instance: UnitRegistry, unit_name: str) -> float | None:
    """Return the SI base-unit magnitude of ureg_instance.<unit_name>, or None on failure.

    Pint raises a broad range of exception types (UndefinedUnitError, DimensionalityError,
    AttributeError, …) for unknown or ill-formed unit names, so the bare except is intentional.
    """
    try:
        return float((1 * getattr(ureg_instance, unit_name)).to_base_units().magnitude)
    except Exception:  # noqa: BLE001
        return None


# CST transformer


class _QuantityStripper(cst.CSTTransformer):
    """Strip unit-aware Quantity syntax into float operations for the fast zone."""

    def __init__(
        self,
        ureg_vars: dict[str, UnitRegistry],
        *,
        strip_cast: bool = True,
        nonlocals: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self._ureg_vars = ureg_vars
        self._strip_cast = strip_cast
        self._bindings = None
        self._nonlocals = nonlocals

    def visit_Module(self, node: cst.Module) -> None:
        self._bindings = LexicalBindings(node, self._nonlocals)

    def leave_Attribute(
        self, original_node: cst.Attribute, updated_node: cst.Attribute
    ) -> cst.BaseExpression:
        if updated_node.attr.value == "magnitude":
            # x.to(ureg.UNIT).magnitude -> x / SI_scale(UNIT)
            if (
                isinstance(updated_node.value, cst.Call)
                and isinstance(updated_node.value.func, cst.Attribute)
                and updated_node.value.func.attr.value == "to"
                and len(updated_node.value.args) == 1
            ):
                scale = conversion_scale(
                    original_node.value.args[0].value, self._ureg_vars, self._bindings
                )
                if scale is not None:
                    return cst.BinaryOperation(
                        left=updated_node.value.func.value,
                        operator=cst.Divide(),
                        right=cst.Float(repr(scale)),
                        lpar=[cst.LeftParen()],
                        rpar=[cst.RightParen()],
                    )
            return updated_node.value
        # ureg.UNIT -> SI float (e.g. ureg.s -> 1.0, ureg.cm -> 0.01)
        if (
            isinstance(updated_node.value, cst.Name)
            and self._bindings is not None
            and self._bindings.external(original_node.value)
        ):
            ureg_instance = self._ureg_vars.get(updated_node.value.value)
            if ureg_instance is not None:
                si_val = _ureg_si_magnitude(ureg_instance, updated_node.attr.value)
                if si_val is not None:
                    return cst.Float(repr(si_val))
        return updated_node

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.BaseExpression:
        # x.to_base_units() -> x
        if (
            isinstance(updated_node.func, cst.Attribute)
            and updated_node.func.attr.value == "to_base_units"
            and not updated_node.args
        ):
            return updated_node.func.value

        # cast("Quantity", x) -> x
        if (
            self._strip_cast
            and self._bindings is not None
            and self._bindings.external(original_node.func)
            and isinstance(updated_node.func, cst.Name)
            and updated_node.func.value == "cast"
            and len(updated_node.args) == 2
            and isinstance(updated_node.args[0].value, cst.SimpleString)
            and "Quantity" in updated_node.args[0].value.value
        ):
            return updated_node.args[1].value

        return updated_node


# Boundary helpers


def _restore_inplace(saved: list[tuple[Any, str, Any, Any]]) -> None:
    """Restore every journaled field, including after preparation fails."""
    errors: list[Exception] = []
    for obj, name, registry, units in reversed(saved):
        try:
            if registry is _NT_RESTORE:
                obj.__dict__[name] = units
            else:
                obj.__dict__[name] = registry.Quantity(obj.__dict__[name], units)
        except Exception as exc:
            errors.append(exc)
    if errors:
        raise ExceptionGroup("failed to restore JIT argument fields", errors)


def _base_quantity(arg: Any) -> Any:
    """Shared conversion boundary for direct quantities and object fields."""
    return arg.to_base_units()


def _prepare_arg(
    arg: Any,
    stripped: list[tuple[Any, str, Any, Any]],
    memo: dict[int, Any] | None = None,
) -> Any:
    """Convert one argument graph, journaling changes before recursive preparation."""
    if memo is None:
        memo = {}
    if isinstance(arg, _QUANTITY_TYPES):
        return _base_quantity(arg).magnitude
    if isinstance(arg, (int, float, bool, str, bytes, type(None))) or callable(arg):
        return arg
    if isinstance(arg, (np.ndarray, np.random.Generator)):
        return arg
    if id(arg) in memo:
        return memo[id(arg)]
    if isinstance(arg, list):
        result: list[Any] = []
        memo[id(arg)] = result
        result.extend(_prepare_arg(el, stripped, memo) for el in arg)
        return result
    if isinstance(arg, tuple):
        # A tuple -> object -> same tuple cycle is broken by pre-registering the
        # original tuple. Its object's fields are prepared independently below.
        memo[id(arg)] = arg
        fields = [_prepare_arg(el, stripped, memo) for el in arg]
        result_tuple = type(arg)._make(fields) if hasattr(type(arg), "_fields") else tuple(fields)
        memo[id(arg)] = result_tuple
        return result_tuple
    memo[id(arg)] = arg
    for name, value in list(getattr(arg, "__dict__", {}).items()):
        if isinstance(value, _QUANTITY_TYPES):
            base = _base_quantity(value)
            stripped.append((arg, name, value._REGISTRY, base.units))
            arg.__dict__[name] = base.magnitude
        elif isinstance(value, (tuple, list)):
            stripped.append((arg, name, _NT_RESTORE, value))
            arg.__dict__[name] = _prepare_arg(value, stripped, memo)
        else:
            _prepare_arg(value, stripped, memo)
    return arg


@contextmanager
def _prepared(bound: inspect.BoundArguments):
    """Own the conversion memo and rollback journal for one body invocation."""
    journal: list[tuple[Any, str, Any, Any]] = []
    try:
        memo: dict[int, Any] = {}
        for name, value in bound.arguments.items():
            bound.arguments[name] = _prepare_arg(value, journal, memo)
        yield
    finally:
        _restore_inplace(journal)


def _wrap(result: Any, unit_info: AbstractValue, wrap_ureg: UnitRegistry | None) -> Any:
    """Wrap a float/array result back into a Quantity using cached SI units.

    Nested structures retain their per-field schemas. Only explicitly homogeneous
    variable-length containers repeat an inferred element unit.
    """
    if unit_info is PLAIN:
        return result
    if isinstance(unit_info, SequenceValue):
        n = len(result)
        units = unit_info.units
        if unit_info.repeated:
            units = units * n
        elif len(units) != n:
            raise RuntimeError("JIT return shape differs from its verified schema")
        wrapped = [_wrap(r, u, wrap_ureg) for r, u in zip(result, units)]
        if unit_info.kind == "namedtuple" and unit_info.cls is not None:
            return unit_info.cls._make(wrapped)  # type: ignore[attr-defined]
        return wrapped if unit_info.kind == "list" else tuple(wrapped)
    if not isinstance(unit_info, QuantityValue) or wrap_ureg is None:
        raise RuntimeError("unverified return representation")
    return wrap_ureg.Quantity(result, unit_info.unit)


# Compilation


def _compile_module(module_name: str) -> None:
    """Rewrite all @unit_jit functions from a module at once."""
    funcs = _registry[module_name]
    module_globals = funcs[0].__globals__
    ureg_vars = {k: v for k, v in module_globals.items() if isinstance(v, _REGISTRY_TYPES)}
    fast: dict[Callable[..., Any], Callable[..., Any]] = {}

    for func in funcs:
        try:
            src = inspect.getsource(func)
            src = textwrap.dedent(src)
            src = _strip_decorators(src)
            tree = cst.parse_module(src)
            # Boundary binding supplies every default. Re-evaluating defaults or
            # annotations here could execute arbitrary user code a second time.
            definition = tree.body[0]
            if not isinstance(definition, cst.FunctionDef):
                raise SyntaxError("expected a function definition")

            def parameter(param: cst.Param) -> cst.Param:
                return param.with_changes(
                    default=None, equal=cst.MaybeSentinel.DEFAULT, annotation=None
                )

            params = definition.params
            definition = definition.with_changes(
                returns=None,
                params=params.with_changes(
                    params=[parameter(p) for p in params.params],
                    posonly_params=[parameter(p) for p in params.posonly_params],
                    kwonly_params=[parameter(p) for p in params.kwonly_params],
                    star_arg=parameter(params.star_arg)
                    if isinstance(params.star_arg, cst.Param)
                    else params.star_arg,
                    star_kwarg=parameter(params.star_kwarg) if params.star_kwarg else None,
                ),
            )
            tree = tree.with_changes(body=[definition])
            stripper = _QuantityStripper(ureg_vars, nonlocals=func.__code__.co_freevars)
            new_src = tree.visit(stripper).code
            namespace: dict[str, Any] = {}
            exec(new_src, module_globals, namespace)
            rewritten = namespace[func.__name__]
            if func in _use_numba:
                try:
                    import numba as _numba  # lazy: only when use_numba=True
                except ImportError:
                    _log.warning(
                        "numba not installed; '%s' will run without Numba JIT", func.__name__
                    )
                else:
                    rewritten = _numba.jit(nopython=True, boundscheck=True)(rewritten)
                    _log.debug("applied numba.jit to '%s'", func.__name__)
            fast[func] = rewritten
            _rewritten_src[func.__qualname__] = new_src
            if new_src != src:
                _log.debug("rewrote '%s'", func.__name__)
        except (OSError, cst.ParserSyntaxError, SyntaxError, NameError) as exc:
            _log.debug("could not rewrite '%s': %s", func.__name__, exc)
            fast[func] = func

    _compiled[module_name] = fast


@dataclass(frozen=True)
class _Plan:
    original: Callable[..., Any]
    fast: Callable[..., Any]
    units: AbstractValue
    registry: Any
    callees: dict[Callable[..., Any], Callable[..., Any]]
    bindings: tuple[_Binding | DispatchBinding | CallableBinding, ...]
    schema: dict[str, Any]

    def valid(self) -> bool:
        return all(binding.unchanged() for binding in self.bindings) and all(
            not is_jit_disabled(callee) for callee in self.callees
        )


@dataclass(frozen=True)
class _Fallback:
    reason: str


_states: dict[str, _Plan | _Fallback] = {}
_compilation_lock = threading.RLock()


def _make_plan(
    func: Callable[..., Any],
    units: Any,
    registry: Any,
    context: _InferenceContext,
    schema: dict[str, Any],
) -> _Plan:
    def check_return(unit: Any) -> None:
        if isinstance(unit, SequenceValue):
            for element in unit.units:
                check_return(element)
        elif unit is not PLAIN and not isinstance(unit, QuantityValue):
            raise _Unsupported("unsupported return representation")

    check_return(units)
    if context.quantity_sources:
        try:
            context.bindings.extend(
                _QUANTITY_DISPATCH.guard(None, context.quantity_sources.values())
            )
        except ValueError as exc:
            raise _Unsupported(str(exc)) from exc
    callees: dict[Callable[..., Any], Callable[..., Any]] = {}
    for dependency in [func, *context.callees]:
        module = dependency.__module__
        registered = dependency in _registry.get(module, ())
        if registered:
            if is_jit_disabled(dependency):
                raise _Unsupported(f"callee {dependency.__qualname__} is disabled")
            if module not in _compiled:
                _compile_module(module)
            fast = _compiled[module].get(dependency, dependency)
            if fast is dependency:
                raise _Unsupported(f"source for {dependency.__qualname__} cannot be rewritten")
            callees[dependency] = fast
        if dependency in context.plain_callees or not registered:
            # Plain helpers run unchanged on the prepared original object. Allow
            # them only when their verified body needs no unit-specific rewrite.
            source = textwrap.dedent(_strip_decorators(inspect.getsource(dependency)))
            tree = cst.parse_module(source)
            registries = {
                k: v for k, v in dependency.__globals__.items() if isinstance(v, _REGISTRY_TYPES)
            }
            if (
                tree.visit(
                    _QuantityStripper(
                        registries, strip_cast=False, nonlocals=dependency.__code__.co_freevars
                    )
                ).code
                != tree.code
            ):
                raise _Unsupported(f"plain callee {dependency.__qualname__} requires rewriting")
    return _Plan(func, callees[func], units, registry, callees, tuple(context.bindings), schema)


def _get_plan(func: Callable[..., Any], bound: inspect.BoundArguments) -> _Plan | None:
    """Guard and publish one specialization atomically; never execute user code here."""
    key = _state_key(func)
    with _compilation_lock:
        cached = _states.get(key)
        if isinstance(cached, _Fallback):
            return None
        try:
            if cached is not None and not cached.valid():
                return None
            schema = argument_schema(bound.arguments)
            if cached is not None:
                if cached.original is not func or not check_schema(cached.schema, schema):
                    return None
                return cached
            context = _InferenceContext()
            units, registry = infer_return_units(
                func,
                bound.args,
                bound.kwargs,
                context=context,
            )
            if units is _SENTINEL:
                raise _Unsupported("unit inference did not establish a safe computation")
            plan = _make_plan(func, units, registry, context, schema)
        except _Unsupported as exc:
            if cached is not None:
                return None
            _states[key] = _Fallback(str(exc))
            _log.warning(
                "'%s': unit inference failed; running as plain Pint (%s)", func.__qualname__, exc
            )
            return None
        _states[key] = plan
        return plan


def compile(instance: Any) -> None:  # noqa: A001 (intentional shadow of built-in)
    """Pre-warm unit inference for all @unit_jit methods on *instance*.

    Iterates every @unit_jit-wrapped method defined on ``type(instance)`` and
    triggers the first-call inference path.  Dummy argument values are derived
    from the method's parameter type annotations:

    * ``np.random.Generator`` parameters → ``np.random.default_rng(0)``
    * ``Quantity`` parameters → ``1 * <matching attr unit>`` from the instance
    * ``Sequence[Quantity]`` / list-of-Quantity parameters → ``self.init_state``
      equivalent, built from Quantity attrs on the instance
    * Everything else → skipped (inference may fall back to lazy on first real call)

    Call this once after constructing an instance if you need inner method
    calls (e.g. ``self.reaction_rates(...)`` called from within a JIT-fast
    function) to be compiled before the first real call.
    """
    # Collect all Quantity attrs (and one level of nesting) on the instance.
    qty_pool: list[Any] = []
    for val in vars(instance).values():
        if isinstance(val, _QUANTITY_TYPES):
            qty_pool.append(1 * val.units)
        elif hasattr(val, "__dict__"):
            for inner_val in vars(val).values():
                if isinstance(inner_val, _QUANTITY_TYPES):
                    qty_pool.append(1 * inner_val.units)

    qty_list = list(qty_pool)  # dummy Sequence[Quantity] arg

    def _dummy_for_param(param: inspect.Parameter) -> Any:
        """Build a dummy value for one function parameter based on its annotation."""
        ann = param.annotation
        if ann is inspect.Parameter.empty:
            return None  # can't guess; skip
        # np.random.Generator
        if ann is np.random.Generator or ann == "np.random.Generator":
            return np.random.default_rng(0)
        # Bare Quantity — match by parameter name first to pick the right unit.
        # Also accept string annotations produced by `from __future__ import annotations`.
        ann_str = str(ann)
        _is_bare_quantity = (
            ann in _QUANTITY_TYPES
            or (isinstance(ann, type) and issubclass(ann, tuple(_QUANTITY_TYPES)))
            or (
                isinstance(ann, str)
                and "Quantity" in ann_str
                and not any(c in ann_str for c in ("Sequence", "list", "List", "["))
            )
        )
        if _is_bare_quantity:
            pname = param.name.lower().lstrip("_")
            if pname in ("t", "time", "dt") and hasattr(instance, "time_horizon"):
                return 1 * instance.time_horizon.units  # type: ignore[operator]
            return qty_pool[0] if qty_pool else None
        # list / Sequence of Quantity — use init_state if available (correct species units),
        # otherwise fall back to the generic qty_list (may have wrong units for some methods).
        if "Quantity" in ann_str and (
            "Sequence" in ann_str or "list" in ann_str or "List" in ann_str
        ):
            init = getattr(instance, "init_state", None)
            return list(init) if init is not None else qty_list
        return None

    for name in dir(type(instance)):
        if name.startswith("__"):
            continue
        method = getattr(type(instance), name, None)
        if method is None or not getattr(method, "__unit_jit_wrapped__", False):
            continue
        qualname = method.__qualname__
        key = f"{method.__module__}::{qualname}"
        if key in _states:
            continue  # already compiled
        inner_func = getattr(method, "__wrapped__", None)
        if inner_func is None:
            continue
        # Build dummy args from the function's type annotations, skipping 'self'.
        try:
            sig = inspect.signature(inner_func)
        except (ValueError, TypeError):
            continue
        params = list(sig.parameters.values())[1:]  # drop 'self'
        dummy_args = [_dummy_for_param(p) for p in params]
        if any(a is None for a in dummy_args):
            continue  # can't build complete dummy args; skip
        bound = getattr(instance, name)
        try:
            bound(*dummy_args)
        except Exception:
            pass  # inference errors are non-fatal


def get_rewritten_source(func: Callable[..., Any]) -> str:
    """Return the rewritten (unit-stripped) source of a @unit_jit function.

    Triggers compilation of the module if it has not happened yet.
    Useful for debugging: inspect what code actually runs in the fast zone.
    """
    module_name = func.__module__
    if module_name not in _compiled:
        _compile_module(module_name)
    src = _rewritten_src.get(func.__qualname__)
    if src is None:
        raise ValueError(f"no rewritten source found for '{func.__qualname__}'")
    return src


def _state_key(func: Callable[..., Any]) -> str:
    """Module-qualified key under which runtime JIT state is tracked for func."""
    return f"{func.__module__}::{func.__qualname__}"


def is_jit_active(func: Callable[..., Any]) -> bool:
    """Return whether func has a cached fast specialization and is not disabled.

    This does not prove that a particular call used that specialization: a changed
    schema or dependency can cause per-call fallback. Use trace_execution() to
    observe actual execution, stripped arguments and raw results.
    """
    key = _state_key(func)
    return isinstance(_states.get(key), _Plan)


def is_jit_disabled(func: Callable[..., Any]) -> bool:
    """Return True if JIT was disabled for func because unit inference failed.

    Such a function runs as plain Pint on every call (no speedup). Returns False for a
    @unit_jit function that has never been called.
    """
    return isinstance(_states.get(_state_key(func)), _Fallback)


# Decorator


@overload
def unit_jit(
    func: type, *, use_numba: bool = ..., input_args: tuple[Any, ...] | None = ...
) -> type: ...


@overload
def unit_jit[**P, R](
    func: Callable[P, R],
    *,
    use_numba: bool = ...,
    input_args: tuple[Any, ...] | None = ...,
) -> Callable[P, R]: ...


@overload
def unit_jit(
    func: None = ...,
    *,
    use_numba: bool = ...,
    input_args: tuple[Any, ...] | None = ...,
) -> Callable[[Any], Any]: ...


def unit_jit(
    func: Any = None,
    *,
    use_numba: bool = False,
    input_args: tuple[Any, ...] | None = None,
) -> Any:
    """JIT decorator for functions and classes: strips Pint overhead, runs fast after first call.

    When applied to a function:
    - First call: abstract-interprets the function body with input units to check
      dimensional correctness and infer return units. Falls back to running the
      original Pint function if source is unavailable.
    - Subsequent calls: converts args to SI floats, runs rewritten version,
      wraps result back into Quantity with cached units.
    - If called from within the fast zone (inner call): skips boundary
      conversion, calls rewritten version directly.

    When applied to a class: applies the function decorator to all non-dunder
    methods defined directly on the class.

    Args:
        use_numba: if True, apply numba.jit(nopython=True) to the rewritten
            float function. Requires numba to be installed. Best suited for
            functions whose rewritten body is pure float/NumPy with no calls
            to other @unit_jit-decorated functions.
        input_args: optional tuple of example arguments used to trigger unit
            inference immediately at decoration time. Equivalent to calling the
            function once with these arguments right after decoration. Note: only
            functions registered before this decorator runs are compiled together;
            use the module-level compile() if that ordering matters.
    """
    if func is None:
        return lambda f: unit_jit(f, use_numba=use_numba, input_args=input_args)

    if getattr(func, "__unit_jit_wrapped__", False):
        return func  # idempotent: already wrapped, skip

    if isinstance(func, type):
        for name, method in func.__dict__.items():
            if inspect.isfunction(method) and not name.startswith("__"):
                setattr(func, name, unit_jit(method, use_numba=use_numba))
        return func

    if use_numba:
        _use_numba.add(func)
    else:
        _use_numba.discard(func)

    module_name = func.__module__
    _registry[module_name].append(func)
    _compiled.pop(module_name, None)  # Late decorators must be included on the next compile.

    signature = inspect.signature(func)
    key = f"{module_name}::{func.__qualname__}"
    # A new definition with the same qualified name must not inherit old inference.
    _states.pop(key, None)

    initial_callable = CallableBinding.capture(func)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        current_signature = signature if initial_callable.unchanged() else inspect.signature(func)
        bound = current_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if _in_fast_zone():
            fast_func = _fast_zone.callees.get(func)
            if fast_func is None:
                raise RuntimeError(f"unverified inner JIT call: {func.__qualname__}")
            with _prepared(bound):
                return _invoke(func, fast_func, bound, "fast")

        plan = _get_plan(func, bound)
        if plan is None:
            return _invoke(func, func, bound, "fallback")
        with _prepared(bound):
            try:
                _fast_zone.callees = plan.callees
                _fast_zone.active = True
                raw = _invoke(func, plan.fast, bound, "fast")
            finally:
                _fast_zone.active = False
                _fast_zone.callees = {}

        return _wrap(raw, plan.units, plan.registry)

    wrapper.__name__ = func.__name__
    wrapper.__qualname__ = func.__qualname__
    wrapper.__module__ = func.__module__
    wrapper.__doc__ = func.__doc__
    wrapper.__annotations__ = func.__annotations__
    w: Any = wrapper
    w.__unit_jit_wrapped__ = True
    w.__wrapped__ = func  # standard unwrap convention
    if input_args is not None:
        wrapper(*(1 * a if isinstance(a, _UNIT_TYPES) else a for a in input_args))
    return wrapper  # type: ignore[return-value]
