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

Quantity attributes on objects (e.g. self.params.alpha) are handled via an
eager snapshot: all Quantity attrs are converted to SI floats once at
boundary entry, so attribute access inside the loop is a plain dict lookup.
"""

import inspect
import logging
import textwrap
import threading
import weakref
from collections import defaultdict
from collections.abc import Callable
from typing import Any, overload

import libcst as cst
import numpy as np
from pint import UnitRegistry

from unit_jit._inferrer import (  # noqa: E402
    _QUANTITY_TYPES,
    _REGISTRY_TYPES,
    _SENTINEL,
    _SNAP_KEY,
    _UNIT_TYPES,
    _UNKNOWN,  # noqa: F401 (re-exported for tests)
    _ListReturn,
    _strip_decorators,
    infer_return_units,
)

_log = logging.getLogger(__name__)

_fast_zone = threading.local()
_registry: dict[str, list[Callable[..., Any]]] = defaultdict(list)
_compiled: dict[str, dict[str, Callable[..., Any]]] = {}
_rewritten_src: dict[str, str] = {}  # qualname -> rewritten source
_return_units: dict[str, Any] = {}
_arg_dims: dict[str, tuple[list[Any], dict[str, Any]]] = {}  # qualname -> (positional, keyword)
_use_numba: set[str] = set()  # qualnames for which numba.jit should be applied
_return_registry: dict[str, UnitRegistry | None] = {}  # qualname -> registry used to wrap results
_jit_disabled: set[str] = set()  # qualnames where inference failed: always run original
_snapshot_cache: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()


def _in_fast_zone() -> bool:
    return getattr(_fast_zone, "active", False)


def _eval_numeric_cst(node: cst.BaseExpression) -> float | None:
    if isinstance(node, cst.Integer):
        return float(node.value)
    if isinstance(node, cst.Float):
        return float(node.value)
    if isinstance(node, cst.UnaryOperation) and isinstance(node.operator, cst.Minus):
        value = _eval_numeric_cst(node.expression)
        return -value if value is not None else None
    if isinstance(node, cst.BinaryOperation):
        left = _eval_numeric_cst(node.left)
        right = _eval_numeric_cst(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.operator, cst.Add):
            return left + right
        if isinstance(node.operator, cst.Subtract):
            return left - right
        if isinstance(node.operator, cst.Multiply):
            return left * right
        if isinstance(node.operator, cst.Divide):
            return left / right
        if isinstance(node.operator, cst.FloorDivide):
            return left // right
        if isinstance(node.operator, cst.Power):
            return left**right
    return None


def _unit_jit_rescale_to_magnitude(value: Any, scale: float) -> Any:
    return value / scale


# CST transformer


class _QuantityStripper(cst.CSTTransformer):
    """Strip unit-aware Quantity syntax into float operations for the fast zone."""

    def __init__(self, ureg_vars: dict[str, UnitRegistry]) -> None:
        super().__init__()
        self._ureg_vars = ureg_vars

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
                unit_arg = updated_node.value.args[0].value
                scale = _eval_numeric_cst(unit_arg)
                if scale is not None:
                    return cst.Call(
                        func=cst.Name("_unit_jit_rescale_to_magnitude"),
                        args=[
                            cst.Arg(updated_node.value.func.value),
                            cst.Arg(cst.Float(repr(float(scale)))),
                        ],
                    )
                if isinstance(unit_arg, cst.Attribute) and isinstance(unit_arg.value, cst.Name):
                    ureg_instance = self._ureg_vars.get(unit_arg.value.value)
                    if ureg_instance is not None:
                        try:
                            si_val = (
                                (1 * getattr(ureg_instance, unit_arg.attr.value))
                                .to_base_units()
                                .magnitude
                            )
                            return cst.Call(
                                func=cst.Name("_unit_jit_rescale_to_magnitude"),
                                args=[
                                    cst.Arg(updated_node.value.func.value),
                                    cst.Arg(cst.Float(repr(float(si_val)))),
                                ],
                            )
                        except Exception:
                            pass
            return updated_node.value
        # ureg.UNIT -> SI float (e.g. ureg.s -> 1.0, ureg.cm -> 0.01)
        if isinstance(updated_node.value, cst.Name):
            ureg_instance = self._ureg_vars.get(updated_node.value.value)
            if ureg_instance is not None:
                try:
                    si_val = (
                        (1 * getattr(ureg_instance, updated_node.attr.value))
                        .to_base_units()
                        .magnitude
                    )
                    return cst.Float(repr(float(si_val)))
                except Exception:
                    pass
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
            isinstance(updated_node.func, cst.Name)
            and updated_node.func.value == "cast"
            and len(updated_node.args) == 2
            and isinstance(updated_node.args[0].value, cst.SimpleString)
            and "Quantity" in updated_node.args[0].value.value
        ):
            return updated_node.args[1].value

        return updated_node


# Boundary helpers


def _snapshot(obj: Any) -> Any:
    """Eagerly convert all Quantity attrs to SI floats, once at boundary entry.

    Plain Quantity objects are converted directly to their SI magnitude.
    NamedTuples are reconstructed with each field recursively snapshotted.
    For other objects, returns an instance of the same class (so method lookup
    still works) with a float-valued __dict__.

    Results are cached in a WeakKeyDictionary so that repeated calls with the
    same object (e.g. self on every SDE step) pay the Pint conversion cost only
    once.  The cache entry is evicted automatically when the object is garbage
    collected.  Caching is skipped for objects whose __dict__ may change
    (detected by the _SNAP_KEY sentinel already being present, meaning we have
    already snapshotted and there is nothing to do).
    """
    if isinstance(obj, _QUANTITY_TYPES):
        return obj.to_base_units().magnitude
    if hasattr(type(obj), "_fields"):  # NamedTuple — immutable, always cache-safe
        try:
            cached = _snapshot_cache[obj]
            return cached
        except (KeyError, TypeError):
            pass
        result = type(obj)._make(_snapshot(v) for v in obj)  # type: ignore[attr-defined]
        try:
            _snapshot_cache[obj] = result
        except TypeError:
            pass
        return result
    try:
        cached = _snapshot_cache[obj]
        return cached
    except (KeyError, TypeError):
        pass
    try:
        snap = object.__new__(type(obj))
        snap_dict: dict[str, Any] = {_SNAP_KEY: True}
        for name, val in getattr(obj, "__dict__", {}).items():
            if isinstance(val, _QUANTITY_TYPES):
                snap_dict[name] = val.to_base_units().magnitude
            elif isinstance(val, list):
                snap_dict[name] = [_snapshot(el) for el in val]
            elif hasattr(type(val), "_fields") and isinstance(val, tuple):  # NamedTuple
                snap_dict[name] = _snapshot(val)
            elif isinstance(val, tuple):
                snap_dict[name] = tuple(_snapshot(el) for el in val)
            elif (
                hasattr(val, "__dict__")
                and not callable(val)
                and not hasattr(val, "__array_interface__")
            ):
                snap_dict[name] = _snapshot(val)
            else:
                snap_dict[name] = val
        snap.__dict__.update(snap_dict)
        try:
            _snapshot_cache[obj] = snap
        except TypeError:
            pass
        return snap
    except Exception:
        return obj  # fallback: use original object as-is


def _to_fast(arg: Any) -> Any:
    """Convert a Quantity to an SI float; snapshot complex objects; leave the rest unchanged."""
    if isinstance(arg, _QUANTITY_TYPES):
        return arg.to_base_units().magnitude
    if isinstance(arg, list):
        return [_to_fast(el) for el in arg]
    if hasattr(type(arg), "_fields") and isinstance(arg, tuple):  # NamedTuple before plain tuple
        return type(arg)._make(_to_fast(el) for el in arg)  # type: ignore[attr-defined]
    if isinstance(arg, tuple):
        return tuple(_to_fast(el) for el in arg)
    if isinstance(arg, (int, float, bool, str, bytes, type(None))):
        return arg
    if hasattr(arg, "__array_interface__"):  # numpy arrays
        return arg
    if _SNAP_KEY in getattr(arg, "__dict__", {}):
        return arg  # already snapshotted
    return _snapshot(arg)


def _wrap(result: Any, unit_info: Any, wrap_ureg: UnitRegistry | None) -> Any:
    """Wrap a float/array result back into a Quantity using cached SI units.

    Handles nested structures: _ListReturn entries may themselves be _ListReturn,
    enabling list[tuple[Quantity, ...]] and similar return types. Variable-length
    lists are supported by repeating the last inferred element unit.
    """
    if unit_info is None or unit_info is _UNKNOWN:
        return result
    assert wrap_ureg is not None
    if isinstance(unit_info, _ListReturn):
        n = len(result)
        units = unit_info.units
        if len(units) < n:
            units = list(units) + [units[-1]] * (n - len(units))
        wrapped = [_wrap(r, u, wrap_ureg) for r, u in zip(result, units)]
        if unit_info.kind == "namedtuple" and unit_info.cls is not None:
            return unit_info.cls._make(wrapped)  # type: ignore[attr-defined]
        return wrapped if unit_info.kind == "list" else tuple(wrapped)
    if isinstance(unit_info, tuple):
        cls, units = unit_info
        if isinstance(units, list):
            return cls(
                wrap_ureg.Quantity(r, u) if u is not None else r for r, u in zip(result, units)
            )
        return cls(wrap_ureg.Quantity(r, units) for r in result)
    return wrap_ureg.Quantity(result, unit_info)


# Compilation


def _compile_module(module_name: str) -> None:
    """Rewrite all @unit_jit functions from a module at once."""
    funcs = _registry[module_name]
    module_globals = funcs[0].__globals__
    module_globals.setdefault("_unit_jit_rescale_to_magnitude", _unit_jit_rescale_to_magnitude)
    ureg_vars = {k: v for k, v in module_globals.items() if isinstance(v, _REGISTRY_TYPES)}
    stripper = _QuantityStripper(ureg_vars)
    fast: dict[str, Callable[..., Any]] = {}

    for func in funcs:
        try:
            src = inspect.getsource(func)
            src = textwrap.dedent(src)
            src = _strip_decorators(src)
            tree = cst.parse_module(src)
            new_src = tree.visit(stripper).code
            namespace: dict[str, Any] = {}
            exec(new_src, module_globals, namespace)
            rewritten = namespace[func.__name__]
            if func.__qualname__ in _use_numba:
                import numba as _numba  # lazy: only when use_numba=True

                rewritten = _numba.jit(nopython=True)(rewritten)
                _log.debug("applied numba.jit to '%s'", func.__name__)
            fast[func.__qualname__] = rewritten
            _rewritten_src[func.__qualname__] = new_src
            if new_src != src:
                _log.debug("rewrote '%s'", func.__name__)
        except (OSError, cst.ParserSyntaxError, SyntaxError) as exc:
            _log.debug("could not rewrite '%s': %s", func.__name__, exc)
            fast[func.__name__] = func

    _compiled[module_name] = fast


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

    After all methods have been attempted, any stale snapshot that was cached for
    ``instance`` during failed warm-up calls is evicted from ``_snapshot_cache``
    so that the next real call builds a fresh, fully-populated snapshot.

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
        if qualname in _return_units or qualname in _jit_disabled:
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

    # Evict any stale snapshot that was cached during warm-up calls so the next
    # real simulation call builds a fresh snapshot with all instance attributes.
    try:
        del _snapshot_cache[instance]
    except (KeyError, TypeError):
        pass


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
        _use_numba.add(func.__qualname__)

    module_name = func.__module__
    _registry[module_name].append(func)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if module_name not in _compiled:
            _compile_module(module_name)

        fast_func: Callable[..., Any] = _compiled[module_name].get(func.__qualname__, func)  # type: ignore[assignment]
        qualname = func.__qualname__

        if _in_fast_zone():
            # Already in float world: snapshot any objects still carrying
            # Quantities (e.g. original self via Python's descriptor protocol).
            fast_args = tuple(_to_fast(a) for a in args)
            fast_kwargs = {k: _to_fast(v) for k, v in kwargs.items()}
            return fast_func(*fast_args, **fast_kwargs)

        # Functions where inference failed always run as original Pint (no JIT).
        if qualname in _jit_disabled:
            return func(*args, **kwargs)

        # Entry point: infer units on first call via abstract interpretation.
        if qualname not in _return_units:
            _arg_dims[qualname] = (
                [a.dimensionality if isinstance(a, _QUANTITY_TYPES) else None for a in args],
                {
                    k: v.dimensionality if isinstance(v, _QUANTITY_TYPES) else None
                    for k, v in kwargs.items()
                },
            )
            inferred_info, inferred_reg = infer_return_units(func, args, kwargs, _return_units)
            if inferred_info is not _SENTINEL:
                _return_units[qualname] = inferred_info
                if inferred_info is not None and inferred_reg is None:
                    raise RuntimeError(
                        f"'{func.__qualname__}': could not determine a UnitRegistry from "
                        "arguments or module globals; pass Quantity arguments or define "
                        "ureg = UnitRegistry() at module level."
                    )
                _return_registry[qualname] = inferred_reg
                # Fall through to fast path below.
            else:
                # Inference failed: disable JIT for this function permanently.
                _jit_disabled.add(qualname)
                _log.warning(
                    "'%s': unit inference failed; running as plain Pint on every call "
                    "(no JIT speedup). Enable debug logging for details.",
                    func.__qualname__,
                )
                return func(*args, **kwargs)

        # Subsequent calls (and first call when inference succeeded): check
        # dimensions, convert, run fast version, wrap result.
        # Dimension check: skipped when inference failed to record arg dims (no _arg_dims entry).
        if qualname in _arg_dims:
            pos_dims, kw_dims = _arg_dims[qualname]
            for i, (arg, dim) in enumerate(zip(args, pos_dims)):
                if (
                    dim is not None
                    and isinstance(arg, _QUANTITY_TYPES)
                    and arg.dimensionality != dim
                ):  # noqa: E501
                    msg = (
                        f"{func.__qualname__}: argument {i} has dimensions "
                        f"{dict(arg.dimensionality)}, expected {dict(dim)}"
                    )
                    _log.warning("dimension mismatch: %s", msg)
                    raise TypeError(msg)
            for key, dim in kw_dims.items():
                arg: Any = kwargs.get(key)
                if (
                    dim is not None
                    and isinstance(arg, _QUANTITY_TYPES)
                    and arg.dimensionality != dim
                ):
                    msg = (
                        f"{func.__qualname__}: argument '{key}' has dimensions "
                        f"{dict(arg.dimensionality)}, expected {dict(dim)}"
                    )
                    _log.warning("dimension mismatch: %s", msg)
                    raise TypeError(msg)
        fast_args = tuple(_to_fast(a) for a in args)
        fast_kwargs = {k: _to_fast(v) for k, v in kwargs.items()}
        _fast_zone.active = True
        try:
            raw = fast_func(*fast_args, **fast_kwargs)
        finally:
            _fast_zone.active = False
        return _wrap(raw, _return_units[qualname], _return_registry[qualname])

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
