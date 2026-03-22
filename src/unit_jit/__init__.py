"""JIT unit-stripping decorator for Pint-annotated Python.

All @unit_jit functions in the same module are rewritten together on the
first call to any of them. Pint Quantities are converted to SI floats at
the outermost boundary; inner @unit_jit calls within the fast zone skip
conversion entirely.

The first call per entry point runs the original (Pint) function to infer
return units; all subsequent calls use the rewritten float version.

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
import types
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, overload

import libcst as cst
from pint import Quantity, Unit, UnitRegistry

ureg = UnitRegistry()
_log = logging.getLogger(__name__)

_fast_zone = threading.local()
_registry: dict[str, list[Callable[..., Any]]] = defaultdict(list)
_compiled: dict[str, dict[str, Callable[..., Any]]] = {}
_rewritten_src: dict[str, str] = {}  # qualname -> rewritten source
_return_units: dict[str, Any] = {}
_arg_dims: dict[str, tuple[list[Any], dict[str, Any]]] = {}  # qualname -> (positional, keyword)
_use_numba: set[str] = set()  # qualnames for which numba.jit should be applied
_return_registry: dict[str, UnitRegistry] = {}  # qualname -> registry used to wrap results


def _in_fast_zone() -> bool:
    return getattr(_fast_zone, "active", False)


@contextmanager
def fast_zone(*objects: Any) -> Iterator[tuple[Any, ...]]:
    """Enter the fast zone manually for a block of code.

    Any @unit_jit-decorated function called within this block skips boundary
    conversion and returns raw SI float values. Use this when you own the loop
    but not the functions called inside it: declare which objects cross the
    boundary, pay the conversion cost once on entry, and work in floats
    throughout.

    Nests safely: if already inside a fast zone, this is a no-op and objects
    are returned unconverted.

    Args:
        *objects: objects whose Quantity attributes should be converted to SI
            floats on entry. The converted proxies are yielded, one per object.

    Example:
        with fast_zone(self) as (fast_self,):
            while ...:
                rates_si = fast_self.reaction_rates(conc_si)  # raw floats

        with fast_zone(self, params) as (fast_self, fast_params):
            ...
    """
    already_active = _in_fast_zone()
    snapped = tuple(objects if already_active else (_snapshot(o) for o in objects))
    if not already_active:
        _fast_zone.active = True
    try:
        yield snapped
    finally:
        if not already_active:
            _fast_zone.active = False


# CST transformer


class _QuantityStripper(cst.CSTTransformer):
    """Strip .magnitude, .to_base_units(), cast("Quantity", x), and ureg.UNIT -> SI float."""

    def __init__(self, ureg_vars: dict[str, UnitRegistry]) -> None:
        super().__init__()
        self._ureg_vars = ureg_vars

    def leave_Attribute(
        self, original_node: cst.Attribute, updated_node: cst.Attribute
    ) -> cst.BaseExpression:
        if updated_node.attr.value == "magnitude":
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

_SNAP_KEY = "__unit_jit_snap__"


def snapshot(obj: Any) -> Any:
    """Convert all Quantity attributes of obj to SI floats, returning a fast proxy.

    Use this before a fast_zone block to pay the conversion cost once rather than
    on every decorated call inside the loop:

        fast_self = snapshot(self)
        with fast_zone():
            while ...:
                rates = fast_self.reaction_rates(state_si)

    The returned proxy is an instance of the same class (method lookup still works)
    but with float-valued attributes. Passing it repeatedly into @unit_jit functions
    inside the fast zone incurs no further conversion overhead.
    """
    return _snapshot(obj)


def _snapshot(obj: Any) -> Any:
    """Eagerly convert all Quantity attrs to SI floats, once at boundary entry.

    Returns an instance of the same class (so method lookup still works) but
    with a float-valued __dict__. Subsequent attribute access inside the fast
    zone is a plain dict lookup, no Pint calls.
    """
    try:
        snap = object.__new__(type(obj))
        snap_dict: dict[str, Any] = {_SNAP_KEY: True}
        for name, val in getattr(obj, "__dict__", {}).items():
            if isinstance(val, Quantity):
                snap_dict[name] = val.to_base_units().magnitude
            elif (
                hasattr(val, "__dict__")
                and not callable(val)
                and not hasattr(val, "__array_interface__")
            ):
                snap_dict[name] = _snapshot(val)
            else:
                snap_dict[name] = val
        snap.__dict__.update(snap_dict)
        return snap
    except Exception:
        return obj  # fallback: use original object as-is


def _to_fast(arg: Any) -> Any:
    """Convert a Quantity to an SI float; snapshot complex objects; leave the rest unchanged."""
    if isinstance(arg, Quantity):
        return arg.to_base_units().magnitude
    if isinstance(arg, (int, float, bool, str, bytes, type(None))):
        return arg
    if hasattr(arg, "__array_interface__"):  # numpy arrays
        return arg
    if _SNAP_KEY in getattr(arg, "__dict__", {}):
        return arg  # already snapshotted
    return _snapshot(arg)


def _infer_units(result: Any) -> tuple[Any, UnitRegistry | None]:
    """Extract SI unit structure and the source registry from a Pint result."""
    if isinstance(result, Quantity):
        return result.to_base_units().units, result._REGISTRY  # noqa: SLF001
    if isinstance(result, (list, tuple)):
        units = [r.to_base_units().units if isinstance(r, Quantity) else None for r in result]
        reg = next((r._REGISTRY for r in result if isinstance(r, Quantity)), None)  # noqa: SLF001
        return (type(result), units), reg
    return None, None


def _parse_return_units(return_units: Any) -> tuple[Any, UnitRegistry | None]:
    """Convert a user-supplied return_units declaration to internal (unit_info, registry) format.

    Accepts Quantity or Unit objects (or lists/tuples thereof):
      - a single Quantity or Unit  -> scalar unit_info
      - a list/tuple thereof       -> (container_type, [units]) unit_info
    """

    def _to_unit_and_reg(r: Any) -> tuple[Any, UnitRegistry | None]:
        if isinstance(r, Quantity):
            return r.to_base_units().units, r._REGISTRY  # noqa: SLF001
        if isinstance(r, Unit):
            return (1 * r).to_base_units().units, r._REGISTRY  # noqa: SLF001
        return None, None

    if isinstance(return_units, (Quantity, Unit)):
        unit, reg = _to_unit_and_reg(return_units)
        return unit, reg
    if isinstance(return_units, types.GenericAlias):
        # list[ureg.mol/ureg.L/ureg.s]: one unit applied to all elements
        unit, reg = _to_unit_and_reg(return_units.__args__[0])
        return (return_units.__origin__, unit), reg
    if isinstance(return_units, (list, tuple)):
        units, reg = [], None
        for r in return_units:
            u, r_ = _to_unit_and_reg(r)
            units.append(u)
            if reg is None:
                reg = r_
        return (type(return_units), units), reg
    return None, None


def _wrap(result: Any, unit_info: Any, wrap_ureg: UnitRegistry) -> Any:
    """Wrap a float/array result back into a Quantity using cached SI units."""
    if unit_info is None:
        return result
    if isinstance(unit_info, tuple):
        cls, units = unit_info
        if isinstance(units, list):
            return cls(
                wrap_ureg.Quantity(r, u) if u is not None else r for r, u in zip(result, units)
            )
        return cls(wrap_ureg.Quantity(r, units) for r in result)
    return wrap_ureg.Quantity(result, unit_info)


# Compilation


def _strip_decorators(src: str) -> str:
    """Remove leading decorator lines from a function's source before rewriting."""
    lines = src.splitlines()
    while lines and lines[0].lstrip().startswith("@"):
        lines.pop(0)
    return "\n".join(lines)


def _compile_module(module_name: str) -> None:
    """Rewrite all @unit_jit functions from a module at once."""
    funcs = _registry[module_name]
    module_globals = funcs[0].__globals__
    ureg_vars = {k: v for k, v in module_globals.items() if isinstance(v, UnitRegistry)}
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
def unit_jit(func: type, *, use_numba: bool = ..., return_units: Any = ...) -> type: ...


@overload
def unit_jit[**P, R](
    func: Callable[P, R], *, use_numba: bool = ..., return_units: Any = ...
) -> Callable[P, R]: ...


@overload
def unit_jit(
    func: None = ..., *, use_numba: bool = ..., return_units: Any = ...
) -> Callable[[Any], Any]: ...


def unit_jit(func: Any = None, *, use_numba: bool = False, return_units: Any = None) -> Any:
    """JIT decorator for functions and classes: strips Pint overhead, runs fast after first call.

    When applied to a function:
    - First call: runs original function (Pint), infers return units, caches them.
      Skipped entirely if return_units is provided.
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
        return_units: declare return units upfront to skip the warm-up call.
            Pass a Quantity for scalar returns or a list of Quantities for list
            returns. The magnitude is ignored; only the unit is used.
            Example: return_units=ureg.mol/ureg.L/ureg.s
            Example: return_units=[ureg.mol/ureg.L/ureg.s, ureg.mol/ureg.L/ureg.s]
    """
    if func is None:
        return lambda f: unit_jit(f, use_numba=use_numba, return_units=return_units)

    if isinstance(func, type):
        for name, method in func.__dict__.items():
            if inspect.isfunction(method) and not name.startswith("__"):
                setattr(
                    func, name, unit_jit(method, use_numba=use_numba, return_units=return_units)
                )
        return func

    if use_numba:
        _use_numba.add(func.__qualname__)

    module_name = func.__module__
    _registry[module_name].append(func)

    if return_units is not None:
        unit_info, reg = _parse_return_units(return_units)
        _return_units[func.__qualname__] = unit_info
        _return_registry[func.__qualname__] = reg if reg is not None else ureg

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

        # Entry point. First call: run original to infer return units + cache arg dimensions.
        if qualname not in _return_units:
            result = func(*args, **kwargs)
            unit_info, reg = _infer_units(result)
            _return_units[qualname] = unit_info
            _return_registry[qualname] = reg if reg is not None else ureg
            _arg_dims[qualname] = (
                [a.dimensionality if isinstance(a, Quantity) else None for a in args],
                {
                    k: v.dimensionality if isinstance(v, Quantity) else None
                    for k, v in kwargs.items()
                },
            )
            return result

        # Subsequent calls: check dimensions, convert, run fast version, wrap result.
        # Skipped when return_units is declared (no warm-up call to populate _arg_dims).
        if qualname in _arg_dims:
            pos_dims, kw_dims = _arg_dims[qualname]
            for i, (arg, dim) in enumerate(zip(args, pos_dims)):
                if dim is not None and isinstance(arg, Quantity) and arg.dimensionality != dim:
                    raise TypeError(
                        f"{func.__qualname__}: argument {i} has dimensions "
                        f"{dict(arg.dimensionality)}, expected {dict(dim)}"
                    )
            for key, dim in kw_dims.items():
                arg = kwargs.get(key)
                if dim is not None and isinstance(arg, Quantity) and arg.dimensionality != dim:  # type: ignore[union-attr]
                    raise TypeError(
                        f"{func.__qualname__}: argument '{key}' has dimensions "
                        f"{dict(arg.dimensionality)}, expected {dict(dim)}"  # type: ignore[union-attr]
                    )
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
    return wrapper  # type: ignore[return-value]
