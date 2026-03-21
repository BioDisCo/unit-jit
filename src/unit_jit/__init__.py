"""JIT unit-stripping decorator for Pint-annotated Python.

All @unit_jit functions in the same module are rewritten together on the
first call to any of them. Pint Quantities are converted to SI floats at
the outermost boundary; inner @unit_jit calls within the fast zone skip
conversion entirely.

The first call per entry point runs the original (Pint) function to infer
return units; all subsequent calls use the rewritten float version.

Rewrites applied inside the fast zone:
  - x.magnitude          → x
  - x.to_base_units()    → x
  - cast("Quantity", x)  → x
  - ureg.UNIT            → SI float (e.g. ureg.s → 1.0, ureg.cm → 0.01)
  - arithmetic unchanged (works identically for floats)

Quantity attributes on objects (e.g. self.params.alpha) are handled via an
eager snapshot: all Quantity attrs are converted to SI floats once at
boundary entry, so attribute access inside the loop is a plain dict lookup.
"""

import inspect
import textwrap
import threading
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import libcst as cst
from pint import Quantity, UnitRegistry

ureg = UnitRegistry()

_fast_zone = threading.local()
_registry: dict[str, list[Callable[..., Any]]] = defaultdict(list)
_compiled: dict[str, dict[str, Callable[..., Any]]] = {}
_return_units: dict[str, Any] = {}
_arg_dims: dict[str, tuple[list[Any], dict[str, Any]]] = {}  # qualname → (positional, keyword)


def _in_fast_zone() -> bool:
    return getattr(_fast_zone, "active", False)


# ── CST transformer ────────────────────────────────────────────────────────────


class _QuantityStripper(cst.CSTTransformer):
    """Strip .magnitude, .to_base_units(), cast("Quantity", x), and ureg.UNIT → SI float."""

    def __init__(self, ureg_vars: dict[str, UnitRegistry]) -> None:
        super().__init__()
        self._ureg_vars = ureg_vars

    def leave_Attribute(
        self, original_node: cst.Attribute, updated_node: cst.Attribute
    ) -> cst.BaseExpression:
        if updated_node.attr.value == "magnitude":
            return updated_node.value
        # ureg.UNIT → SI float (e.g. ureg.s → 1.0, ureg.cm → 0.01)
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
        # x.to_base_units() → x
        if (
            isinstance(updated_node.func, cst.Attribute)
            and updated_node.func.attr.value == "to_base_units"
            and not updated_node.args
        ):
            return updated_node.func.value

        # cast("Quantity", x) → x
        if (
            isinstance(updated_node.func, cst.Name)
            and updated_node.func.value == "cast"
            and len(updated_node.args) == 2
            and isinstance(updated_node.args[0].value, cst.SimpleString)
            and "Quantity" in updated_node.args[0].value.value
        ):
            return updated_node.args[1].value

        return updated_node


# ── Boundary helpers ───────────────────────────────────────────────────────────

_SNAP_KEY = "__unit_jit_snap__"


def _snapshot(obj: Any) -> Any:
    """Eagerly convert all Quantity attrs to SI floats, once at boundary entry.

    Returns an instance of the same class (so method lookup still works) but
    with a float-valued __dict__. Subsequent attribute access inside the fast
    zone is a plain dict lookup — no Pint calls.
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
    """Convert Quantity → SI float; snapshot complex objects; leave the rest."""
    if isinstance(arg, Quantity):
        return arg.to_base_units().magnitude
    if isinstance(arg, (int, float, bool, str, bytes, type(None))):
        return arg
    if hasattr(arg, "__array_interface__"):  # numpy arrays
        return arg
    if _SNAP_KEY in getattr(arg, "__dict__", {}):
        return arg  # already snapshotted
    return _snapshot(arg)


def _infer_units(result: Any) -> Any:
    """Extract SI unit structure from a Pint result (for later wrapping)."""
    if isinstance(result, Quantity):
        return result.to_base_units().units
    if isinstance(result, (list, tuple)):
        units = [r.to_base_units().units if isinstance(r, Quantity) else None for r in result]
        return (type(result), units)
    return None


def _wrap(result: Any, unit_info: Any) -> Any:
    """Wrap a float/array result back into Quantity using cached SI units."""
    if unit_info is None:
        return result
    if isinstance(unit_info, tuple):
        cls, units = unit_info
        return cls(ureg.Quantity(r, u) if u is not None else r for r, u in zip(result, units))
    return ureg.Quantity(result, unit_info)


# ── Compilation ────────────────────────────────────────────────────────────────


def _strip_decorators(src: str) -> str:
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
            fast[func.__name__] = namespace[func.__name__]
            tag = "rewrote" if new_src != src else "no changes"
            print(f"[unit_jit] {tag}: '{func.__name__}'")
        except Exception as exc:
            print(f"[unit_jit] could not rewrite '{func.__name__}': {exc}")
            fast[func.__name__] = func

    _compiled[module_name] = fast


# ── Decorator ──────────────────────────────────────────────────────────────────


def unit_jit[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """JIT decorator: strips Pint overhead, runs fast after first call.

    - First call: runs original function (Pint), infers return units, caches them.
    - Subsequent calls: converts args to SI floats, runs rewritten version,
      wraps result back into Quantity with cached units.
    - If called from within the fast zone (inner call): skips boundary
      conversion, calls rewritten version directly.
    """
    module_name = func.__module__
    _registry[module_name].append(func)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if module_name not in _compiled:
            _compile_module(module_name)

        fast_func = _compiled[module_name].get(func.__name__, func)
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
            _return_units[qualname] = _infer_units(result)
            _arg_dims[qualname] = (
                [a.dimensionality if isinstance(a, Quantity) else None for a in args],
                {
                    k: v.dimensionality if isinstance(v, Quantity) else None
                    for k, v in kwargs.items()
                },
            )
            return result

        # Subsequent calls: check dimensions, convert → fast → wrap.
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
        return _wrap(raw, _return_units[qualname])

    wrapper.__name__ = func.__name__
    wrapper.__qualname__ = func.__qualname__
    wrapper.__module__ = func.__module__
    wrapper.__doc__ = func.__doc__
    wrapper.__annotations__ = func.__annotations__
    return wrapper  # type: ignore[return-value]
