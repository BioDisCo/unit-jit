"""Abstract interpretation of Pint units through function CSTs.

Propagates units symbolically through all branches of a function body,
catching dimensional errors (e.g. adding meters to seconds) without
executing the function.
"""

from __future__ import annotations

import inspect
import logging
import textwrap
import types
from collections.abc import Callable
from typing import Any

import libcst as cst
import numpy as np
from pint import Quantity, Unit, UnitRegistry


def _collect_types() -> tuple[tuple[type, ...], tuple[type, ...], tuple[type, ...]]:
    qtypes: list[type] = [Quantity]
    rtypes: list[type] = [UnitRegistry]
    utypes: list[type] = [Unit]
    try:
        import pintrs as _pintrs  # optional dependency

        for attr, lst in (
            ("Quantity", qtypes),
            ("ArrayQuantity", qtypes),
            ("UnitRegistry", rtypes),
            ("Unit", utypes),
        ):
            t = getattr(_pintrs, attr, None)
            if t is not None:
                lst.append(t)
    except ImportError:
        pass
    return tuple(qtypes), tuple(rtypes), tuple(utypes)


_QUANTITY_TYPES, _REGISTRY_TYPES, _UNIT_TYPES = _collect_types()

_log = logging.getLogger("unit_jit")

# Snapshot marker: set on proxy objects created by _snapshot().
_SNAP_KEY = "__unit_jit_snap__"

# Sentinel: no return statement seen yet (void function or inference not run).
_SENTINEL: object = object()
# Sentinel: unit cannot be determined (unresolved call), distinct from None = dimensionless.
_UNKNOWN: object = object()

# All methods on np.random.Generator that return plain (dimensionless) int/float/ndarray.
_RNG_DIMENSIONLESS_METHODS: frozenset[str] = frozenset(
    {
        "poisson",
        "binomial",
        "geometric",
        "negative_binomial",
        "hypergeometric",
        "multinomial",
        "standard_normal",
        "normal",
        "exponential",
        "standard_exponential",
        "standard_gamma",
        "standard_t",
        "uniform",
        "random",
        "integers",
        "choice",
        "permutation",
        "shuffle",
        "rayleigh",
        "laplace",
        "logistic",
        "gumbel",
        "pareto",
        "weibull",
        "power",
        "vonmises",
        "beta",
        "chisquare",
        "f",
        "gamma",
    }
)


class _ListReturn:
    """Unit info for a function returning list[Quantity] or tuple[Quantity, ...]."""

    __slots__ = ("kind", "units")

    def __init__(self, kind: str, units: list[Any]) -> None:
        self.kind = kind
        self.units = units

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _ListReturn)
            and self.kind == other.kind
            and len(self.units) == len(other.units)
            and all(u1 == u2 for u1, u2 in zip(self.units, other.units))
        )


# ---------------------------------------------------------------------------
# Unit arithmetic helpers
# ---------------------------------------------------------------------------


def _unit_mul(u1: Any, u2: Any) -> Any:
    if u1 is _UNKNOWN or u2 is _UNKNOWN:
        return _UNKNOWN
    if u1 is None:
        return u2
    if u2 is None:
        return u1
    try:
        return u1 * u2
    except Exception:
        return _UNKNOWN


def _unit_div(u1: Any, u2: Any) -> Any:
    if u1 is _UNKNOWN or u2 is _UNKNOWN:
        return _UNKNOWN
    if u1 is None and u2 is None:
        return None
    if u2 is None:
        return u1
    try:
        if u1 is None:
            reg = u2._REGISTRY  # noqa: SLF001
            return (reg.Quantity(1) / reg.Quantity(1, u2)).to_base_units().units
        return u1 / u2
    except Exception:
        return _UNKNOWN


def _unit_pow(u: Any, exp: float) -> Any:
    if u is _UNKNOWN:
        return _UNKNOWN
    if u is None:
        return None
    try:
        return u**exp
    except Exception:
        return _UNKNOWN


def _eval_literal(node: Any) -> float | None:
    """Return the numeric value of a CST literal node, or None if not a literal."""
    if isinstance(node, cst.Integer):
        return int(node.value)
    if isinstance(node, cst.Float):
        return float(node.value)
    if isinstance(node, cst.UnaryOperation) and isinstance(node.operator, cst.Minus):
        v = _eval_literal(node.expression)
        return -v if v is not None else None
    return None


# ---------------------------------------------------------------------------
# Known function unit signatures
# ---------------------------------------------------------------------------

# Method names on arrays/quantities that preserve the receiver's unit.
# Used for method calls (arr.sum()), not module-level functions (np.sum(arr)).
_UNIT_PRESERVING_METHODS: frozenset[str] = frozenset(
    {
        "sum",
        "min",
        "max",
        "mean",
        "std",
        "flatten",
        "ravel",
        "copy",
        "clip",
        "abs",
        "conj",
        "real",
        "imag",
        "squeeze",
        "reshape",
        "cumsum",
        "sort",
        "diagonal",
        "trace",
        "round",
        "floor",
        "ceil",
        "astype",
    }
)


# Short-hand lambdas used across multiple entries.
def _p(us: list[Any]) -> Any:
    return us[0] if us else None  # preserve first arg unit


def _d(_: list[Any]) -> Any:
    return None  # dimensionless


def _sqrt(us: list[Any]) -> Any:
    return _unit_pow(us[0], 0.5) if us else None


def _sq(us: list[Any]) -> Any:
    return _unit_pow(us[0], 2) if us else None


def _cbrt(us: list[Any]) -> Any:
    return _unit_pow(us[0], 1 / 3) if us else None


def _mul2(us: list[Any]) -> Any:
    return _unit_mul(us[0], us[1]) if len(us) > 1 else _p(us)


_KNOWN_CALLS: dict[str, Callable[[list[Any]], Any]] = {
    # ---- Python builtins ------------------------------------------------
    "abs": _p,
    "round": _p,
    "float": _d,
    "int": _d,
    "bool": _d,
    "len": _d,
    "sum": _p,
    "min": _p,
    "max": _p,
    # ---- math module ----------------------------------------------------
    "math.sqrt": _sqrt,
    "math.exp": _d,
    "math.log": _d,
    "math.log2": _d,
    "math.log10": _d,
    "math.log1p": _d,
    "math.expm1": _d,
    "math.sin": _d,
    "math.cos": _d,
    "math.tan": _d,
    "math.asin": _d,
    "math.acos": _d,
    "math.atan": _d,
    "math.atan2": _d,
    "math.sinh": _d,
    "math.cosh": _d,
    "math.tanh": _d,
    "math.fabs": _p,
    "math.hypot": _p,
    "math.floor": _d,  # returns int
    "math.ceil": _d,  # returns int
    "math.trunc": _d,  # returns int
    # ---- numpy: dimensionless output ------------------------------------
    "np.exp": _d,
    "np.expm1": _d,
    "np.log": _d,
    "np.log2": _d,
    "np.log10": _d,
    "np.log1p": _d,
    "np.sin": _d,
    "np.cos": _d,
    "np.tan": _d,
    "np.arcsin": _d,
    "np.arccos": _d,
    "np.arctan": _d,
    "np.arctan2": _d,
    "np.sinh": _d,
    "np.cosh": _d,
    "np.tanh": _d,
    "np.arcsinh": _d,
    "np.arccosh": _d,
    "np.arctanh": _d,
    "np.sign": _d,
    "np.isfinite": _d,
    "np.isnan": _d,
    "np.isinf": _d,
    "np.isreal": _d,
    "np.iscomplex": _d,
    "np.any": _d,
    "np.all": _d,
    "np.argmin": _d,
    "np.argmax": _d,
    "np.argsort": _d,
    "np.nonzero": _d,
    "np.linspace": _d,
    "np.arange": _d,
    "np.logspace": _d,
    "np.geomspace": _d,
    # ---- numpy: unit-preserving ----------------------------------------
    "np.abs": _p,
    "np.fabs": _p,
    "np.sum": _p,
    "np.nansum": _p,
    "np.min": _p,
    "np.nanmin": _p,
    "np.max": _p,
    "np.nanmax": _p,
    "np.mean": _p,
    "np.nanmean": _p,
    "np.std": _p,
    "np.nanstd": _p,
    "np.median": _p,
    "np.nanmedian": _p,
    "np.cumsum": _p,
    "np.diff": _p,
    "np.clip": _p,
    "np.sort": _p,
    "np.real": _p,
    "np.imag": _p,
    "np.conj": _p,
    "np.floor": _p,
    "np.ceil": _p,
    "np.around": _p,
    "np.round_": _p,
    "np.trunc": _p,
    "np.fix": _p,
    "np.roll": _p,
    "np.flip": _p,
    "np.rot90": _p,
    "np.tile": _p,
    "np.repeat": _p,
    "np.squeeze": _p,
    "np.expand_dims": _p,
    "np.transpose": _p,
    "np.reshape": _p,
    "np.ravel": _p,
    "np.broadcast_to": _p,
    "np.pad": _p,
    "np.append": _p,
    "np.concatenate": _p,
    "np.stack": _p,
    "np.hstack": _p,
    "np.vstack": _p,
    "np.dstack": _p,
    "np.array": _p,
    "np.asarray": _p,
    "np.ascontiguousarray": _p,
    "np.copy": _p,
    "np.hypot": _p,
    "np.linalg.norm": _p,
    # ---- numpy: array creation (dimensionless) -------------------------
    "np.empty": _d,
    "np.zeros": _d,
    "np.ones": _d,
    "np.empty_like": _d,
    "np.zeros_like": _d,
    "np.ones_like": _d,
    "np.full": lambda us: us[1] if len(us) > 1 else None,
    "np.full_like": lambda us: us[1] if len(us) > 1 else None,
    # ---- numpy: multiply units of both args ----------------------------
    "np.dot": _mul2,
    "np.inner": _mul2,
    "np.outer": _mul2,
    "np.matmul": _mul2,
    "np.cross": _mul2,
    "np.tensordot": _mul2,
    "np.kron": _mul2,
    # ---- numpy: power / root -------------------------------------------
    "np.sqrt": _sqrt,
    "np.square": _sq,
    "np.cbrt": _cbrt,
    # ---- numpy.linalg: solve Ax=b, x has units [b]/[A] ----------------
    "np.linalg.solve": lambda us: _unit_div(us[1], us[0]) if len(us) > 1 else None,
    # ---- numpy: shape/structure (dimensionless output) --------------------
    "np.shape": _d,
    "np.ndim": _d,
    "np.size": _d,
    # ---- numpy: array manipulation (unit-preserving first arg) ------------
    "np.atleast_1d": _p,
    "np.atleast_2d": _p,
    "np.atleast_3d": _p,
    "np.add.reduceat": _p,
    "np.multiply.reduceat": _p,
    "np.maximum.reduceat": _p,
    "np.minimum.reduceat": _p,
    "np.searchsorted": _d,
    "np.unravel_index": _d,
    "np.ravel_index": _d,
    "np.unique": _p,
    "np.where": lambda us: us[1] if len(us) > 1 else None,
    "np.select": lambda us: us[0] if us else None,
    "np.piecewise": _p,
    "np.frompyfunc": _d,
    "np.fromiter": _d,
    "np.fromfunction": _d,
}


# ---------------------------------------------------------------------------
# Attribute unit extraction
# ---------------------------------------------------------------------------


def _extract_attr_units(obj: Any) -> dict[str, Any]:
    """Recursively extract Quantity attribute units from an object into a nested dict."""
    result: dict[str, Any] = {}
    if hasattr(type(obj), "_fields"):  # NamedTuple: use _asdict() since no __dict__
        items = obj._asdict().items()
    else:
        items = getattr(obj, "__dict__", {}).items()
    for k, v in items:
        if k == _SNAP_KEY:
            continue
        if isinstance(v, _QUANTITY_TYPES):
            result[k] = v.to_base_units().units
        elif hasattr(type(v), "_fields"):  # nested NamedTuple
            nested = _extract_attr_units(v)
            if nested:
                result[k] = nested
        elif hasattr(v, "__dict__") and not callable(v) and not hasattr(v, "__array_interface__"):
            nested = _extract_attr_units(v)
            if nested:
                result[k] = nested
    return result


# ---------------------------------------------------------------------------
# Abstract interpreter
# ---------------------------------------------------------------------------


class _UnitInferrer:
    """Abstract interpreter: propagates Pint units through a function's CST.

    Analyzes all branches of if/else statements, so dimensional errors are
    caught regardless of which path the runtime would take.
    """

    def __init__(
        self,
        env: dict[str, Any],
        ureg_vars: dict[str, UnitRegistry],
        module_globals: dict[str, Any],
        return_units: dict[str, Any],
        param_objects: dict[str, Any] | None = None,
    ) -> None:
        self.env: dict[str, Any] = dict(env)
        self.ureg_vars = ureg_vars
        self.module_globals = module_globals
        self.return_units = return_units
        self.param_objects: dict[str, Any] = param_objects or {}
        self._return: Any = _SENTINEL  # _SENTINEL = no return seen yet
        self._inferring: set[str] = set()  # qualnames currently being lazily inferred (cycle guard)

    def infer(self, func_node: cst.FunctionDef) -> Any:
        """Return inferred return unit (pint.Unit | None), or _SENTINEL if no return found."""
        self._block(func_node.body.body)
        return self._return

    # Statement dispatch

    def _block(self, stmts: Any) -> None:
        for stmt in stmts:
            self._stmt(stmt)

    def _stmt(self, node: Any) -> None:
        if isinstance(node, cst.SimpleStatementLine):
            for small in node.body:
                self._small(small)
        elif isinstance(node, cst.If):
            self._if(node)
        elif isinstance(node, (cst.For, cst.While)):
            self._block(node.body.body)
        elif isinstance(node, cst.With):
            self._block(node.body.body)
        elif isinstance(node, cst.Try):
            self._block(node.body.body)
            env_after_try = dict(self.env)
            for handler in node.handlers:
                self.env = dict(env_after_try)
                self._block(handler.body.body)
            if node.orelse is not None:
                self.env = dict(env_after_try)
                self._block(node.orelse.body.body)
            if node.finalbody is not None:
                self._block(node.finalbody.body.body)

    def _small(self, node: Any) -> None:
        if isinstance(node, cst.Assign):
            unit = self._expr(node.value)
            for t in node.targets:
                self._bind(t.target, unit)
        elif isinstance(node, cst.AnnAssign) and node.value is not None:
            self._bind(node.target, self._expr(node.value))
        elif isinstance(node, cst.AugAssign):
            lhs = self.env.get(node.target.value) if isinstance(node.target, cst.Name) else None
            self._bind(node.target, self._binop(node.operator, lhs, self._expr(node.value)))
        elif isinstance(node, cst.Return):
            new_ret = self._expr(node.value) if node.value is not None else None
            if (
                self._return is not _SENTINEL
                and self._return is not None
                and new_ret is not None
                and hasattr(self._return, "dimensionality")
                and hasattr(new_ret, "dimensionality")
                and self._return.dimensionality != new_ret.dimensionality
            ):
                raise TypeError(
                    f"inconsistent return dimensions: "
                    f"{self._return} ({dict(self._return.dimensionality)}) vs "
                    f"{new_ret} ({dict(new_ret.dimensionality)})"
                )
            self._return = new_ret

    def _bind(self, target: Any, unit: Any) -> None:
        if isinstance(target, cst.Name):
            self.env[target.value] = unit
        elif isinstance(target, (cst.Tuple, cst.List)):
            for el in target.elements:
                self._bind(el.value, None)  # conservative: unknown per element

    def _if(self, node: cst.If) -> None:
        env_before, ret_before = dict(self.env), self._return

        self._block(node.body.body)
        env_then, ret_then = dict(self.env), self._return

        self.env, self._return = dict(env_before), ret_before
        if isinstance(node.orelse, cst.If):
            self._if(node.orelse)
        elif isinstance(node.orelse, cst.Else):
            self._block(node.orelse.body.body)
        env_else, ret_else = dict(self.env), self._return

        # Keep unit only when both branches agree.
        self.env = {
            k: env_then.get(k) if env_then.get(k) == env_else.get(k) else None
            for k in set(env_then) | set(env_else)
        }
        if ret_then is not _SENTINEL and ret_else is not _SENTINEL:
            if ret_then == ret_else:
                self._return = ret_then
            else:
                # Both branches return: check that dimensionality agrees.
                if (
                    ret_then is not None
                    and ret_else is not None
                    and hasattr(ret_then, "dimensionality")
                    and hasattr(ret_else, "dimensionality")
                    and ret_then.dimensionality != ret_else.dimensionality
                ):
                    raise TypeError(
                        f"inconsistent return dimensions across branches: "
                        f"{ret_then} ({dict(ret_then.dimensionality)}) vs "
                        f"{ret_else} ({dict(ret_else.dimensionality)})"
                    )
                self._return = None  # same dimensionality, different scale: no wrapping
        else:
            self._return = ret_then if ret_then is not _SENTINEL else ret_else

    # Expression inference

    def _expr(self, node: Any) -> Any:
        if node is None:
            return None
        if isinstance(node, (cst.Integer, cst.Float, cst.Imaginary)):
            return None
        if isinstance(node, cst.Name):
            return self.env.get(node.value)
        if isinstance(node, (cst.SimpleString, cst.FormattedString, cst.ConcatenatedString)):
            return None
        if isinstance(node, cst.UnaryOperation):
            if isinstance(node.operator, (cst.Minus, cst.Plus)):
                return self._expr(node.expression)
            return None
        if isinstance(node, cst.BinaryOperation):
            return self._binop(
                node.operator, self._expr(node.left), self._expr(node.right), node.right
            )
        if isinstance(node, (cst.BooleanOperation, cst.Comparison)):
            return None
        if isinstance(node, cst.Attribute):
            return self._attr(node)
        if isinstance(node, cst.Call):
            return self._call(node)
        if isinstance(node, cst.IfExp):
            t, f = self._expr(node.body), self._expr(node.orelse)
            return t if t == f else None
        if isinstance(node, cst.List):
            return _ListReturn("list", [self._expr(el.value) for el in node.elements])
        if isinstance(node, cst.Tuple):
            return _ListReturn("tuple", [self._expr(el.value) for el in node.elements])
        if isinstance(node, cst.Subscript):
            container = self._expr(node.value)
            if isinstance(container, _ListReturn) and node.slice:
                slice_node = node.slice[0].slice
                if isinstance(slice_node, cst.Index):
                    idx = _eval_literal(slice_node.value)
                    if idx is not None:
                        n = len(container.units)
                        i = int(idx)
                        if -n <= i < n:
                            return container.units[i]
                return None  # unknown index: conservative
            return container
        return None

    def _get_obj_map(self, node: Any) -> dict[str, Any] | None:
        """Return the attribute unit map for a node if its env entry is a dict."""
        if isinstance(node, cst.Name):
            val = self.env.get(node.value)
            return val if isinstance(val, dict) else None
        if isinstance(node, cst.Attribute):
            parent = self._get_obj_map(node.value)
            if parent is not None:
                val = parent.get(node.attr.value)
                return val if isinstance(val, dict) else None
        return None

    def _attr(self, node: cst.Attribute) -> Any:
        name = node.attr.value
        if name == "magnitude":
            return None
        if isinstance(node.value, cst.Name):
            reg_inst = self.ureg_vars.get(node.value.value)
            if reg_inst is not None:
                try:
                    return (1 * getattr(reg_inst, name)).to_base_units().units
                except Exception:
                    pass
        obj_map = self._get_obj_map(node.value)
        if obj_map is not None:
            return obj_map.get(name)
        return None

    def _call(self, node: cst.Call) -> Any:
        # x.to_base_units() -> unit of x
        if (
            isinstance(node.func, cst.Attribute)
            and node.func.attr.value == "to_base_units"
            and not node.args
        ):
            return self._expr(node.func.value)

        # x.to(unit) -> unit of the argument
        if isinstance(node.func, cst.Attribute) and node.func.attr.value == "to" and node.args:
            return self._expr(node.args[0].value)

        # cast("Quantity", x) -> unit of x
        if (
            isinstance(node.func, cst.Name)
            and node.func.value == "cast"
            and len(node.args) >= 2
            and isinstance(node.args[0].value, cst.SimpleString)
            and "Quantity" in node.args[0].value.value
        ):
            return self._expr(node.args[1].value)

        # Method calls that preserve the receiver's unit (e.g. arr.sum(), arr.mean()).
        # Skip when the receiver is a module (e.g. np.sum is a function, not a method).
        if isinstance(node.func, cst.Attribute):
            if node.func.attr.value in _UNIT_PRESERVING_METHODS:
                receiver = node.func.value
                receiver_is_module = isinstance(receiver, cst.Name) and isinstance(
                    self.module_globals.get(receiver.value), types.ModuleType
                )
                if not receiver_is_module:
                    return self._expr(receiver)

        arg_units = [self._expr(a.value) for a in node.args if a.keyword is None]
        func_name = self._resolve_name(node.func)

        # Method call on a known object: self.rate(...) -> look up type(self).rate
        if isinstance(node.func, cst.Attribute) and isinstance(node.func.value, cst.Name):
            receiver_name = node.func.value.value
            method_name = node.func.attr.value
            obj = self.param_objects.get(receiver_name)
            if obj is not None:
                # np.random.Generator methods always return plain (dimensionless) values.
                if (
                    isinstance(obj, np.random.Generator)
                    and method_name in _RNG_DIMENSIONLESS_METHODS
                ):
                    return None
                obj_qualname = f"{type(obj).__qualname__}.{method_name}"
                if obj_qualname in self.return_units:
                    return self.return_units[obj_qualname]

        # @unit_jit callee already inferred
        if func_name and func_name in self.module_globals:
            callee = self.module_globals[func_name]
            callee_qualname = getattr(callee, "__qualname__", None)
            if callee_qualname and callee_qualname in self.return_units:
                return self.return_units[callee_qualname]
            # Callee is @unit_jit but not yet inferred (e.g. because it uses RNG and was
            # disabled, or because inference runs in a different order than call order).
            # Trigger lazy inference now so the cross-call unit is resolved correctly.
            if (
                callee_qualname
                and getattr(callee, "__unit_jit_wrapped__", False)
                and callee_qualname not in self._inferring
            ):
                lazy = self._lazy_infer_callee(
                    getattr(callee, "__wrapped__", None), callee_qualname, node, arg_units
                )
                if lazy is not _UNKNOWN and lazy is not _SENTINEL:
                    return lazy

        if func_name in _KNOWN_CALLS:
            try:
                return _KNOWN_CALLS[func_name](arg_units)
            except Exception:
                return _UNKNOWN

        return _UNKNOWN

    def _lazy_infer_callee(
        self,
        inner_func: Any,
        callee_qualname: str,
        node: cst.Call,
        arg_units: list[Any],
    ) -> Any:
        """Infer return unit of an un-inferred @unit_jit callee without touching global state.

        Constructs dummy arguments from the inferred arg_units plus actual runtime objects
        for non-Quantity parameters (e.g. np.random.Generator), then runs inference
        recursively. The result is returned to the caller's inference pass only; global
        _return_units/_return_registry are updated in the normal boundary-entry path when
        the callee is first called at the module boundary.
        """
        if inner_func is None:
            return _UNKNOWN
        reg = next(iter(self.ureg_vars.values()), None)
        if reg is None:
            # Module has no ureg global; fall back to the registry embedded in a unit object.
            for u in arg_units:
                if u is not None and u is not _UNKNOWN and hasattr(u, "_REGISTRY"):
                    reg = u._REGISTRY  # noqa: SLF001
                    break
        if reg is None:
            return _UNKNOWN

        call_positional = [a for a in node.args if a.keyword is None]
        dummy_args: list[Any] = []
        for call_arg, unit in zip(call_positional, arg_units):
            if unit is None or unit is _UNKNOWN:
                # Non-Quantity arg: pass the actual runtime object when accessible by name.
                if isinstance(call_arg.value, cst.Name):
                    obj = self.param_objects.get(call_arg.value.value)
                    if obj is not None:
                        dummy_args.append(obj)
                        continue
                dummy_args.append(1.0)
            else:
                try:
                    dummy_args.append(reg.Quantity(1.0, unit))
                except Exception:
                    dummy_args.append(1.0)

        self._inferring.add(callee_qualname)
        try:
            inferred, _ = infer_return_units(inner_func, tuple(dummy_args), {}, self.return_units)
        except TypeError:
            return _UNKNOWN
        except Exception:
            return _UNKNOWN
        finally:
            self._inferring.discard(callee_qualname)

        if inferred is _SENTINEL or inferred is _UNKNOWN:
            return _UNKNOWN
        return inferred

    def _resolve_name(self, node: Any) -> str:
        if isinstance(node, cst.Name):
            return node.value
        if isinstance(node, cst.Attribute):
            p = self._resolve_name(node.value)
            return f"{p}.{node.attr.value}" if p else node.attr.value
        return ""

    def _binop(self, op: Any, left: Any, right: Any, right_node: Any = None) -> Any:
        if isinstance(left, _ListReturn) or isinstance(right, _ListReturn):
            return _UNKNOWN
        try:
            if isinstance(op, (cst.Add, cst.Subtract)):
                # Propagate _UNKNOWN without raising: cannot check an unknown unit.
                if left is _UNKNOWN or right is _UNKNOWN:
                    return left if right is _UNKNOWN else right
                if left is None or right is None:
                    return left if left is not None else right
                if left.dimensionality != right.dimensionality:
                    raise TypeError(
                        f"cannot add/subtract {left} and {right}: "
                        f"{dict(left.dimensionality)} vs {dict(right.dimensionality)}"
                    )
                return left
            if isinstance(op, cst.Multiply):
                return _unit_mul(left, right)
            if isinstance(op, (cst.Divide, cst.FloorDivide)):
                return _unit_div(left, right)
            if isinstance(op, cst.Power):
                if left is None:
                    return None
                exp = _eval_literal(right_node) if right_node is not None else None
                return _unit_pow(left, exp) if exp is not None else None
            if isinstance(op, cst.Modulo):
                return left
        except TypeError:
            raise
        except Exception:
            return None
        return None


# ---------------------------------------------------------------------------
# Source helpers
# ---------------------------------------------------------------------------


def _strip_decorators(src: str) -> str:
    """Remove leading decorator lines from a function's source before rewriting."""
    lines = src.splitlines()
    while lines and lines[0].lstrip().startswith("@"):
        lines.pop(0)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def infer_return_units(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    return_units: dict[str, Any],
    default_ureg: UnitRegistry | None = None,
) -> tuple[Any, UnitRegistry | None]:
    """Abstract-interpret func's body with units derived from args.

    Returns (unit_info, registry) on success, or (_SENTINEL, None) if
    inference cannot be performed (source unavailable, parse error, etc.).
    Raises TypeError for dimensional errors detected in the function body.
    """
    try:
        src = textwrap.dedent(_strip_decorators(inspect.getsource(func)))
        tree = cst.parse_module(src)
        if not tree.body or not isinstance(tree.body[0], cst.FunctionDef):
            return _SENTINEL, None
        func_node = tree.body[0]

        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())

        def _arg_unit(arg: Any) -> Any:
            if isinstance(arg, _QUANTITY_TYPES):
                return arg.to_base_units().units
            if isinstance(arg, list):
                return _ListReturn("list", [_arg_unit(el) for el in arg])
            if isinstance(arg, tuple):
                return _ListReturn("tuple", [_arg_unit(el) for el in arg])
            attr_units = _extract_attr_units(arg)
            return attr_units if attr_units else None

        env: dict[str, Any] = {name: _arg_unit(arg) for name, arg in zip(param_names, args)}
        env.update({name: _arg_unit(arg) for name, arg in kwargs.items()})

        module_globals = func.__globals__
        ureg_vars = {k: v for k, v in module_globals.items() if isinstance(v, _REGISTRY_TYPES)}

        param_objects = {
            name: arg
            for name, arg in zip(param_names, args)
            if not isinstance(arg, _QUANTITY_TYPES)
        }
        param_objects.update(
            {name: arg for name, arg in kwargs.items() if not isinstance(arg, _QUANTITY_TYPES)}
        )

        inferred = _UnitInferrer(env, ureg_vars, module_globals, return_units, param_objects).infer(
            func_node
        )

        if inferred is _SENTINEL:
            return None, None  # void function
        if inferred is _UNKNOWN:
            # Body was fully traversed; all ops between known units were already checked.
            # _UNKNOWN means the return expression depends on an unresolvable call,
            # so return unit cannot be determined; JIT will be disabled for this function.
            return _SENTINEL, None

        def _find_reg(arg: Any) -> Any:
            if isinstance(arg, _QUANTITY_TYPES):
                return getattr(arg, "_REGISTRY", None)  # noqa: SLF001
            if isinstance(arg, (list, tuple)):
                return next((r for el in arg if (r := _find_reg(el)) is not None), None)
            return None

        reg = next((r for a in args if (r := _find_reg(a)) is not None), None)
        reg = reg or next((r for v in kwargs.values() if (r := _find_reg(v)) is not None), None)
        reg = reg or next(iter(ureg_vars.values()), default_ureg)
        return inferred, reg
    except TypeError:
        raise
    except Exception as exc:
        _log.debug("unit inference failed for '%s': %s", func.__qualname__, exc)
        return _SENTINEL, None
