"""Abstract interpretation of Pint units through function CSTs.

Propagates units symbolically through all branches of a function body,
catching dimensional errors (e.g. adding meters to seconds) without
executing the function.
"""

from __future__ import annotations

import builtins
import functools
import inspect
import logging
import math
import textwrap
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import libcst as cst
import numpy as np
from pint import Quantity, Unit, UnitRegistry

from ._admission import CallableBinding, immutable_default, native_scalar, plain_storage
from ._calls import FUNCTIONS, METHODS
from ._dispatch import DispatchBinding, QuantityDispatch
from ._scope import LexicalBindings, conversion_scale
from ._values import (
    _SENTINEL,
    _UNKNOWN,
    PLAIN,
    AbstractValue,
    LambdaValue,
    ObjectValue,
    QuantityValue,
    SequenceValue,
    _copy_env,
    _join,
    _known,
    _same_unit,
    _unit_div,
    _unit_mul,
    _unit_pow,
    _Unsupported,
    contains_quantity,
)


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
        # pintrs exposes facade classes whose metaclasses accept several Rust
        # representations. Keep the concrete classes for dispatch verification.
        core = getattr(_pintrs, "_core", None)
        for name in ("Quantity", "RustArrayQuantity"):
            concrete = getattr(core, name, None)
            if isinstance(concrete, type) and concrete not in qtypes:
                qtypes.append(concrete)
    except ImportError:
        pass
    return tuple(qtypes), tuple(rtypes), tuple(utypes)


_QUANTITY_TYPES, _REGISTRY_TYPES, _UNIT_TYPES = _collect_types()
_QUANTITY_DISPATCH = QuantityDispatch(_QUANTITY_TYPES)

_log = logging.getLogger("unit_jit")


@dataclass
class _Binding:
    owner: Any
    name: str
    value: Any

    def unchanged(self) -> bool:
        current = (
            self.owner.get(self.name, _UNKNOWN)
            if isinstance(self.owner, dict)
            else getattr(self.owner, self.name, _UNKNOWN)
        )
        return current is self.value


@dataclass
class _InferenceContext:
    """Dependencies and recursion state shared by an entire inference traversal."""

    active: set[Callable[..., Any]] = field(default_factory=set)
    callees: set[Callable[..., Any]] = field(default_factory=set)
    bindings: list[_Binding | DispatchBinding | CallableBinding] = field(default_factory=list)
    plain_callees: set[Callable[..., Any]] = field(default_factory=set)
    array_inputs: bool = False
    units: dict[tuple[Any, ...], Any] = field(default_factory=dict)
    unit_origins: dict[int, Any] = field(default_factory=dict)
    quantity_sources: dict[int, Any] = field(default_factory=dict)

    def canonical_unit(self, unit: Any, registry: Any) -> QuantityValue:
        """Intern equivalent abstract units using their verified registry origin."""
        if isinstance(unit, QuantityValue):
            unit = unit.unit
        key = (id(registry), tuple(sorted(unit.dimensionality.items())))
        canonical = self.units.setdefault(key, QuantityValue(unit))
        # Keep canonical units and their original registries alive during inference.
        self.unit_origins[id(canonical)] = registry
        return canonical


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


# ---------------------------------------------------------------------------
# Attribute unit extraction
# ---------------------------------------------------------------------------


def _argument_units(
    arg: Any, memo: dict[int, Any] | None = None, context: _InferenceContext | None = None
) -> AbstractValue:
    """Extract a recursive abstract value, preserving object/container aliases."""
    if memo is None:
        memo = {}
    if isinstance(arg, _QUANTITY_TYPES):
        if context is not None and isinstance(arg.magnitude, np.ndarray):
            context.array_inputs = True
        base_unit = arg.to_base_units().units
        if context is not None:
            context.quantity_sources[id(arg)] = arg
        if context is None:
            return QuantityValue(base_unit)
        return context.canonical_unit(base_unit, arg._REGISTRY)
    if id(arg) in memo:
        return memo[id(arg)]
    if isinstance(arg, (list, tuple)):
        kind = "list" if isinstance(arg, list) else "tuple"
        cls = type(arg) if hasattr(type(arg), "_fields") else None
        result = SequenceValue("namedtuple" if cls else kind, [], cls, external=True)
        memo[id(arg)] = result
        result.units = [_argument_units(el, memo, context) for el in arg]
        if not cls and result.units and all(_same_unit(u, result.units[0]) for u in result.units):
            result.units = result.units[:1]
            result.repeated = True
        return result
    if hasattr(arg, "__dict__") and not callable(arg) and not hasattr(arg, "__array_interface__"):
        if not plain_storage(arg):
            raise _Unsupported("custom attribute protocols require original execution")
        attrs = ObjectValue()
        memo[id(arg)] = attrs
        attrs.update(
            {name: _argument_units(value, memo, context) for name, value in vars(arg).items()}
        )
        return attrs
    return PLAIN


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
        param_objects: dict[str, Any] | None = None,
        context: _InferenceContext | None = None,
        bindings: LexicalBindings | None = None,
    ) -> None:
        self.env: dict[str, AbstractValue] = dict(env)
        self.ureg_vars = ureg_vars
        self.module_globals = module_globals
        self.param_objects: dict[str, Any] = param_objects or {}
        self._return: Any = _SENTINEL  # _SENTINEL = no return seen yet
        self.context = context or _InferenceContext()
        self._loop_depth = 0
        self._checked: set[int] = set()
        self.bindings = bindings

    def _external(self, node: Any) -> bool:
        root = node
        while isinstance(root, cst.Attribute):
            root = root.value
        return (
            isinstance(root, cst.Name)
            and root.value not in self.env
            and (self.bindings is None or self.bindings.external(node))
        )

    def infer(self, func_node: cst.FunctionDef) -> Any:
        """Infer a return value, or NO_RETURN when the body has no explicit return."""
        returns = self._block(func_node.body.body)
        if not returns and self._return is not _SENTINEL:
            raise _Unsupported("return value can be implicit None")
        self._audit(func_node.body)
        return self._return

    def infer_lambda(self, lambda_node: cst.Lambda) -> Any:
        """Return the inferred unit of a lambda's body expression (its implicit return)."""
        result = self._expr(lambda_node.body)
        self._audit(lambda_node.body)
        return result

    def _audit(self, body: cst.CSTNode) -> None:
        checked = self._checked

        class Audit(cst.CSTVisitor):
            def on_visit(self, node):
                if isinstance(node, cst.Attribute):
                    checked.add(id(node.attr))
                elif isinstance(node, cst.Arg) and node.keyword is not None:
                    checked.add(id(node.keyword))
                elif isinstance(node, cst.Param):
                    checked.add(id(node.name))
                if isinstance(node, cst.BaseExpression) and id(node) not in checked:
                    raise _Unsupported(f"unchecked expression: {type(node).__name__}")
                return super().on_visit(node)

            def visit_Annotation(self, node):
                return False

        body.visit(Audit())

    def _reference(self, node: Any) -> None:
        """Account for names resolved by a binding or method rule."""
        if isinstance(node, (cst.Name, cst.Attribute)):
            self._checked.add(id(node))
            if isinstance(node, cst.Attribute):
                self._reference(node.value)

    def _indices(self, node: cst.Subscript) -> None:
        for element in node.slice:
            part = element.slice
            values = (
                [part.value] if isinstance(part, cst.Index) else [part.lower, part.upper, part.step]
            )
            for value in values:
                if value is not None and contains_quantity(self._expr(value)):
                    raise _Unsupported("quantity indices require original magnitudes")

    # Statement dispatch

    def _block(self, stmts: Any) -> bool:
        for stmt in stmts:
            if self._stmt(stmt):
                return True
        return False

    def _stmt(self, node: Any) -> bool:
        if isinstance(node, cst.SimpleStatementLine):
            for small in node.body:
                self._small(small)
                if isinstance(small, cst.Return):
                    return True
        elif isinstance(node, cst.If):
            return self._if(node)
        elif isinstance(node, (cst.For, cst.While)):
            self._loop(node)
        elif isinstance(node, cst.With):
            raise _Unsupported("context managers may observe stripped objects")
        elif isinstance(node, cst.Try):
            raise _Unsupported("exception paths require separate unit environments")
        else:
            raise _Unsupported(f"unsupported statement: {type(node).__name__}")
        return False

    def _loop(self, node: cst.For | cst.While) -> None:
        before = _copy_env(self.env)
        if isinstance(node, cst.For):
            iterable = self._expr(node.iter)
            if isinstance(iterable, SequenceValue):
                if not iterable.units:
                    raise _Unsupported("empty iterable")
                element = iterable.units[0]
                for other in iterable.units[1:]:
                    element = _known(_join(element, other))
            else:
                element = iterable
            self._bind(node.target, element)
        else:
            self._expr(node.test)
        self._loop_depth += 1
        try:
            self._block(node.body.body)
        finally:
            self._loop_depth -= 1
        if not _same_unit(before, {name: self.env.get(name, _UNKNOWN) for name in before}):
            raise _Unsupported("loop changes unit schemas or aliases")
        # Variables created only inside the loop are not defined on zero iterations.
        self.env = {name: self.env[name] for name in before}
        if node.orelse is not None:
            self._block(node.orelse.body.body)

    def _small(self, node: Any) -> None:
        if isinstance(node, cst.Assign):
            unit = self._expr(node.value)
            for t in node.targets:
                self._bind(t.target, unit)
        elif isinstance(node, cst.AnnAssign) and node.value is not None:
            self._bind(node.target, self._expr(node.value))
        elif isinstance(node, cst.AugAssign):
            lhs = self._expr(node.target)
            if self.context.array_inputs and contains_quantity(lhs):
                raise _Unsupported("in-place quantity array operations require original storage")
            # Map AugAssign operators to their base binary operator counterparts.
            _aug_to_binop: dict[type, Any] = {
                cst.AddAssign: cst.Add(),
                cst.SubtractAssign: cst.Subtract(),
                cst.MultiplyAssign: cst.Multiply(),
                cst.DivideAssign: cst.Divide(),
                cst.FloorDivideAssign: cst.FloorDivide(),
                cst.PowerAssign: cst.Power(),
                cst.ModuloAssign: cst.Modulo(),
            }
            base_op = _aug_to_binop.get(type(node.operator), node.operator)
            self._bind(
                node.target, _known(self._binop(base_op, lhs, self._expr(node.value), node.value))
            )
        elif isinstance(node, cst.Return):
            if self._loop_depth:
                raise _Unsupported("loop return paths require Pint")
            new_ret = self._expr(node.value) if node.value is not None else PLAIN
            self._return = _known(_join(self._return, new_ret))
        elif isinstance(node, cst.Expr):
            if not (isinstance(node.value, cst.Call) and self._mutation_call(node.value)):
                self._expr(node.value)
        elif isinstance(node, (cst.Break, cst.Continue)):
            raise _Unsupported("loop exit paths require Pint")
        elif isinstance(node, cst.Pass):
            pass
        elif isinstance(node, cst.AnnAssign):
            pass  # An annotation without a value has no runtime effect.
        else:
            raise _Unsupported(f"unsupported statement: {type(node).__name__}")

    def _bind(self, target: Any, unit: Any) -> None:
        self._checked.add(id(target))
        if isinstance(target, cst.Name):
            self.env[target.value] = unit
            self.param_objects.pop(target.value, None)
        elif isinstance(target, (cst.Tuple, cst.List)):
            if not isinstance(unit, SequenceValue) or len(unit.units) != len(target.elements):
                raise _Unsupported("unknown unpacking shape")
            for el, item in zip(target.elements, unit.units):
                self._bind(el.value, item)
        elif isinstance(target, cst.Attribute):
            existing = self._expr(target)
            if not _same_unit(existing, unit) or isinstance(unit, (ObjectValue, SequenceValue)):
                raise _Unsupported("attribute assignment changes its unit schema")
        elif isinstance(target, cst.Subscript):
            self._indices(target)
            existing = self._expr(target.value)
            if isinstance(existing, QuantityValue):
                raise _Unsupported("quantity array writes require original storage")
            if isinstance(existing, SequenceValue):
                if existing.external:
                    raise _Unsupported("mutation of an input container requires Pint")
                index = target.slice[0].slice
                i = _eval_literal(index.value) if isinstance(index, cst.Index) else None
                if existing.repeated:
                    if not existing.units or not _same_unit(existing.units[0], unit):
                        raise _Unsupported("write changes a variable-length container's units")
                elif i is not None and -len(existing.units) <= int(i) < len(existing.units):
                    existing.units[int(i)] = unit
                elif not all(_same_unit(u, unit) for u in existing.units):
                    raise _Unsupported("indexed write has unknown destination units")
            elif not _same_unit(existing, unit):
                raise _Unsupported("array write changes element units")
        else:
            raise _Unsupported("unsupported assignment target")

    def _mutation_call(self, call: cst.Call) -> bool:
        """Track in-place list mutations (.append, .extend) that alter element types."""
        if not isinstance(call.func, cst.Attribute) or not isinstance(call.func.value, cst.Name):
            return False
        var_name = call.func.value.value
        method = call.func.attr.value
        existing = self.env.get(var_name)
        if method not in {"append", "extend"}:
            return False
        if not isinstance(existing, SequenceValue) or existing.repeated or existing.external:
            raise _Unsupported("mutation of a container with unknown length")
        self._checked.add(id(call))
        self._reference(call.func)
        if len(call.args) != 1 or call.args[0].star or call.args[0].keyword:
            raise _Unsupported("unsupported mutation arguments")
        if method == "append" and call.args:
            elem_unit = self._expr(call.args[0].value)
            existing.units.append(elem_unit)
        elif method == "extend" and call.args:
            iter_unit = self._expr(call.args[0].value)
            if not isinstance(iter_unit, SequenceValue) or iter_unit.repeated:
                raise _Unsupported("extend has unknown element count")
            existing.units.extend(iter_unit.units)
        return True

    def _if(self, node: cst.If) -> bool:
        self._expr(node.test)
        env_before, ret_before = _copy_env(self.env), self._return

        then_returns = self._block(node.body.body)
        env_then, ret_then = _copy_env(self.env), self._return

        self.env, self._return = _copy_env(env_before), ret_before
        else_returns = False
        if isinstance(node.orelse, cst.If):
            else_returns = self._if(node.orelse)
        elif isinstance(node.orelse, cst.Else):
            else_returns = self._block(node.orelse.body.body)
        env_else, ret_else = _copy_env(self.env), self._return

        # Keep unit only when both branches agree.
        if not _same_unit(env_then, env_else):
            raise _Unsupported("branches have different unit schemas or aliases")
        self.env = env_then
        self._return = _known(_join(ret_then, ret_else))
        return then_returns and else_returns

    # Expression inference

    def _expr(self, node: Any) -> AbstractValue:
        self._checked.add(id(node))
        return _known(self._expr_impl(node))

    def _expr_impl(self, node: Any) -> AbstractValue:
        if node is None:
            return PLAIN
        if isinstance(node, (cst.Integer, cst.Float, cst.Imaginary)):
            return PLAIN
        if isinstance(node, cst.Name):
            if node.value in {"True", "False", "None"}:
                return PLAIN
            if node.value in self.env:
                return self.env[node.value]
            if not self._external(node):
                return _UNKNOWN
            value = self.module_globals.get(node.value, _UNKNOWN)
            if native_scalar(value) or type(value) is type:
                self.context.bindings.append(_Binding(self.module_globals, node.value, value))
                return PLAIN
            return _UNKNOWN
        if isinstance(node, cst.SimpleString):
            return PLAIN
        if isinstance(node, cst.UnaryOperation):
            unit = self._expr(node.expression)
            if isinstance(node.operator, (cst.Minus, cst.Plus)):
                return unit
            return PLAIN if unit is PLAIN else _UNKNOWN
        if isinstance(node, cst.BinaryOperation):
            return self._binop(
                node.operator, self._expr(node.left), self._expr(node.right), node.right
            )
        if isinstance(node, cst.BooleanOperation):
            return _join(self._expr(node.left), self._expr(node.right))
        if isinstance(node, cst.Comparison):
            if any(isinstance(part.operator, (cst.Is, cst.IsNot)) for part in node.comparisons):
                raise _Unsupported("identity comparisons require original objects")
            left = self._expr(node.left)
            for comparison in node.comparisons:
                right = self._expr(comparison.comparator)
                if isinstance(left, (SequenceValue, ObjectValue)) or isinstance(
                    right, (SequenceValue, ObjectValue)
                ):
                    raise _Unsupported("container comparisons require original identity semantics")
                if not _same_unit(left, right):
                    raise _Unsupported("comparison requires compatible operands")
                left = right
            return PLAIN
        if isinstance(node, cst.Attribute):
            return self._attr(node)
        if isinstance(node, cst.Call):
            return self._call(node)
        if isinstance(node, cst.IfExp):
            self._expr(node.test)
            t, f = self._expr(node.body), self._expr(node.orelse)
            return _join(t, f)
        if isinstance(node, cst.List):
            if any(isinstance(el, cst.StarredElement) for el in node.elements):
                raise _Unsupported("expanded list elements need a shape proof")
            return SequenceValue("list", [self._expr(el.value) for el in node.elements])
        if isinstance(node, cst.Tuple):
            if any(isinstance(el, cst.StarredElement) for el in node.elements):
                raise _Unsupported("expanded tuple elements need a shape proof")
            return SequenceValue("tuple", [self._expr(el.value) for el in node.elements])
        if isinstance(node, cst.ListComp):
            # Infer element unit from the elt expression, with iteration vars in scope.
            saved_env = dict(self.env)
            for_in = node.for_in
            while for_in is not None:
                iter_unit = self._expr(for_in.iter)
                # Iteration variable gets element unit of the iterable.
                elem_unit = iter_unit
                if isinstance(iter_unit, SequenceValue):
                    if not iter_unit.units:
                        raise _Unsupported("empty comprehension iterable")
                    elem_unit = iter_unit.units[0]
                    for other in iter_unit.units[1:]:
                        elem_unit = _known(_join(elem_unit, other))
                self._bind(for_in.target, elem_unit)
                for condition in for_in.ifs:
                    self._expr(condition.test)
                for_in = for_in.inner_for_in
            elt_unit = self._expr(node.elt)
            self.env = saved_env
            return SequenceValue("list", [elt_unit], repeated=True)
        if isinstance(node, cst.Subscript):
            self._indices(node)
            container = self._expr(node.value)
            if isinstance(container, SequenceValue) and node.slice:
                slice_node = node.slice[0].slice
                if isinstance(slice_node, cst.Index):
                    idx = _eval_literal(slice_node.value)
                    if idx is not None:
                        n = len(container.units)
                        i = int(idx)
                        if -n <= i < n:
                            return container.units[i]
                    # Non-literal index: use common unit if all elements agree.
                    if container.units:
                        first = container.units[0]
                        if all(u == first for u in container.units):
                            return first
                return _UNKNOWN
            return container
        if isinstance(node, cst.Lambda):
            return LambdaValue(node)
        return _UNKNOWN

    def _get_obj_map(self, node: Any) -> dict[str, Any] | None:
        """Return the attribute unit map for a node if its env entry is a dict."""
        if isinstance(node, cst.Name):
            val = self.env.get(node.value)
            return val if isinstance(val, ObjectValue) else None
        if isinstance(node, cst.Attribute):
            parent = self._get_obj_map(node.value)
            if parent is not None:
                val = parent.get(node.attr.value)
                return val if isinstance(val, ObjectValue) else None
        return None

    def _attr(self, node: cst.Attribute) -> AbstractValue:
        name = node.attr.value
        if name == "magnitude":
            if isinstance(node.value, cst.Call) and isinstance(node.value.func, cst.Attribute):
                method = node.value.func.attr.value
                if method == "to_base_units":
                    self._expr(node.value)
                    return PLAIN
                if method == "to" and len(node.value.args) == 1:
                    self._checked.add(id(node.value))
                    self._reference(node.value.func)
                    receiver = self._expr(node.value.func.value)
                    destination = self._expr(node.value.args[0].value)
                    if (
                        self.bindings is None
                        or conversion_scale(node.value.args[0].value, self.ureg_vars, self.bindings)
                        is None
                    ):
                        raise _Unsupported("conversion target lacks a static unit-scale proof")
                    if not _same_unit(receiver, destination):
                        raise _Unsupported("incompatible explicit unit conversion")
                    return PLAIN
            raise _Unsupported("bare magnitude observes the original unit scale")
        if isinstance(node.value, cst.Name):
            reg_inst = self.ureg_vars.get(node.value.value) if self._external(node.value) else None
            if reg_inst is not None:
                self._reference(node.value)
                self.context.bindings.append(
                    _Binding(self.module_globals, node.value.value, reg_inst)
                )
                try:
                    quantity = 1 * getattr(reg_inst, name)
                    self.context.quantity_sources[id(quantity)] = quantity
                    if not getattr(quantity, "_is_multiplicative", True):
                        raise _Unsupported("unit constants require multiplicative conversion")
                    return self.context.canonical_unit(quantity.to_base_units().units, reg_inst)
                except Exception:  # noqa: BLE001 — Pint raises varied types for unknown unit names
                    pass
        obj_map = self._get_obj_map(node.value)
        if obj_map is not None:
            self._expr(node.value)
            return obj_map.get(name, _UNKNOWN)
        value_unit = self._expr(node.value)
        if (
            isinstance(value_unit, SequenceValue)
            and value_unit.kind == "namedtuple"
            and value_unit.cls is not None
            and hasattr(value_unit.cls, "_fields")
        ):
            fields: tuple[str, ...] = value_unit.cls._fields  # type: ignore[attr-defined]
            try:
                return value_unit.units[fields.index(name)]
            except ValueError:
                return _UNKNOWN
        if name in {"shape", "size", "ndim"}:
            return PLAIN
        return _UNKNOWN

    def _call(self, node: cst.Call) -> AbstractValue:
        self._reference(node.func)
        # x.to_base_units() -> unit of x
        if (
            isinstance(node.func, cst.Attribute)
            and node.func.attr.value == "to_base_units"
            and not node.args
        ):
            unit = self._expr(node.func.value)
            if not isinstance(unit, QuantityValue):
                raise _Unsupported("to_base_units requires a quantity")
            return unit

        # x.to(unit) -> unit of the argument
        if isinstance(node.func, cst.Attribute) and node.func.attr.value == "to" and node.args:
            raise _Unsupported("standalone conversion needs a quantity representation")

        # cast("Quantity", x) -> unit of x
        if (
            isinstance(node.func, cst.Name)
            and node.func.value == "cast"
            and len(node.args) >= 2
            and isinstance(node.args[0].value, cst.SimpleString)
            and "Quantity" in node.args[0].value.value
        ):
            if not self._external(node.func) or self.module_globals.get("cast") is not cast:
                raise _Unsupported("cast must refer to typing.cast")
            self.context.bindings.append(_Binding(self.module_globals, "cast", cast))
            self._expr(node.args[0].value)
            return self._expr(node.args[1].value)

        all_units = [self._expr(a.value) for a in node.args]
        if any(a.star for a in node.args):
            raise _Unsupported("expanded call arguments require explicit binding")

        # Method calls that preserve the receiver's unit (e.g. arr.sum(), arr.mean()).
        # Skip when the receiver is a module (e.g. np.sum is a function, not a method).
        if isinstance(node.func, cst.Attribute):
            if node.func.attr.value in METHODS:
                receiver = node.func.value
                receiver_is_module = isinstance(receiver, cst.Name) and isinstance(
                    self.module_globals.get(receiver.value), types.ModuleType
                )
                if not receiver_is_module:
                    unit = self._expr(receiver)
                    if isinstance(unit, (ObjectValue, SequenceValue)):
                        raise _Unsupported("method receiver needs its own callable plan")
                    arguments = [
                        (None, unit),
                        *[
                            (a.keyword.value if a.keyword else None, u)
                            for a, u in zip(node.args, all_units)
                        ],
                    ]
                    result = METHODS[node.func.attr.value].infer(arguments)
                    if any(contains_quantity(u) for _, u in arguments):
                        try:
                            self.context.bindings.extend(
                                _QUANTITY_DISPATCH.guard(
                                    node.func.attr.value, self.context.quantity_sources.values()
                                )
                            )
                            if node.args:
                                concrete = (
                                    self.param_objects.get(receiver.value)
                                    if isinstance(receiver, cst.Name)
                                    else None
                                )
                                if concrete is None:
                                    raise ValueError(
                                        "method options need a concrete receiver signature"
                                    )
                                method_signature = inspect.signature(
                                    getattr(concrete, node.func.attr.value)
                                )
                                method_signature.bind(
                                    *[u for a, u in zip(node.args, all_units) if a.keyword is None],
                                    **{
                                        a.keyword.value: u
                                        for a, u in zip(node.args, all_units)
                                        if a.keyword is not None
                                    },
                                )
                        except (TypeError, ValueError, AttributeError) as exc:
                            raise _Unsupported(str(exc)) from exc
                    return result
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
                    if any(u is not PLAIN for u in all_units):
                        raise _Unsupported("RNG arguments must be plain numbers or arrays")
                    return PLAIN
                method = getattr(type(obj), method_name, None)
                if method_name in getattr(obj, "__dict__", {}):
                    raise _Unsupported("instance overrides require their own callable plan")
                self.context.bindings.append(_Binding(type(obj), method_name, method))
                original = getattr(method, "__wrapped__", method)
                if inspect.isfunction(original):
                    return self._lazy_infer_callee(
                        functools.partial(original, obj), node, method is original
                    )

        callee = self.module_globals.get(func_name) if self._external(node.func) else None
        original = getattr(callee, "__wrapped__", callee)
        if inspect.isfunction(original) or isinstance(original, functools.partial):
            self.context.bindings.append(_Binding(self.module_globals, func_name, callee))
            return self._lazy_infer_callee(original, node, callee is original)

        # Lambda stored in env from an assignment — infer it with the call-site args.
        if isinstance(node.func, cst.Name):
            env_val = self.env.get(node.func.value)
            if isinstance(env_val, LambdaValue):
                return self._infer_stored_lambda(env_val.node, node)

        if func_name in FUNCTIONS:
            if not self._external(node.func):
                raise _Unsupported("callable is locally bound")
            parts = func_name.split(".")
            expected = {"np": np, "math": math}.get(parts[0], getattr(builtins, parts[0], None))
            actual = self.module_globals.get(parts[0], getattr(builtins, parts[0], None))
            if parts[0] in self.env or expected is not actual:
                raise _Unsupported("known function name has been rebound")
            owner = self.module_globals if parts[0] in self.module_globals else builtins
            self.context.bindings.append(_Binding(owner, parts[0], actual))
            for part in parts[1:]:
                value = getattr(actual, part, _UNKNOWN)
                self.context.bindings.append(_Binding(actual, part, value))
                actual = value
            if actual is not FUNCTIONS[func_name].implementation:
                raise _Unsupported("known callable implementation has changed")
            return FUNCTIONS[func_name].infer(
                [(a.keyword.value if a.keyword else None, u) for a, u in zip(node.args, all_units)]
            )

        if func_name == "range":
            if (
                not self._external(node.func)
                or self.module_globals.get("range", range) is not range
            ):
                raise _Unsupported("range is locally bound or rebound")
            owner = self.module_globals if "range" in self.module_globals else builtins
            self.context.bindings.append(_Binding(owner, "range", range))
            if any(u is not PLAIN for u in all_units):
                raise _Unsupported("range requires plain integer arguments")
            return PLAIN

        # NamedTuple constructor: infer each field's unit from positional/keyword args.
        callee = self.module_globals.get(func_name) if self._external(node.func) else None
        if (
            callee is not None
            and isinstance(callee, type)
            and issubclass(callee, tuple)
            and hasattr(callee, "_fields")
        ):
            self.context.bindings.append(_Binding(self.module_globals, func_name, callee))
            fields: tuple[str, ...] = callee._fields  # type: ignore[attr-defined]
            field_units: dict[str, Any] = {}
            for i, a in enumerate(node.args):
                unit = self._expr(a.value)
                if a.keyword is not None:
                    field_units[a.keyword.value] = unit
                elif i < len(fields):
                    field_units[fields[i]] = unit
            return SequenceValue(
                "namedtuple", [field_units.get(f, _UNKNOWN) for f in fields], cls=callee
            )

        return _UNKNOWN

    def _infer_stored_lambda(self, lambda_node: cst.Lambda, call_node: cst.Call) -> AbstractValue:
        """Infer the return unit of a lambda stored in a local variable at the call site.

        Binds the lambda's parameters to the call-site argument units (positional from
        call_node, defaults from the lambda's default-value CST nodes evaluated in the
        current env), then evaluates the lambda body in that child scope.
        """
        pos_units = [self._expr(a.value) for a in call_node.args if a.keyword is None]
        kw_units = {
            a.keyword.value: self._expr(a.value) for a in call_node.args if a.keyword is not None
        }

        child_env = dict(self.env)
        for i, param in enumerate(lambda_node.params.params):
            pname = param.name.value
            if pname in kw_units:
                child_env[pname] = kw_units[pname]
            elif i < len(pos_units):
                child_env[pname] = pos_units[i]
            elif param.default is not None:
                # Default is a CST expression in the enclosing scope — evaluate it there.
                child_env[pname] = self._expr(param.default)

        saved_env = self.env
        self.env = child_env
        try:
            return self._expr(lambda_node.body)
        finally:
            self.env = saved_env

    def _lazy_infer_callee(
        self, inner_func: Any, node: cst.Call, plain: bool = False
    ) -> AbstractValue:
        """Infer all callees with one argument-binding path and shared recursion state."""

        def dummy(unit: Any) -> Any:
            if unit is PLAIN:
                return 1.0
            if isinstance(unit, SequenceValue):
                values = [dummy(u) for u in unit.units]
                if unit.cls is not None:
                    return unit.cls._make(values)
                return values if unit.kind == "list" else tuple(values)
            if isinstance(unit, QuantityValue):
                return unit._REGISTRY.Quantity(1.0, unit.unit)
            raise _Unsupported("callee argument has no concrete unit representation")

        def argument(arg: cst.Arg) -> Any:
            unit = self._expr(arg.value)
            if (
                isinstance(arg.value, cst.Name)
                and arg.value.value in self.param_objects
                and (unit is PLAIN or isinstance(unit, ObjectValue))
            ):
                return self.param_objects[arg.value.value]
            return dummy(unit)

        positional = []
        keywords = {}
        for arg in node.args:
            value = argument(arg)
            if arg.keyword is None:
                positional.append(value)
            else:
                keywords[arg.keyword.value] = value
        if plain:
            partial = inner_func
            while isinstance(partial, functools.partial):
                self.context.bindings.append(CallableBinding.capture(partial))
                for value in (*partial.args, *partial.keywords.values()):
                    prepared_object = any(
                        value is obj and isinstance(self.env.get(name), ObjectValue)
                        for name, obj in self.param_objects.items()
                    )
                    if not immutable_default(value) and not prepared_object:
                        raise _Unsupported("unprepared bound operands require original execution")
                partial = partial.func
            signature = inspect.signature(inner_func)
            supplied = signature.bind(*positional, **keywords)
            for name, parameter in signature.parameters.items():
                if (
                    name not in supplied.arguments
                    and parameter.default is not inspect.Parameter.empty
                ):
                    if not immutable_default(parameter.default):
                        raise _Unsupported("unprepared helper defaults require original execution")
        inferred, _ = infer_return_units(
            inner_func,
            tuple(positional),
            keywords,
            context=self.context,
        )
        if inferred is _SENTINEL or inferred is _UNKNOWN:
            return _UNKNOWN
        original = inner_func
        while isinstance(original, functools.partial):
            original = original.func
        self.context.callees.add(original)
        if plain:
            self.context.plain_callees.add(original)
        return inferred

    def _resolve_name(self, node: Any) -> str:
        if isinstance(node, cst.Name):
            return node.value
        if isinstance(node, cst.Attribute):
            p = self._resolve_name(node.value)
            return f"{p}.{node.attr.value}" if p else node.attr.value
        return ""

    def _binop(self, op: Any, left: Any, right: Any, right_node: Any = None) -> AbstractValue:
        operands = [unit for unit in (left, right) if isinstance(unit, QuantityValue)]
        origins = [self.context.unit_origins.get(id(unit)) for unit in operands]
        known_origins = [origin for origin in origins if origin is not None]
        if known_origins and any(origin is not known_origins[0] for origin in known_origins):
            raise _Unsupported("arithmetic combines different registry origins")
        result = self._binop_impl(op, left, right, right_node)
        if (
            isinstance(result, QuantityValue)
            and origins
            and all(origin is not None for origin in origins)
        ):
            return self.context.canonical_unit(result, origins[0])
        return result

    def _binop_impl(self, op: Any, left: Any, right: Any, right_node: Any = None) -> AbstractValue:
        # List repetition: [expr] * n  →  result keeps same structure as the list.
        if isinstance(op, cst.Multiply):
            sequence = (
                left
                if isinstance(left, SequenceValue) and right is PLAIN
                else (right if isinstance(right, SequenceValue) and left is PLAIN else None)
            )
            if sequence is not None:
                if sequence.units and all(_same_unit(u, sequence.units[0]) for u in sequence.units):
                    return SequenceValue(sequence.kind, sequence.units[:1], repeated=True)
                return _UNKNOWN
        if isinstance(left, SequenceValue) or isinstance(right, SequenceValue):
            return _UNKNOWN
        try:
            if isinstance(op, (cst.Add, cst.Subtract)):
                # Propagate _UNKNOWN without raising: cannot check an unknown unit.
                if left is _UNKNOWN or right is _UNKNOWN:
                    return _UNKNOWN
                if left is PLAIN or right is PLAIN:
                    quantity = left if left is not PLAIN else right
                    if quantity is not PLAIN and quantity.dimensionality:
                        # A numeric parameter may be zero. Preserve Pint's value-dependent
                        # exception by falling back rather than guessing from the first call.
                        return _UNKNOWN
                    return quantity
                if left.dimensionality != right.dimensionality:
                    msg = (
                        f"cannot add/subtract {left} and {right}: "
                        f"{dict(left.dimensionality)} vs {dict(right.dimensionality)}"
                    )
                    _log.warning("dimension mismatch: %s", msg)
                    raise TypeError(msg)
                if left._REGISTRY is not right._REGISTRY:
                    return _UNKNOWN
                return left
            if isinstance(op, cst.Multiply):
                return _unit_mul(left, right)
            if isinstance(op, cst.FloorDivide):
                if left is not PLAIN or right is not PLAIN:
                    raise _Unsupported("quantity floor division requires backend semantics")
                return PLAIN
            if isinstance(op, cst.Divide):
                return _unit_div(left, right)
            if isinstance(op, cst.Power):
                if right is not PLAIN and (
                    not isinstance(right, QuantityValue) or right.dimensionality
                ):
                    return _UNKNOWN
                if left is PLAIN:
                    return PLAIN
                exp = _eval_literal(right_node) if right_node is not None else None
                return _unit_pow(left, exp) if exp is not None else _UNKNOWN
            if isinstance(op, cst.Modulo):
                return left if _same_unit(left, right) else _UNKNOWN
        except TypeError:
            raise
        except Exception:
            return _UNKNOWN
        return _UNKNOWN


# ---------------------------------------------------------------------------
# Source helpers
# ---------------------------------------------------------------------------


def _strip_decorators(src: str) -> str:
    """Remove leading decorator lines from a function's source before rewriting."""
    lines = src.splitlines()
    while lines and lines[0].lstrip().startswith("@"):
        lines.pop(0)
    return "\n".join(lines)


class _LambdaFinder(cst.CSTTransformer):
    """Collect the first Lambda node in a parsed source fragment (read-only use)."""

    def __init__(self) -> None:
        super().__init__()
        self.node: cst.Lambda | None = None

    def visit_Lambda(self, node: cst.Lambda) -> bool:
        if self.node is None:
            self.node = node
        return False  # do not descend into nested lambdas


def _find_lambda(tree: cst.Module) -> cst.Lambda | None:
    """Return the first Lambda in *tree*, or None.

    Used when a callable's source is not a standalone ``def`` (e.g. a lambda stored
    in a list or assigned to a subscript); inference then runs over the lambda body.
    """
    finder = _LambdaFinder()
    tree.visit(finder)
    return finder.node


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def infer_return_units(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    default_ureg: UnitRegistry | None = None,
    *,
    context: _InferenceContext | None = None,
) -> tuple[AbstractValue, UnitRegistry | None]:
    """Abstract-interpret func's body with units derived from args.

    Returns (unit_info, registry) on success, or (_SENTINEL, None) if
    inference cannot be performed (source unavailable, parse error, etc.).
    Raises TypeError for dimensional errors detected in the function body.
    """
    # Unwrap functools.partial so getsource and signature work on the raw function,
    # and prepend the bound positional/keyword args to the call arguments.
    import functools

    while isinstance(func, functools.partial):
        kwargs = {**func.keywords, **kwargs}
        args = func.args + args
        func = func.func

    context = context or _InferenceContext()
    if func in context.active:
        return _SENTINEL, None
    context.active.add(func)
    context.bindings.append(CallableBinding.capture(func))
    try:
        src = textwrap.dedent(_strip_decorators(inspect.getsource(func)))
        tree = cst.parse_module(src)
        func_node: cst.FunctionDef | None = None
        lambda_node: cst.Lambda | None = None
        if tree.body and isinstance(tree.body[0], cst.FunctionDef):
            func_node = tree.body[0]
        else:
            # Not a standalone def (e.g. a lambda assigned to a subscript): infer its body.
            lambda_node = _find_lambda(tree)
            if lambda_node is None:
                return _SENTINEL, None

        bound = inspect.signature(func).bind(*args, **kwargs)
        bound.apply_defaults()
        memo: dict[int, Any] = {}
        env = {
            name: _argument_units(value, memo, context) for name, value in bound.arguments.items()
        }
        module_globals = func.__globals__
        ureg_vars = {k: v for k, v in module_globals.items() if isinstance(v, _REGISTRY_TYPES)}
        param_objects = dict(bound.arguments)
        inferrer = _UnitInferrer(
            env,
            ureg_vars,
            module_globals,
            param_objects,
            context,
            LexicalBindings(tree, func.__code__.co_freevars),
        )
        inferred = (
            inferrer.infer(func_node)
            if func_node is not None
            else inferrer.infer_lambda(cast("cst.Lambda", lambda_node))
        )

        if inferred is _SENTINEL:
            return PLAIN, None  # void function
        if inferred is _UNKNOWN:
            # Body was fully traversed; all ops between known units were already checked.
            # _UNKNOWN means the return expression depends on an unresolvable call,
            # so return unit cannot be determined; JIT will be disabled for this function.
            return _SENTINEL, None

        _find_reg_visited: set[int] = set()

        def _find_reg(arg: Any) -> Any:
            if isinstance(arg, _QUANTITY_TYPES):
                return getattr(arg, "_REGISTRY", None)  # noqa: SLF001
            arg_id = id(arg)
            if arg_id in _find_reg_visited:
                return None
            _find_reg_visited.add(arg_id)
            if isinstance(arg, (list, tuple)):
                return next((r for el in arg if (r := _find_reg(el)) is not None), None)
            # Fall back to scanning object attributes (e.g. a dataclass / BCRN instance).
            try:
                for v in vars(arg).values():
                    r = _find_reg(v)
                    if r is not None:
                        return r
            except TypeError:
                pass
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
    finally:
        context.active.discard(func)
