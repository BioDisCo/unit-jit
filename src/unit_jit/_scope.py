"""Lexical binding information shared by inference and source rewriting."""

import libcst as cst
from libcst.metadata import MetadataWrapper, ScopeProvider
from libcst.metadata.scope_provider import BuiltinScope, GlobalScope


class LexicalBindings:
    def __init__(self, tree: cst.Module, nonlocals: tuple[str, ...] = ()) -> None:
        self.nonlocals = frozenset(nonlocals)
        self.scopes = MetadataWrapper(tree, unsafe_skip_copy=True).resolve(ScopeProvider)

    def external(self, node: cst.BaseExpression) -> bool:
        """Whether a reference resolves outside a local Python scope."""
        while isinstance(node, cst.Attribute):
            node = node.value
        if not isinstance(node, cst.Name):
            return False
        if node.value in self.nonlocals:
            return False
        scope = self.scopes.get(node)
        if scope is None:
            return False
        assignments = scope[node.value]
        return not assignments or all(
            isinstance(assignment.scope, (GlobalScope, BuiltinScope)) for assignment in assignments
        )


def conversion_scale(node, registries, bindings):
    """Resolve a literal conversion target using lexical bindings and unit algebra.

    A Quantity supplied to Pint.to specifies its units, not its magnitude.
    Unknown/dynamic targets have no compile-time scale proof.
    """
    import operator

    operations = {cst.Multiply: operator.mul, cst.Divide: operator.truediv, cst.Power: operator.pow}

    def evaluate(expr):
        if isinstance(expr, (cst.Integer, cst.Float)):
            return expr.evaluated_value
        if isinstance(expr, cst.UnaryOperation):
            signs = {cst.Plus: operator.pos, cst.Minus: operator.neg}
            return signs[type(expr.operator)](evaluate(expr.expression))
        if isinstance(expr, cst.BinaryOperation):
            return operations[type(expr.operator)](evaluate(expr.left), evaluate(expr.right))
        if (
            isinstance(expr, cst.Attribute)
            and isinstance(expr.value, cst.Name)
            and bindings.external(expr.value)
        ):
            return getattr(registries[expr.value.value], expr.attr.value)
        raise ValueError("conversion target is not a literal unit expression")

    try:
        target = evaluate(node)
        unit = getattr(target, "units", target)
        return float((1 * unit).to_base_units().magnitude)
    except (AttributeError, KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
