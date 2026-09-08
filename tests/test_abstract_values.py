"""An admitted expression must produce a domain value, not an arbitrary object."""

import libcst as cst
import pytest
from _jit_state import expect_execution
from pint import UnitRegistry

from unit_jit import unit_jit
from unit_jit._inferrer import _UnitInferrer

ureg = UnitRegistry()


def add(x, y):
    return x + y


@pytest.mark.parametrize("malformed", [None, ureg.m, {"x": ureg.m}, object()])
def test_untyped_inference_result_cannot_authorize_stripping(monkeypatch, malformed):
    original = _UnitInferrer._expr_impl

    def faulty(self, node):
        if isinstance(node, cst.BinaryOperation):
            # Account for every child so that this checks representation validity
            # independently of the existing missing-subtree audit tests.
            self._expr(node.left)
            self._expr(node.right)
            return malformed
        return original(self, node)

    monkeypatch.setattr(_UnitInferrer, "_expr_impl", faulty)
    wrapped = unit_jit(add)
    with expect_execution(wrapped, "fallback"):
        assert wrapped(1 * ureg.m, 2 * ureg.cm) == 1.02 * ureg.m


def test_typed_quantity_result_still_strips():
    wrapped = unit_jit(add)
    with expect_execution(wrapped):
        assert wrapped(1 * ureg.m, 2 * ureg.cm) == 1.02 * ureg.m
