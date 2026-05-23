"""Regression tests for cyclic object graphs passed to @unit_jit functions.

Prior to the fix, _snapshot() and _extract_attr_units() would recurse
infinitely on objects that reference themselves or form a reference cycle,
raising RecursionError.  Each test asserts that the decorated function
produces the correct result despite the cycle.
"""

from pint import Quantity, UnitRegistry

from unit_jit import unit_jit

ureg = UnitRegistry()


# ---------------------------------------------------------------------------
# Direct self-reference: obj.self_ref = obj
# ---------------------------------------------------------------------------


class _SelfRef:
	alpha: Quantity

	def __init__(self) -> None:
		self.alpha = 2.0 * ureg.m
		self.self_ref = self  # cycle: self → self


@unit_jit
def _step_selfref(model: _SelfRef, dt: Quantity) -> Quantity:
	return model.alpha / dt


def test_direct_self_reference() -> None:
	m = _SelfRef()
	result = _step_selfref(m, 1.0 * ureg.s)  # warm-up (triggers inference + snapshot)
	result = _step_selfref(m, 2.0 * ureg.s)
	assert isinstance(result, Quantity)
	assert abs(result.to("m/s").magnitude - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# Mutual reference: A.ref = B, B.ref = A
# ---------------------------------------------------------------------------


class _Params:
	owner: "_Model"


class _Model:
	alpha: Quantity
	params: _Params

	def __init__(self) -> None:
		self.alpha = 6.0 * ureg.m
		self.params = _Params()
		self.params.owner = self  # cycle: model → params → model


@unit_jit
def _step_mutual(model: _Model, dt: Quantity) -> Quantity:
	return model.alpha / dt


def test_mutual_reference() -> None:
	m = _Model()
	result = _step_mutual(m, 1.0 * ureg.s)  # warm-up
	result = _step_mutual(m, 3.0 * ureg.s)
	assert isinstance(result, Quantity)
	assert abs(result.to("m/s").magnitude - 2.0) < 1e-12


# ---------------------------------------------------------------------------
# Three-node cycle: A → B → C → A
# ---------------------------------------------------------------------------


class _NodeC:
	root: "_NodeA"


class _NodeB:
	child: _NodeC


class _NodeA:
	value: Quantity
	child: _NodeB

	def __init__(self) -> None:
		self.value = 9.0 * ureg.m
		self.child = _NodeB()
		self.child.child = _NodeC()
		self.child.child.root = self  # cycle: A → B → C → A


@unit_jit
def _step_three_node(node: _NodeA, dt: Quantity) -> Quantity:
	return node.value / dt


def test_three_node_cycle() -> None:
	a = _NodeA()
	result = _step_three_node(a, 1.0 * ureg.s)  # warm-up
	result = _step_three_node(a, 3.0 * ureg.s)
	assert isinstance(result, Quantity)
	assert abs(result.to("m/s").magnitude - 3.0) < 1e-12
