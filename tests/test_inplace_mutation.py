"""Regression tests: in-place strip+restore correctly propagates mutations.

Prior to this fix, _to_fast() created a snapshot copy of object arguments.
Mutations inside the JIT loop were lost — the original object retained its
pre-call Pint state.  Each test below verifies that:
  1. The computation ran correctly (result matches plain-Pint baseline).
  2. The original object's Pint attrs reflect the post-call state.
"""

import numpy as np
import pytest
from pint import Quantity, UnitRegistry

from unit_jit import unit_jit

ureg = UnitRegistry()


# ---------------------------------------------------------------------------
# Simple scalar accumulator
# ---------------------------------------------------------------------------


class _Counter:
    value: Quantity

    def __init__(self) -> None:
        self.value = 0.0 * ureg.m

    def step(self, dx: Quantity) -> None:
        self.value = self.value + dx


@unit_jit
def _accumulate(counter: _Counter, dx: Quantity, n: int) -> None:
    for _ in range(n):
        counter.step(dx)


def test_scalar_mutation_propagates() -> None:
    c = _Counter()
    dx = 0.5 * ureg.m
    _accumulate(c, dx, 10)
    assert isinstance(c.value, Quantity)
    assert abs(c.value.to("m").magnitude - 5.0) < 1e-12


def test_scalar_mutation_matches_pint_baseline() -> None:
    c_jit = _Counter()
    c_ref = _Counter()
    dx = 0.3 * ureg.m
    _accumulate(c_jit, dx, 7)
    for _ in range(7):
        c_ref.step(dx)
    assert abs(c_jit.value.to("m").magnitude - c_ref.value.to("m").magnitude) < 1e-12


# ---------------------------------------------------------------------------
# Array state — models a simple Euler integrator
# ---------------------------------------------------------------------------


class _EulerState:
    x: Quantity  # position array  [m]
    t: Quantity  # current time    [s]

    def __init__(self, n: int) -> None:
        self.x = np.zeros(n) * ureg.m
        self.t = 0.0 * ureg.s

    def advance(self, v: Quantity, dt: Quantity) -> None:
        self.x = self.x + v * dt
        self.t = self.t + dt


@unit_jit
def _euler_run(state: _EulerState, v: Quantity, dt: Quantity, n: int) -> None:
    for _ in range(n):
        state.advance(v, dt)


def test_array_mutation_propagates() -> None:
    state = _EulerState(4)
    v = np.array([1.0, 2.0, 3.0, 4.0]) * ureg("m/s")
    dt = 0.1 * ureg.s
    _euler_run(state, v, dt, 5)
    assert isinstance(state.x, Quantity)
    assert isinstance(state.t, Quantity)
    expected_x = np.array([0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(state.x.to("m").magnitude, expected_x, rtol=1e-12)
    assert abs(state.t.to("s").magnitude - 0.5) < 1e-12


def test_array_mutation_matches_pint_baseline() -> None:
    state_jit = _EulerState(3)
    state_ref = _EulerState(3)
    v = np.array([1.0, -1.0, 0.5]) * ureg("m/s")
    dt = 0.05 * ureg.s
    n = 20
    _euler_run(state_jit, v, dt, n)
    for _ in range(n):
        state_ref.advance(v, dt)
    np.testing.assert_allclose(
        state_jit.x.to("m").magnitude, state_ref.x.to("m").magnitude, rtol=1e-10
    )
    assert abs(state_jit.t.to("s").magnitude - state_ref.t.to("s").magnitude) < 1e-12


# ---------------------------------------------------------------------------
# Nested objects — outer holds inner; mutation on inner must propagate
# ---------------------------------------------------------------------------


class _Inner:
    count: Quantity

    def __init__(self) -> None:
        self.count = 0.0 * ureg.dimensionless


class _Outer:
    inner: _Inner
    total: Quantity

    def __init__(self) -> None:
        self.inner = _Inner()
        self.total = 0.0 * ureg.m

    def step(self, dx: Quantity) -> None:
        self.total = self.total + dx
        self.inner.count = self.inner.count + 1.0 * ureg.dimensionless


@unit_jit
def _nested_run(obj: _Outer, dx: Quantity, n: int) -> None:
    for _ in range(n):
        obj.step(dx)


def test_nested_mutation_propagates() -> None:
    obj = _Outer()
    _nested_run(obj, 2.0 * ureg.m, 6)
    assert abs(obj.total.to("m").magnitude - 12.0) < 1e-12
    assert abs(obj.inner.count.to("dimensionless").magnitude - 6.0) < 1e-12


# ---------------------------------------------------------------------------
# Cyclic reference — obj.self_ref = obj; must not recurse infinitely
# ---------------------------------------------------------------------------


class _Cyclic:
    alpha: Quantity

    def __init__(self) -> None:
        self.alpha = 1.0 * ureg.m
        self.self_ref = self

    def step(self, dt: Quantity) -> None:
        self.alpha = self.alpha + dt


@unit_jit
def _cyclic_run(obj: _Cyclic, dt: Quantity, n: int) -> None:
    for _ in range(n):
        obj.step(dt)


def test_cyclic_mutation_propagates() -> None:
    obj = _Cyclic()
    _cyclic_run(obj, 0.1 * ureg.m, 5)
    assert abs(obj.alpha.to("m").magnitude - 1.5) < 1e-12
    assert obj.self_ref is obj


# ---------------------------------------------------------------------------
# Pint units are preserved after restore (not just magnitude correctness)
# ---------------------------------------------------------------------------


def test_units_preserved_after_restore() -> None:
    state = _EulerState(2)
    v = np.array([3.0, 4.0]) * ureg("m/s")
    dt = 1.0 * ureg.s
    _euler_run(state, v, dt, 1)
    assert state.x.dimensionality == ureg.m.dimensionality
    assert state.t.dimensionality == ureg.s.dimensionality
