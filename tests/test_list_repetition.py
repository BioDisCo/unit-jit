"""Tests for Python list repetition  [x] * n  inside @unit_jit functions.

Root cause: _binop returned _UNKNOWN for any _ListReturn operand, including
[x] * n which is pure Python list repetition (not arithmetic).  The entire
call that built an initial state via [init_val] * n_species then lost its
element type.
Fix: in _binop, when one operand is _ListReturn and the other is None (a
plain integer), the result is the same _ListReturn unchanged.
"""

from __future__ import annotations

from typing import cast

from pint import Quantity, UnitRegistry

from unit_jit import unit_jit

ureg = UnitRegistry()


@unit_jit
def _repeat_and_index(x: Quantity, n: int) -> Quantity:
    states = [x] * n
    return cast("Quantity", states[0])


@unit_jit
def _repeat_and_sum(x: Quantity, n: int) -> Quantity:
    states = [x] * n
    return cast("Quantity", sum(states))


@unit_jit
def _repeat_then_modify(x: Quantity, dx: Quantity, n: int) -> Quantity:
    """Repeat then mutate an element — typical SDE state init."""
    states = [x] * n
    states[0] = states[0] + dx
    return cast("Quantity", states[0])


# --- results ---


def test_repeat_index_result():
    v = 5.0 / ureg.m**3
    assert abs(_repeat_and_index(v, 3).to("1/m^3").magnitude - 5.0) < 1e-12


def test_repeat_sum_result():
    v = 1.0 / ureg.m**3
    assert abs(_repeat_and_sum(v, 4).to("1/m^3").magnitude - 4.0) < 1e-12


def test_repeat_then_modify_result():
    x = 2.0 / ureg.m**3
    dx = 1.0 / ureg.m**3
    assert abs(_repeat_then_modify(x, dx, 3).to("1/m^3").magnitude - 3.0) < 1e-12


# --- units ---


def test_repeat_index_returns_quantity():
    assert isinstance(_repeat_and_index(1.0 / ureg.m**3, 2), Quantity)


def test_repeat_index_dimension():
    assert _repeat_and_index(1.0 / ureg.m**3, 2).dimensionality == {"[length]": -3}


def test_repeat_sum_dimension():
    assert _repeat_and_sum(1.0 / ureg.m**3, 2).dimensionality == {"[length]": -3}
