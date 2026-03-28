"""Tests for AugAssign operators (+=, -=, *=, /=) inside @unit_jit functions.

Root cause: _binop only handled cst.Add/Sub/Mul/Div.  The AugAssign node uses
cst.AddAssign/SubAssign/… instead, so `time += tau` wiped out the unit of
`time` (it mapped to None, losing the inferred type).
Fix: _aug_to_binop dict maps every aug-assign operator to its base operator
before delegating to _binop.
"""

from __future__ import annotations

import pytest
from pint import Quantity, UnitRegistry

from unit_jit import unit_jit

ureg = UnitRegistry()


@unit_jit
def _add(t: Quantity, dt: Quantity) -> Quantity:
    t += dt
    return t


@unit_jit
def _sub(x: Quantity, dx: Quantity) -> Quantity:
    x -= dx
    return x


@unit_jit
def _mul(x: Quantity, factor: Quantity) -> Quantity:
    x *= factor
    return x


@unit_jit
def _div(x: Quantity, factor: Quantity) -> Quantity:
    x /= factor
    return x


@unit_jit
def _accumulate(dt: Quantity, n: int) -> Quantity:
    """Typical Gillespie time accumulation loop — the original failing pattern."""
    t = 0.0 * ureg.s
    for _ in range(n):
        t += dt
    return t


# --- results ---


def test_add_result():
    assert abs(_add(1.0 * ureg.s, 0.5 * ureg.s).to("s").magnitude - 1.5) < 1e-12


def test_sub_result():
    assert abs(_sub(3.0 * ureg.m, 1.0 * ureg.m).to("m").magnitude - 2.0) < 1e-12


def test_mul_result():
    assert abs(_mul(2.0 * ureg.m, 3.0 * ureg.s).to("m*s").magnitude - 6.0) < 1e-12


def test_div_result():
    assert abs(_div(10.0 * ureg.m, 2.0 * ureg.s).to("m/s").magnitude - 5.0) < 1e-12


def test_accumulate_result():
    result = _accumulate(0.1 * ureg.s, 5)
    assert isinstance(result, Quantity)
    assert abs(result.to("s").magnitude - 0.5) < 1e-9


# --- units ---


def test_add_returns_quantity():
    assert isinstance(_add(1.0 * ureg.s, 0.5 * ureg.s), Quantity)


def test_add_dimension():
    assert _add(1.0 * ureg.s, 0.5 * ureg.s).dimensionality == {"[time]": 1}


def test_accumulate_dimension():
    assert _accumulate(0.1 * ureg.s, 3).dimensionality == {"[time]": 1}


# --- dimension mismatch still raises ---


def test_add_wrong_unit_raises():
    with pytest.raises(Exception):
        _add(1.0 * ureg.s, 1.0 * ureg.m)
