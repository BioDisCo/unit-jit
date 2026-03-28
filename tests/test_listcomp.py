"""Tests for list comprehension  [f(x) for x in items]  inside @unit_jit functions.

Root cause: cst.ListComp nodes were not handled in _expr, so any list
comprehension fell through to return _UNKNOWN.  Downstream indexing or summing
over the result then produced _UNKNOWN, killing JIT for the whole function.
Fix: _expr dispatches on cst.ListComp, infers the element expression with the
loop variable bound to the iterable's element unit.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from pint import Quantity, UnitRegistry

from unit_jit import unit_jit

ureg = UnitRegistry()


@unit_jit
def _scale_each(items: Sequence[Quantity], factor: Quantity) -> Quantity:
    scaled = [x * factor for x in items]
    return cast("Quantity", scaled[0])


@unit_jit
def _sum_scaled(items: Sequence[Quantity], factor: Quantity) -> Quantity:
    return cast("Quantity", sum([x * factor for x in items]))


@unit_jit
def _double_all_sum(items: Sequence[Quantity]) -> Quantity:
    doubled = [x * 2 for x in items]
    return cast("Quantity", sum(doubled))


@unit_jit
def _divide_each(items: Sequence[Quantity], denom: Quantity) -> Quantity:
    result = [x / denom for x in items]
    return cast("Quantity", result[0])


# --- results ---


def test_scale_each_result():
    items = [2.0 * ureg.m, 3.0 * ureg.m]
    result = _scale_each(items, 2.0 * ureg.s)
    assert abs(result.to("m*s").magnitude - 4.0) < 1e-12


def test_sum_scaled_result():
    items = [1.0 / ureg.s, 2.0 / ureg.s, 3.0 / ureg.s]
    result = _sum_scaled(items, 2.0 * ureg.m)
    assert abs(result.to("m/s").magnitude - 12.0) < 1e-12


def test_double_all_sum_result():
    items = [1.0 / ureg.s, 2.0 / ureg.s, 3.0 / ureg.s]
    result = _double_all_sum(items)
    assert abs(result.to("1/s").magnitude - 12.0) < 1e-12


def test_divide_each_result():
    items = [6.0 * ureg.m]
    result = _divide_each(items, 2.0 * ureg.s)
    assert abs(result.to("m/s").magnitude - 3.0) < 1e-12


# --- units ---


def test_scale_each_returns_quantity():
    assert isinstance(_scale_each([1.0 * ureg.m], 1.0 * ureg.s), Quantity)


def test_scale_each_dimension():
    result = _scale_each([1.0 * ureg.m], 1.0 * ureg.s)
    assert result.dimensionality == {"[length]": 1, "[time]": 1}


def test_divide_each_dimension():
    result = _divide_each([1.0 * ureg.m], 1.0 * ureg.s)
    assert result.dimensionality == {"[length]": 1, "[time]": -1}
