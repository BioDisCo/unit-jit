"""Tests for sum/min/max reducing a list[Quantity] inside @unit_jit functions.

Root cause: the _KNOWN_CALLS handler for sum/min/max used _p (identity on
first arg), so sum(rates) where rates was _ListReturn returned _ListReturn
itself rather than the element unit.  The caller's return type then became
_ListReturn instead of a pint.Unit, causing _wrap to misfire.
Fix: _reduce() extracts the common element unit from a _ListReturn.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from pint import Quantity, UnitRegistry

from unit_jit import unit_jit

ureg = UnitRegistry()


@unit_jit
def _sum_rates(rates: Sequence[Quantity]) -> Quantity:
    return cast("Quantity", sum(rates))


@unit_jit
def _min_rates(rates: Sequence[Quantity]) -> Quantity:
    return cast("Quantity", min(rates))


@unit_jit
def _max_rates(rates: Sequence[Quantity]) -> Quantity:
    return cast("Quantity", max(rates))


@unit_jit
def _sum_of_products(xs: Sequence[Quantity], factor: Quantity) -> Quantity:
    """More realistic: sum over a derived list, as seen in Gillespie."""
    scaled = [x * factor for x in xs]
    return cast("Quantity", sum(scaled))


# --- results ---


def test_sum_result():
    rates = [1.0 / ureg.s, 2.0 / ureg.s, 3.0 / ureg.s]
    assert abs(_sum_rates(rates).to("1/s").magnitude - 6.0) < 1e-12


def test_min_result():
    rates = [3.0 / ureg.s, 1.0 / ureg.s, 2.0 / ureg.s]
    assert abs(_min_rates(rates).to("1/s").magnitude - 1.0) < 1e-12


def test_max_result():
    rates = [3.0 / ureg.s, 1.0 / ureg.s, 2.0 / ureg.s]
    assert abs(_max_rates(rates).to("1/s").magnitude - 3.0) < 1e-12


def test_sum_of_products_result():
    xs = [1.0 / ureg.s, 2.0 / ureg.s, 3.0 / ureg.s]
    factor = 2.0 * ureg.m
    result = _sum_of_products(xs, factor)
    assert abs(result.to("m/s").magnitude - 12.0) < 1e-12


# --- units ---


def test_sum_returns_quantity():
    assert isinstance(_sum_rates([1.0 / ureg.s]), Quantity)


def test_sum_dimension():
    assert _sum_rates([1.0 / ureg.s]).dimensionality == {"[time]": -1}


def test_sum_of_products_dimension():
    result = _sum_of_products([1.0 / ureg.s], 1.0 * ureg.m)
    assert result.dimensionality == {"[length]": 1, "[time]": -1}
