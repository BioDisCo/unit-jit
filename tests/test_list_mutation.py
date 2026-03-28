"""Tests for list.append() and list.extend() mutation tracking inside @unit_jit.

Root cause: the inferrer tracked list variable types from assignment but not
from in-place mutations via .append() or .extend().  After appending Quantities
to an empty list, the list's element type remained _UNKNOWN, so sum(rates)
returned _UNKNOWN.
Fix: _mutation_call intercepts .append(x) and .extend([x, ...]) calls and
updates the inferred element unit of the list variable accordingly.
"""

from __future__ import annotations

from typing import cast

from pint import Quantity, UnitRegistry

from unit_jit import unit_jit

ureg = UnitRegistry()


@unit_jit
def _append_sum(a: Quantity, b: Quantity) -> Quantity:
    rates: list[Quantity] = []
    rates.append(a)
    rates.append(b)
    return cast("Quantity", sum(rates))


@unit_jit
def _extend_index(a: Quantity, b: Quantity) -> Quantity:
    rates: list[Quantity] = []
    rates.extend([a, b])
    return cast("Quantity", rates[0])


@unit_jit
def _append_in_loop(dt: Quantity, n: int) -> Quantity:
    """Typical trajectory recording pattern."""
    times: list[Quantity] = []
    t = 0.0 * ureg.s
    for _ in range(n):
        t += dt
        times.append(t)
    return cast("Quantity", times[-1])


@unit_jit
def _extend_then_sum(a: Quantity, b: Quantity, c: Quantity) -> Quantity:
    rates: list[Quantity] = []
    rates.extend([a, b])
    rates.append(c)
    return cast("Quantity", sum(rates))


# --- results ---


def test_append_sum_result():
    result = _append_sum(1.0 / ureg.s, 2.0 / ureg.s)
    assert abs(result.to("1/s").magnitude - 3.0) < 1e-12


def test_extend_index_result():
    result = _extend_index(4.0 / ureg.s, 2.0 / ureg.s)
    assert abs(result.to("1/s").magnitude - 4.0) < 1e-12


def test_append_in_loop_result():
    result = _append_in_loop(0.1 * ureg.s, 5)
    assert abs(result.to("s").magnitude - 0.5) < 1e-9


def test_extend_then_sum_result():
    result = _extend_then_sum(1.0 / ureg.s, 2.0 / ureg.s, 3.0 / ureg.s)
    assert abs(result.to("1/s").magnitude - 6.0) < 1e-12


# --- units ---


def test_append_sum_returns_quantity():
    assert isinstance(_append_sum(1.0 / ureg.s, 0.5 / ureg.s), Quantity)


def test_append_sum_dimension():
    assert _append_sum(1.0 / ureg.s, 0.5 / ureg.s).dimensionality == {"[time]": -1}


def test_append_in_loop_dimension():
    assert _append_in_loop(0.1 * ureg.s, 2).dimensionality == {"[time]": 1}
