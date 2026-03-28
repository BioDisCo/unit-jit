"""Tests for functools.partial unwrapping in callee unit inference.

Root cause: unit_jit internally wraps callee lookups with
functools.partial(inner, obj) to bind `self`.  inspect.getsource() raises
TypeError on a partial object, so the callee's return unit could not be
inferred, causing _UNKNOWN to propagate.
Fix: infer_return_units unwraps all functools.partial layers at its entry
point, prepending their bound args to the explicit call args before parsing.
"""

from __future__ import annotations

import functools
from typing import cast

from pint import Quantity, UnitRegistry

from unit_jit import unit_jit

ureg = UnitRegistry()


@unit_jit
def _rate_fn(base_rate: Quantity, n: Quantity, factor: int) -> Quantity:
    return cast("Quantity", base_rate * n * factor)


class _PartialCaller:
    def __init__(self, base_rate: Quantity) -> None:
        self.base_rate = base_rate

    @unit_jit
    def compute(self, n: Quantity, factor: int) -> Quantity:
        """Creates a partial binding self.base_rate, then calls it."""
        fn = functools.partial(_rate_fn, self.base_rate)
        return cast("Quantity", fn(n, factor))


class _PartialChained:
    def __init__(self, alpha: Quantity, volume: Quantity) -> None:
        self.alpha = alpha
        self.volume = volume

    @unit_jit
    def total(self, n: Quantity) -> Quantity:
        """Two-level partial — alpha bound first, then volume."""
        step1 = functools.partial(_rate_fn, self.alpha)
        step2 = functools.partial(step1, n)
        return cast("Quantity", step2(1))


# --- results ---


def test_partial_result():
    obj = _PartialCaller(2.0 / ureg.s / (ureg.mol / ureg.L))
    n = 3.0 * ureg.mol / ureg.L
    result = obj.compute(n, 2)
    expected = 2.0 / ureg.s / (ureg.mol / ureg.L) * n * 2
    assert abs(result.to_base_units().magnitude - expected.to_base_units().magnitude) < 1e-10


def test_partial_chained_result():
    obj = _PartialChained(2.0 / ureg.s / (ureg.mol / ureg.L), 1.0 * ureg.L)
    n = 3.0 * ureg.mol / ureg.L
    result = obj.total(n)
    expected = 2.0 / ureg.s / (ureg.mol / ureg.L) * n * 1
    assert abs(result.to_base_units().magnitude - expected.to_base_units().magnitude) < 1e-10


# --- units ---


def test_partial_returns_quantity():
    obj = _PartialCaller(1.0 / ureg.s / (ureg.mol / ureg.L))
    assert isinstance(obj.compute(1.0 * ureg.mol / ureg.L, 1), Quantity)


def test_partial_dimension():
    obj = _PartialCaller(1.0 / ureg.s / (ureg.mol / ureg.L))
    result = obj.compute(1.0 * ureg.mol / ureg.L, 1)
    # (L/(mol·s)) * (mol/L) * 1 = 1/s
    assert result.dimensionality == {"[time]": -1}
