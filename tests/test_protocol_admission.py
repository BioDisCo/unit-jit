"""Cross-cutting boundary contracts: protocols, cache invalidation and types."""

import inspect

import numpy as np
import pytest
from _jit_state import expect_execution
from pint import UnitRegistry

from unit_jit import unit_jit
from unit_jit._dispatch import PROTOCOL_NAMES


def identity(x):
    return x


def add(x):
    return x + x


def total(x):
    return np.sum(x)


@pytest.mark.parametrize("protocol", sorted(PROTOCOL_NAMES))
@pytest.mark.parametrize("warm", [False, True])
def test_protocol_overrides_are_guarded_for_every_plan(monkeypatch, protocol, warm):
    registry = UnitRegistry()
    value = registry.Quantity(np.array([2.0, 3.0]), "cm")
    wrapped = unit_jit(identity)
    if warm:
        with expect_execution(wrapped):
            wrapped(value)
    cls = type(value)
    original = inspect.getattr_static(cls, protocol, None)

    def override(self, *args, **kwargs):
        if original is None:
            return NotImplemented
        return original.__get__(self, type(self))(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(cls, protocol, override, raising=False)
        with expect_execution(wrapped, "fallback"):
            result = wrapped(value)
        assert result is value
    if warm:
        with expect_execution(wrapped):
            wrapped(value)


class QuantityReturningNumber(float):
    def __add__(self, other):
        return UnitRegistry().Quantity(7, "s")


class QuantityReturningArray(np.ndarray):
    def __array_function__(self, function, types, args, kwargs):
        return UnitRegistry().Quantity(7, "s")


class QuantityReturningList(list):
    def __add__(self, other):
        return UnitRegistry().Quantity(7, "s")


@pytest.mark.parametrize(
    ("function", "value"),
    [
        (add, QuantityReturningNumber(2)),
        (add, QuantityReturningList([2])),
        (total, np.array([2]).view(QuantityReturningArray)),
    ],
)
def test_custom_plain_protocols_cannot_be_assumed_unitless(function, value):
    wrapped = unit_jit(function)
    with expect_execution(wrapped, "fallback"):
        result = wrapped(value)
    assert result.magnitude == 7
    assert str(result.units) == "second"


@pytest.mark.parametrize("value", [2, 2.0, np.float64(2), np.array([2.0, 3.0])])
def test_builtin_numeric_representations_remain_fast(value):
    wrapped = unit_jit(add)
    with expect_execution(wrapped):
        result = wrapped(value)
    np.testing.assert_equal(result, value + value)
