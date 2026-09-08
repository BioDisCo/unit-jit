"""Regression tests listed in todo.md."""

from __future__ import annotations

from typing import NamedTuple, cast

import numpy as np
import pytest
from _jit_state import expect_execution
from pint import Quantity, UnitRegistry

from unit_jit import unit_jit

ureg = UnitRegistry()


class _CycleB:
    parent: _CycleA


class _CycleA:
    child: _CycleB
    x: Quantity

    def __init__(self) -> None:
        self.child = _CycleB()
        self.child.parent = self
        self.x = 1.0 * ureg.m

    @unit_jit
    def scaled(self, factor: float) -> Quantity:
        self.x = self.x * factor
        return self.x


def test_cyclic_two_object_reference_strips_and_restores() -> None:
    a = _CycleA()

    with expect_execution(a.scaled):
        result = a.scaled(2.0)

    assert isinstance(result, Quantity)
    assert result.to("m").magnitude == pytest.approx(2.0)
    assert isinstance(a.x, Quantity)
    assert a.x.to("m").magnitude == pytest.approx(2.0)
    assert a.child.parent is a


class _SelfCycle:
    other: _SelfCycle
    length: Quantity
    time: Quantity

    def __init__(self) -> None:
        self.other = self
        self.length = 4.0 * ureg.m
        self.time = 2.0 * ureg.s

    @unit_jit
    def velocity(self) -> Quantity:
        return cast("Quantity", self.length / self.time)

    @unit_jit
    def invalid_sum(self) -> Quantity:
        return cast("Quantity", self.length + self.time)


def test_cyclic_self_reference_strips_and_restores() -> None:
    model = _SelfCycle()

    with expect_execution(model.velocity):
        result = model.velocity()

    assert isinstance(result, Quantity)
    assert result.to("m/s").magnitude == pytest.approx(2.0)
    assert isinstance(model.length, Quantity)
    assert isinstance(model.time, Quantity)
    assert model.other is model


def test_cyclic_reference_dimension_mismatch_raises() -> None:
    model = _SelfCycle()

    with pytest.raises(TypeError):
        model.invalid_sum()

    assert isinstance(model.length, Quantity)
    assert isinstance(model.time, Quantity)
    assert model.other is model


class _MixedParams(NamedTuple):
    rate: Quantity
    label: str
    count: int


class _NamedTupleModel:
    def __init__(self, params: _MixedParams) -> None:
        self.params = params

    @unit_jit
    def advance(self, x: Quantity) -> Quantity:
        return cast("Quantity", x + self.params.rate * self.params.count)


def test_namedtuple_identity_preserved_after_call() -> None:
    params = _MixedParams(rate=0.5 * ureg.m, label="fast", count=4)
    model = _NamedTupleModel(params)

    with expect_execution(model.advance):
        model.advance(1.0 * ureg.m)

    assert model.params is params


def test_mixed_namedtuple_fields_preserve_non_quantities() -> None:
    params = _MixedParams(rate=0.5 * ureg.m, label="fast", count=4)
    model = _NamedTupleModel(params)

    with expect_execution(model.advance):
        result = model.advance(1.0 * ureg.m)

    assert result.to("m").magnitude == pytest.approx(3.0)
    assert model.params.rate is params.rate
    assert model.params.label == "fast"
    assert model.params.count == 4


class _StatefulCounter:
    def __init__(self) -> None:
        self.length = 1.0 * ureg.m
        self.total = 0.0
        self.samples = np.zeros(2)

    @unit_jit
    def step(self, increment: float) -> Quantity:
        self.total += increment
        self.samples[0] += increment
        self.samples[1] = self.total
        self.length = self.length + increment * ureg.m
        return self.length


def test_stateful_non_quantity_attribute_mutation_persists() -> None:
    model = _StatefulCounter()

    with expect_execution(model.step):
        first = model.step(2.0)
    with expect_execution(model.step):
        second = model.step(3.0)

    assert first.to("m").magnitude == pytest.approx(3.0)
    assert second.to("m").magnitude == pytest.approx(6.0)
    assert model.total == pytest.approx(5.0)
    np.testing.assert_allclose(model.samples, np.array([5.0, 5.0]))
    assert isinstance(model.length, Quantity)
    assert model.length.to("m").magnitude == pytest.approx(6.0)


class _DirectParams(NamedTuple):
    distance: Quantity
    dt: Quantity
    label: str


@unit_jit
def _direct_namedtuple_arg(params: _DirectParams, scale: float) -> Quantity:
    return cast("Quantity", scale * params.distance / params.dt)


def test_namedtuple_as_direct_function_argument() -> None:
    params = _DirectParams(distance=3.0 * ureg.m, dt=2.0 * ureg.s, label="run")

    with expect_execution(_direct_namedtuple_arg):
        result = _direct_namedtuple_arg(params, 4.0)

    assert isinstance(result, Quantity)
    assert result.to("m/s").magnitude == pytest.approx(6.0)
    assert params.label == "run"


def test_unit_jit_decoration_is_idempotent() -> None:
    def f(x: Quantity) -> Quantity:
        return x

    wrapped = unit_jit(f)

    assert unit_jit(wrapped) is wrapped


class _BadMixedParams(NamedTuple):
    length: Quantity
    time: Quantity
    label: str


@unit_jit
def _bad_mixed_namedtuple_body(params: _BadMixedParams) -> Quantity:
    return cast("Quantity", params.length + params.time)


def test_mixed_namedtuple_body_dimension_error_raises() -> None:
    params = _BadMixedParams(length=1.0 * ureg.m, time=1.0 * ureg.s, label="bad")

    with pytest.raises(TypeError):
        _bad_mixed_namedtuple_body(params)
