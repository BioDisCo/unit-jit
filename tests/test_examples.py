"""Tests mirroring the README examples."""

from dataclasses import dataclass

import numpy as np
import pytest
from pint import Quantity

from unit_jit import unit_jit, ureg

# Simple function


@unit_jit
def velocity(d: Quantity, t: Quantity) -> Quantity:
    return d / t


def test_velocity_returns_quantity():
    velocity(10 * ureg.m, 2 * ureg.s)  # warm-up
    result = velocity(10 * ureg.m, 2 * ureg.s)
    assert isinstance(result, Quantity)


def test_velocity_same_dimension_different_unit():
    velocity(1 * ureg.m, 1 * ureg.s)  # warm-up
    r1 = velocity(10 * ureg.m, 2 * ureg.s)
    r2 = velocity(10 * ureg.cm, 2 * ureg.cs)
    assert abs(r1.to_base_units().magnitude - r2.to_base_units().magnitude) < 1e-12


def test_velocity_wrong_dimension_raises():
    velocity(10 * ureg.m, 2 * ureg.s)  # warm-up
    with pytest.raises(TypeError):
        velocity(10 * ureg.m, 2 * ureg.m)


# Loop returning Quantity-wrapped ndarray


@unit_jit
def simulate(n: int) -> Quantity:
    mrna = 10.0 * ureg.mol / ureg.L
    dt = 0.1 * ureg.s
    delta = 0.01 / ureg.s
    out = np.empty(n)
    for i in range(n):
        mrna = mrna - delta * mrna * dt
        out[i] = mrna.to_base_units().magnitude
    return out * ureg.mol / ureg.L


def test_simulate_returns_quantity_ndarray():
    simulate(5)  # warm-up
    result = simulate(10)
    assert isinstance(result, Quantity)
    assert isinstance(result.magnitude, np.ndarray)
    assert result.magnitude.shape == (10,)
    assert np.all(np.isfinite(result.magnitude))


# Class with Quantity attributes


@dataclass
class Params:
    alpha: Quantity  # [mol/L/s]
    delta: Quantity  # [1/s]


class Model:
    def __init__(self, params: Params) -> None:
        self.params = params

    @unit_jit
    def rate(self, mrna: Quantity) -> Quantity:
        return self.params.alpha - self.params.delta * mrna

    @unit_jit
    def simulate_model(self, n: int) -> np.ndarray:
        mrna = self.params.alpha / self.params.delta
        out = np.empty(n)
        for i in range(n):
            mrna = mrna + self.rate(mrna) * (0.1 * ureg.s)
            out[i] = mrna.to_base_units().magnitude
        return out


def test_model_simulate_shape_and_finite():
    model = Model(Params(alpha=0.1 * ureg.mol / ureg.L / ureg.s, delta=0.01 / ureg.s))
    model.simulate_model(5)  # warm-up
    out = model.simulate_model(20)
    assert out.shape == (20,)
    assert np.all(np.isfinite(out))


def test_model_rate_returns_quantity():
    model = Model(Params(alpha=0.1 * ureg.mol / ureg.L / ureg.s, delta=0.01 / ureg.s))
    result = model.rate(5 * ureg.mol / ureg.L)
    assert isinstance(result, Quantity)
