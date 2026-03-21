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
def simulate(t: Quantity) -> Quantity:
    mrna = 10.0 * ureg.nmol / ureg.L  # 10 nM initial concentration
    dt = 1.0 * ureg.s  # 1 s timestep
    delta = np.log(2) / (5.0 * ureg.min)  # half-life 5 min (E. coli mRNA)
    n = int((t / dt).to_base_units().magnitude)
    out = np.empty(n)
    for i in range(n):
        mrna = mrna - delta * mrna * dt
        out[i] = mrna.to_base_units().magnitude
    return out * ureg.mol / ureg.m**3


def test_simulate_returns_quantity():
    simulate(5 * ureg.min)  # warm-up
    result = simulate(10 * ureg.min)
    assert isinstance(result, Quantity)
    assert result.magnitude.shape == (600,)
    assert np.all(np.isfinite(result.magnitude))


# Class with Quantity attributes, per-method decorator


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
    def simulate_model(self, t: Quantity) -> Quantity:
        dt = 10.0 * ureg.s
        mrna = self.params.alpha / self.params.delta
        n = int((t / dt).to_base_units().magnitude)
        out = np.empty(n)
        for i in range(n):
            mrna = mrna + self.rate(mrna) * dt
            out[i] = mrna.to_base_units().magnitude
        return out * ureg.mol / ureg.m**3


def test_model_simulate_shape_and_finite():
    model = Model(Params(alpha=0.1 * ureg.mol / ureg.L / ureg.s, delta=0.01 / ureg.s))
    model.simulate_model(5 * ureg.min)  # warm-up
    out = model.simulate_model(10 * ureg.min)
    assert out.magnitude.shape == (60,)
    assert np.all(np.isfinite(out.magnitude))


def test_model_rate_returns_quantity():
    model = Model(Params(alpha=0.1 * ureg.mol / ureg.L / ureg.s, delta=0.01 / ureg.s))
    result = model.rate(5 * ureg.mol / ureg.L)
    assert isinstance(result, Quantity)


# Class decorator


@dataclass
class Params2:
    alpha: Quantity  # [mol/L/s]
    delta: Quantity  # [1/s]


@unit_jit
class Model2:
    def __init__(self, params: Params2) -> None:
        self.params = params

    def rate(self, mrna: Quantity) -> Quantity:
        return self.params.alpha - self.params.delta * mrna

    def simulate(self, t: Quantity) -> Quantity:
        dt = 10.0 * ureg.s
        mrna = self.params.alpha / self.params.delta
        n = int((t / dt).to_base_units().magnitude)
        out = np.empty(n)
        for i in range(n):
            mrna = mrna + self.rate(mrna) * dt
            out[i] = mrna.to_base_units().magnitude
        return out * ureg.mol / ureg.m**3


def test_class_decorator_simulate_shape_and_finite():
    model = Model2(Params2(alpha=0.1 * ureg.mol / ureg.L / ureg.s, delta=0.01 / ureg.s))
    model.simulate(5 * ureg.min)  # warm-up
    out = model.simulate(10 * ureg.min)
    assert out.magnitude.shape == (60,)
    assert np.all(np.isfinite(out.magnitude))


def test_class_decorator_rate_returns_quantity():
    model = Model2(Params2(alpha=0.1 * ureg.mol / ureg.L / ureg.s, delta=0.01 / ureg.s))
    result = model.rate(5 * ureg.mol / ureg.L)
    assert isinstance(result, Quantity)


def test_class_decorator_matches_per_method_decorator():
    """Class decorator and per-method decorator give identical results."""
    params = Params(alpha=0.1 * ureg.mol / ureg.L / ureg.s, delta=0.01 / ureg.s)
    params2 = Params2(alpha=0.1 * ureg.mol / ureg.L / ureg.s, delta=0.01 / ureg.s)
    m1 = Model(params)
    m2 = Model2(params2)
    m1.simulate_model(5 * ureg.min)
    m2.simulate(5 * ureg.min)
    out1 = m1.simulate_model(10 * ureg.min)
    out2 = m2.simulate(10 * ureg.min)
    np.testing.assert_allclose(out1.magnitude, out2.magnitude)


# NumPy array with units


@unit_jit
def path_total(path: Quantity) -> Quantity:
    """Sum of a path given as a Quantity-wrapped ndarray."""
    return np.sum(path)


def test_path_total_returns_quantity():
    path = np.array([1.0, 2.0, 3.0]) * ureg.m
    result = path_total(path)
    assert isinstance(result, Quantity)


def test_path_total_value():
    path = np.array([1.0, 2.0, 3.0]) * ureg.m
    path_total(path)  # warm-up
    result = path_total(path)
    assert abs(result.to_base_units().magnitude - 6.0) < 1e-12


def test_path_total_non_si_units():
    """Path in cm; result should be in SI (metres)."""
    path = np.array([100.0, 200.0, 300.0]) * ureg.cm  # 1 m, 2 m, 3 m
    path_total(path)  # warm-up
    result = path_total(path)
    assert abs(result.to_base_units().magnitude - 6.0) < 1e-12


# Vectorized operations on Quantity arrays


@unit_jit
def speeds(distances: Quantity, times: Quantity) -> Quantity:
    """Element-wise speed from distance and time arrays."""
    return distances / times


def test_speeds_returns_quantity():
    d = np.array([10.0, 20.0, 30.0]) * ureg.m
    t = np.array([2.0, 4.0, 5.0]) * ureg.s
    result = speeds(d, t)
    assert isinstance(result, Quantity)


def test_speeds_values():
    d = np.array([10.0, 20.0, 30.0]) * ureg.m
    t = np.array([2.0, 4.0, 5.0]) * ureg.s
    speeds(d, t)  # warm-up
    result = speeds(d, t)
    np.testing.assert_allclose(result.to_base_units().magnitude, [5.0, 5.0, 6.0])


def test_speeds_non_si_units():
    """Input in km and hours; result should match in SI (m/s)."""
    d = np.array([10.0, 20.0, 30.0]) * ureg.km
    t = np.array([2.0, 4.0, 5.0]) * ureg.hour
    speeds(d, t)  # warm-up
    result = speeds(d, t)
    expected = np.array([10000.0 / 7200.0, 20000.0 / 14400.0, 30000.0 / 18000.0])
    np.testing.assert_allclose(result.to_base_units().magnitude, expected, rtol=1e-12)
