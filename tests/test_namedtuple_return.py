"""Tests for NamedTuple return values from @unit_jit functions.

Root cause: _wrap only handled plain Quantity, list[Quantity], and plain
tuples.  A function returning a NamedTuple fell through to return the raw
fast-zone value (floats/arrays), never reconstructing Quantities.
Fix: _ListReturn carries a cls field.  _wrap detects NamedTuple cls and calls
cls._make() to rebuild the NamedTuple with all fields wrapped back to Quantity.
"""

from __future__ import annotations

from typing import NamedTuple, cast

import pytest
from pint import Quantity, UnitRegistry

from unit_jit import unit_jit

ureg = UnitRegistry()


class _StepResult(NamedTuple):
    time: Quantity
    state: Quantity


class _TrajectoryResult(NamedTuple):
    times: list[Quantity]
    states: list[Quantity]


class _SingleStep:
    def __init__(self, dt: Quantity, decay: Quantity) -> None:
        self.dt = dt
        self.decay = decay

    @unit_jit
    def step(self, t: Quantity, x: Quantity) -> _StepResult:
        t_new = t + self.dt
        x_new = x * (1.0 - self.decay * self.dt)
        return _StepResult(time=t_new, state=x_new)


class _Trajectory:
    def __init__(self, dt: Quantity) -> None:
        self.dt = dt

    @unit_jit
    def run(self, x0: Quantity, n_steps: int) -> _TrajectoryResult:
        times: list[Quantity] = []
        states: list[Quantity] = []
        t = 0.0 * ureg.s
        x = x0
        for _ in range(n_steps):
            t += self.dt
            x = x * 1.0
            times.append(t)
            states.append(x)
        return _TrajectoryResult(times=times, states=states)


# --- type preservation ---


def test_step_returns_namedtuple():
    sys = _SingleStep(dt=0.1 * ureg.s, decay=0.5 / ureg.s)
    result = sys.step(0.0 * ureg.s, 1.0 * ureg.mol / ureg.L)
    assert isinstance(result, _StepResult)


def test_trajectory_returns_namedtuple():
    sys = _Trajectory(dt=0.1 * ureg.s)
    result = sys.run(1.0 * ureg.mol / ureg.L, 3)
    assert isinstance(result, _TrajectoryResult)


# --- fields are Quantities ---


def test_step_time_is_quantity():
    sys = _SingleStep(dt=0.1 * ureg.s, decay=0.5 / ureg.s)
    result = sys.step(0.0 * ureg.s, 1.0 * ureg.mol / ureg.L)
    assert isinstance(result.time, Quantity)


def test_step_state_is_quantity():
    sys = _SingleStep(dt=0.1 * ureg.s, decay=0.5 / ureg.s)
    result = sys.step(0.0 * ureg.s, 1.0 * ureg.mol / ureg.L)
    assert isinstance(result.state, Quantity)


def test_trajectory_times_are_quantity():
    sys = _Trajectory(dt=0.1 * ureg.s)
    result = sys.run(1.0 * ureg.mol / ureg.L, 3)
    assert all(isinstance(t, Quantity) for t in result.times)


def test_trajectory_states_are_quantity():
    sys = _Trajectory(dt=0.1 * ureg.s)
    result = sys.run(1.0 * ureg.mol / ureg.L, 2)
    assert all(isinstance(s, Quantity) for s in result.states)


# --- values are correct ---


def test_step_time_value():
    sys = _SingleStep(dt=0.5 * ureg.s, decay=0.0 / ureg.s)
    result = sys.step(1.0 * ureg.s, 1.0 * ureg.mol / ureg.L)
    assert abs(result.time.to("s").magnitude - 1.5) < 1e-9


def test_trajectory_times_values():
    sys = _Trajectory(dt=0.5 * ureg.s)
    result = sys.run(1.0 * ureg.mol / ureg.L, 4)
    times_s = [t.to("s").magnitude for t in result.times]
    assert times_s == pytest.approx([0.5, 1.0, 1.5, 2.0], abs=1e-9)


# --- dimensions ---


def test_step_time_dimension():
    sys = _SingleStep(dt=0.1 * ureg.s, decay=0.5 / ureg.s)
    result = sys.step(0.0 * ureg.s, 1.0 * ureg.mol / ureg.L)
    assert result.time.dimensionality == {"[time]": 1}


def test_step_state_dimension():
    sys = _SingleStep(dt=0.1 * ureg.s, decay=0.5 / ureg.s)
    result = sys.step(0.0 * ureg.s, 1.0 * ureg.mol / ureg.L)
    assert result.state.dimensionality == {"[substance]": 1, "[length]": -3}
