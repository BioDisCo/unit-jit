"""Basic example: @unit_jit on a function that takes a model object.

Model: simple exponential decay
    dx/dt = -delta * x

The decorated simulate() calls model.rate() inside the loop. Because rate()
is also decorated with @unit_jit, the inner calls run as pure floats once
the outermost boundary is established, incurring no Pint overhead per step.
"""

import time

import numpy as np
from pint import Quantity, UnitRegistry

from unit_jit import unit_jit

ureg: UnitRegistry = UnitRegistry()


@unit_jit
class DecayModel:
    alpha: Quantity  # production rate  [mol/L/s]
    delta: Quantity  # degradation rate [1/s]

    def __init__(self, alpha: Quantity, delta: Quantity) -> None:
        self.alpha = alpha
        self.delta = delta

    def rate(self, x: Quantity) -> Quantity:
        """Net rate of change: alpha - delta * x  [mol/L/s]."""
        return self.alpha - self.delta * x  # type: ignore[return-value]


@unit_jit
def simulate(model: DecayModel, x0: Quantity, dt: Quantity, n_steps: int) -> np.ndarray:
    """Euler integration: inner model.rate() calls run as pure floats."""
    out = np.empty(n_steps)
    x = x0
    for i in range(n_steps):
        x = x + model.rate(x) * dt
        out[i] = x.to_base_units().magnitude
    return out


def simulate_pint(model: DecayModel, x0: Quantity, dt: Quantity, n_steps: int) -> np.ndarray:
    """Plain Pint: model.rate() called with full Pint overhead every step."""
    out = np.empty(n_steps)
    x = x0
    for i in range(n_steps):
        x = x + model.rate(x) * dt  # type: ignore[assignment]
        out[i] = x.to_base_units().magnitude
    return out


if __name__ == "__main__":
    model = DecayModel(
        alpha=0.1 * ureg.mol / ureg.L / ureg.s,
        delta=0.01 / ureg.s,
    )
    x0 = 5.0 * ureg.mol / ureg.L
    dt = 0.1 * ureg.s
    N = 600
    repeats = 300

    # First call: unit inference + compilation; already fast from here on.
    simulate(model, x0, dt, N)

    t0 = time.perf_counter()
    for _ in range(repeats):
        simulate_pint(model, x0, dt, N)
    t_pint = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(repeats):
        simulate(model, x0, dt, N)
    t_fast = time.perf_counter() - t0

    print(f"plain Pint: {t_pint / repeats * 1e3:.2f} ms per call")
    print(f"unit_jit:   {t_fast / repeats * 1e3:.2f} ms per call  ({t_pint / t_fast:.0f}x vs Pint)")
