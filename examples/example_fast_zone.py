"""fast_zone example: own the loop, call decorated functions at full speed.

Use fast_zone when you write the outer simulation loop yourself but call
@unit_jit-decorated functions inside it.  Declare which objects cross the
boundary; their Quantity attributes are converted to SI floats once on entry
so every inner call is free of Pint overhead.

Model: simple mRNA decay
    dx/dt = alpha - delta * x
"""

import time

import numpy as np
from pint import Quantity

from unit_jit import fast_zone, unit_jit, ureg


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


def simulate(model: DecayModel, x0: Quantity, dt: Quantity, n_steps: int) -> np.ndarray:
    """Plain Pint: rate() called with full Pint overhead every step."""
    x = x0
    out = np.empty(n_steps)
    for i in range(n_steps):
        x = x + model.rate(x) * dt  # type: ignore[assignment]
        out[i] = x.to_base_units().magnitude
    return out


def simulate_fast(model: DecayModel, x0: Quantity, dt: Quantity, n_steps: int) -> np.ndarray:
    """fast_zone: boundary crossed once; rate() runs as pure float every step."""
    x_si = x0.to_base_units().magnitude
    dt_si = dt.to_base_units().magnitude
    out = np.empty(n_steps)
    with fast_zone(model) as (fast_model,):
        for i in range(n_steps):
            x_si = x_si + fast_model.rate(x_si) * dt_si
            out[i] = x_si
    return out


if __name__ == "__main__":
    model = DecayModel(
        alpha=0.1 * ureg.mol / ureg.L / ureg.s,
        delta=0.01 / ureg.s,
    )
    x0 = 5.0 * ureg.mol / ureg.L
    dt = 0.1 * ureg.s
    N = 600
    repeats = 500

    # warm-up: first call compiles the rewritten float version
    simulate(model, x0, dt, N)
    simulate_fast(model, x0, dt, N)

    t0 = time.perf_counter()
    for _ in range(repeats):
        simulate(model, x0, dt, N)
    t_plain = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(repeats):
        simulate_fast(model, x0, dt, N)
    t_fast = time.perf_counter() - t0

    print(f"plain Pint:  {t_plain / repeats * 1e3:.2f} ms per call")
    print(
        f"fast_zone:   {t_fast / repeats * 1e3:.2f} ms per call  ({t_plain / t_fast:.0f}x vs plain)"
    )
