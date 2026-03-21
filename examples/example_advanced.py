"""Advanced example: mRNA model with @unit_fast.

Mirrors the structure of real biological models:
  - Params dataclass with Pint Quantity fields
  - Module-level helper called from a class method (inner @unit_fast call,
    no double boundary conversion)
  - Class method with self.params.* Quantity attribute access via proxy

The model: Ornstein-Uhlenbeck process for mRNA concentration.
  dmRNA = (alpha - delta * mRNA) dt + sigma * sqrt(dt) * N(0,1)

Units:
  alpha  [mol/L/s],  delta [1/s],  sigma [mol/L/s^(1/2)],  dt [s]
  drift term: mol/L/s * s = mol/L  ✓
  noise term: mol/L/s^(1/2) * s^(1/2) = mol/L  ✓
"""

import time
from dataclasses import dataclass
from typing import cast

import numpy as np
from pint import Quantity

from unit_fast import unit_fast, ureg


@dataclass
class Params:
    alpha: Quantity   # transcription rate  [mol/L/s]
    delta: Quantity   # degradation rate    [1/s]
    sigma: Quantity   # noise amplitude     [mol/L/s^(1/2)]
    dt: Quantity      # timestep            [s]


# ── Module-level helper ────────────────────────────────────────────────────────
# Compiled together with Model.drift and Model.noise on first call.
# When invoked from Model.noise (already inside the fast zone), _in_fast_zone()
# is True so boundary conversion is skipped — args are already SI floats.

@unit_fast
def _ou_noise(
    rng: np.random.Generator,
    sigma: Quantity,
    dt: Quantity,
) -> Quantity:
    """Additive OU noise: sigma * sqrt(dt) * N(0,1)  [mol/L]."""
    return cast("Quantity", sigma * dt**0.5 * rng.standard_normal())


# ── Model ──────────────────────────────────────────────────────────────────────


class Model:
    def __init__(self, params: Params) -> None:
        self.params = params

    @unit_fast
    def drift(self, mrna: Quantity) -> Quantity:
        """Deterministic drift: (alpha - delta * mRNA) * dt  [mol/L]."""
        return cast("Quantity", (self.params.alpha - self.params.delta * mrna) * self.params.dt)

    @unit_fast
    def noise(self, rng: np.random.Generator, mrna: Quantity) -> Quantity:
        """Stochastic term via helper. mrna unused but kept for interface symmetry."""
        return _ou_noise(rng, self.params.sigma, self.params.dt)  # type: ignore[return-value]

    @unit_fast
    def simulate(self, n_steps: int, seed: int = 0) -> np.ndarray:
        """Euler-Maruyama trajectory. Returns mRNA concentrations [mol/L]."""
        rng = np.random.default_rng(seed)
        mrna: Quantity = self.params.alpha / self.params.delta  # steady-state mean
        trajectory = np.empty(n_steps)
        for i in range(n_steps):
            mrna = mrna + self.drift(mrna) + self.noise(rng, mrna)  # type: ignore[operator]
            trajectory[i] = mrna.to_base_units().magnitude
        return trajectory


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    params = Params(
        alpha=0.1 * ureg.mol / ureg.L / ureg.s,
        delta=0.01 / ureg.s,
        sigma=0.005 * ureg.mol / ureg.L / ureg.s**0.5,
        dt=0.1 * ureg.s,
    )
    model = Model(params)
    N = 500
    repeats = 300

    # Warm-up: first call runs the original Pint version to infer return units.
    # Subsequent calls use the rewritten float version.
    model.simulate(N)
    print()

    t0 = time.perf_counter()
    for _ in range(repeats):
        model.simulate(N)
    t_fast = time.perf_counter() - t0

    print(f"unit_fast: {t_fast:.3f} s  ({repeats} × {N} steps)")
    print(f"  → {t_fast / repeats * 1e3:.2f} ms / trajectory")
