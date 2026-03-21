"""Baseline: identical mRNA model using plain Pint (no @unit_fast).

Same structure and parameters as example_advanced.py — used to measure
the overhead that @unit_fast removes.
"""

import time
from dataclasses import dataclass
from typing import cast

import numpy as np
from pint import Quantity, UnitRegistry

ureg = UnitRegistry()


@dataclass
class Params:
    alpha: Quantity  # transcription rate  [mol/L/s]
    delta: Quantity  # degradation rate    [1/s]
    sigma: Quantity  # noise amplitude     [mol/L/s^(1/2)]
    dt: Quantity  # timestep            [s]


def _ou_noise(
    rng: np.random.Generator,
    sigma: Quantity,
    dt: Quantity,
) -> Quantity:
    """Additive OU noise: sigma * sqrt(dt) * N(0,1)  [mol/L]."""
    return cast(Quantity, sigma * dt**0.5 * rng.standard_normal())


class Model:
    def __init__(self, params: Params) -> None:
        self.params = params

    def drift(self, mrna: Quantity) -> Quantity:
        """Deterministic drift: (alpha - delta * mRNA) * dt  [mol/L]."""
        return cast(Quantity, (self.params.alpha - self.params.delta * mrna) * self.params.dt)

    def noise(self, rng: np.random.Generator, mrna: Quantity) -> Quantity:  # noqa: ARG002
        """Stochastic term via helper."""
        return _ou_noise(rng, self.params.sigma, self.params.dt)

    def simulate(self, n_steps: int, seed: int = 0) -> np.ndarray:
        """Euler-Maruyama trajectory. Returns mRNA concentrations [mol/L]."""
        rng = np.random.default_rng(seed)
        mrna: Quantity = self.params.alpha / self.params.delta
        trajectory = np.empty(n_steps)
        for i in range(n_steps):
            mrna = mrna + self.drift(mrna) + self.noise(rng, mrna)
            trajectory[i] = mrna.to_base_units().magnitude
        return trajectory


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

    t0 = time.perf_counter()
    for _ in range(repeats):
        model.simulate(N)
    t_pint = time.perf_counter() - t0

    print(f"plain Pint: {t_pint:.3f} s  ({repeats} × {N} steps)")
    print(f"  → {t_pint / repeats * 1e3:.2f} ms / trajectory")
