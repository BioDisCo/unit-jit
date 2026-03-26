"""Benchmark: @unit_jit on gillespie (bcrnnoise) vs plain Pint.

BCRN.gillespie is now decorated with @unit_jit, which:
  - Runs the whole loop in the fast zone (SI floats, no Pint overhead)
  - reaction_rates() is called from within the fast zone: args are already SI
    floats, so it also runs without Pint overhead (no @unit_jit needed on it)

PlainSystem overrides gillespie to call the original undecorated version
(__wrapped__) so we have a fair plain-Pint reference.
"""

import time
from collections.abc import Sequence

import numpy as np
from bcrnnoise import BCRN
from pint import Quantity, UnitRegistry

ureg = UnitRegistry()

INIT_MRNA = 0.0 / ureg.femtoliter
VOLUME = 1.0 * ureg.femtoliter
TIME_HORIZON = 60.0 * ureg.minute
DT = 0.1 * ureg.minute
ALPHA = 1.0 / ureg.minute / ureg.femtoliter
DELTA = 0.1 / ureg.minute

N_WARMUP = 3
N_REPS = 20

_plain_gillespie = BCRN.gillespie.__wrapped__  # type: ignore[attr-defined]


class PlainSystem(BCRN):
    """Vanilla subclass using the original (non-JIT) gillespie."""

    def __init__(self) -> None:
        super().__init__([INIT_MRNA], TIME_HORIZON, VOLUME, DT)
        self.alpha = ALPHA
        self.delta = DELTA

    @property
    def stoichiometry(self) -> np.ndarray:
        return np.array([[1], [-1]])

    def reaction_rates(self, state: Sequence[Quantity]) -> list[Quantity]:
        return [self.alpha, self.delta * state[0]]

    def gillespie(self, rng, initial_count_state, max_time):  # type: ignore[override]
        return _plain_gillespie(self, rng, initial_count_state, max_time)


class JitSystem(BCRN):
    """Same model using bcrnnoise's @unit_jit-decorated gillespie."""

    def __init__(self) -> None:
        super().__init__([INIT_MRNA], TIME_HORIZON, VOLUME, DT)
        self.alpha = ALPHA
        self.delta = DELTA

    @property
    def stoichiometry(self) -> np.ndarray:
        return np.array([[1], [-1]])

    def reaction_rates(self, state: Sequence[Quantity]) -> list[Quantity]:
        return [self.alpha, self.delta * state[0]]


def bench(name: str, fn, n_warmup: int, n_reps: int) -> float:
    for _ in range(n_warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_reps):
        fn()
    ms = (time.perf_counter() - t0) / n_reps * 1000
    print(f"  {name:35s}  {ms:.2f} ms/call")
    return ms


def main() -> None:
    plain = PlainSystem()
    jit = JitSystem()

    print(f"\nGillespie ({N_REPS} reps, {N_WARMUP} warmup, TIME_HORIZON={TIME_HORIZON}):")
    t_plain = bench("plain pint", lambda: plain.simulate_markov_chain(seed=42), N_WARMUP, N_REPS)
    t_jit = bench(
        "@unit_jit gillespie", lambda: jit.simulate_markov_chain(seed=42), N_WARMUP, N_REPS
    )
    print(f"  speedup: {t_plain / t_jit:.2f}x")


if __name__ == "__main__":
    main()
