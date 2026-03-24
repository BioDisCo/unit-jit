"""Benchmark: unit_jit vs plain Pint for a Gillespie birth-death loop.

This example keeps the Gillespie loop written with Pint semantics. `unit_jit`
removes the unit overhead after the first call while preserving the same code
structure.

The public API returns the full history as:

- `times`: a Quantity-wrapped ndarray in SI time units
- `counts`: a plain ndarray of molecule counts

Run with:

    python benchmarks/gillespie.py
"""

import math
import time

import numpy as np
from pint import Quantity

from unit_jit import unit_jit, ureg

ALPHA = 1.0 / ureg.minute / ureg.femtoliter
DELTA = 0.1 / ureg.minute
VOLUME = 1.0 * ureg.femtoliter
MAX_TIME = 60.0 * ureg.minute
INIT_COUNT = 0
MAX_EVENTS = 5_000
REPEATS = 300


class PlainBirthDeath:
    def __init__(self) -> None:
        self.alpha = ALPHA
        self.delta = DELTA
        self.volume = VOLUME
        self.init_count = INIT_COUNT

    def birth_rate(self) -> Quantity:
        """Zero-order production propensity."""
        return self.alpha * self.volume

    def death_rate(self, count: int) -> Quantity:
        """First-order degradation propensity."""
        return self.delta * count

    def simulate_history(
        self, rng: np.random.Generator, max_time: Quantity, counts_out: np.ndarray
    ) -> Quantity:
        """Run Gillespie and return the event times as a Quantity array."""
        time_now = 0.0 * max_time
        count = self.init_count
        time_history_mag = np.empty(len(counts_out))
        time_history_mag[0] = 0.0
        counts_out[0] = count
        n_events = 1

        while time_now < max_time and n_events < len(counts_out):
            birth = self.birth_rate()
            death = self.death_rate(count)
            total = birth + death
            if total.magnitude == 0.0:
                break

            r1, r2 = rng.random(2)
            time_now += math.log(1.0 / r1) / total

            if r2 * total.magnitude < birth.magnitude:
                count += 1
            elif count > 0:
                count -= 1

            time_history_mag[n_events] = time_now.to_base_units().magnitude
            counts_out[n_events] = count
            n_events += 1

        return time_history_mag[:n_events] * ureg.s

    def simulate(
        self, rng: np.random.Generator, max_time: Quantity, max_events: int = MAX_EVENTS
    ) -> tuple[Quantity, np.ndarray]:
        counts = np.empty(max_events + 1, dtype=int)
        times = self.simulate_history(rng, max_time, counts)
        return times, counts[: len(times)]


class FastBirthDeath:
    def __init__(self) -> None:
        self.alpha = ALPHA
        self.delta = DELTA
        self.volume = VOLUME
        self.init_count = INIT_COUNT

    @unit_jit
    def birth_rate(self) -> Quantity:
        """Zero-order production propensity."""
        return self.alpha * self.volume

    @unit_jit
    def death_rate(self, count: int) -> Quantity:
        """First-order degradation propensity."""
        return self.delta * count

    @unit_jit
    def simulate_history(
        self, rng: np.random.Generator, max_time: Quantity, counts_out: np.ndarray
    ) -> Quantity:
        """Run Gillespie and return the event times as a Quantity array."""
        time_now = 0.0 * max_time
        count = self.init_count
        time_history_mag = np.empty(len(counts_out))
        time_history_mag[0] = 0.0
        counts_out[0] = count
        n_events = 1

        while time_now < max_time and n_events < len(counts_out):
            birth = self.birth_rate()
            death = self.death_rate(count)
            total = birth + death
            if total.magnitude == 0.0:
                break

            r1, r2 = rng.random(2)
            time_now += math.log(1.0 / r1) / total

            if r2 * total.magnitude < birth.magnitude:
                count += 1
            elif count > 0:
                count -= 1

            time_history_mag[n_events] = time_now.to_base_units().magnitude
            counts_out[n_events] = count
            n_events += 1

        return time_history_mag[:n_events] * ureg.s

    def simulate(
        self, rng: np.random.Generator, max_time: Quantity, max_events: int = MAX_EVENTS
    ) -> tuple[Quantity, np.ndarray]:
        counts = np.empty(max_events + 1, dtype=int)
        times = self.simulate_history(rng, max_time, counts)
        return times, counts[: len(times)]


def bench(fn, repeats: int) -> float:
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return time.perf_counter() - t0


if __name__ == "__main__":
    plain = PlainBirthDeath()
    fast = FastBirthDeath()

    warmup_times, warmup_counts = fast.simulate(np.random.default_rng(0), MAX_TIME)
    plain_times, plain_counts = plain.simulate(np.random.default_rng(0), MAX_TIME)
    np.testing.assert_allclose(
        plain_times.to_base_units().magnitude,
        warmup_times.to_base_units().magnitude,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_array_equal(plain_counts, warmup_counts)

    t_plain = bench(lambda: plain.simulate(np.random.default_rng(42), MAX_TIME), REPEATS)
    t_fast = bench(lambda: fast.simulate(np.random.default_rng(42), MAX_TIME), REPEATS)

    print(f"plain Pint: {t_plain / REPEATS * 1e3:.2f} ms per call")
    print(
        f"unit_jit:   {t_fast / REPEATS * 1e3:.2f} ms per call  ({t_plain / t_fast:.0f}x vs Pint)"
    )
