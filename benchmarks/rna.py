"""Benchmark: unit_jit vs plain Pint for a scalar mRNA decay loop.

Both functions are identical in structure. simulate_pint runs with full Pint
overhead on every call; simulate_fast runs as plain floats on every call. Run with:

    python benchmarks/rna.py
"""

import time

import numpy as np
from pint import Quantity

from unit_jit import unit_jit, ureg


@unit_jit
def simulate_fast(t: Quantity) -> Quantity:
    mrna = 10.0 * ureg.nmol / ureg.L  # 10 nM initial concentration
    dt = 1.0 * ureg.s  # 1 s timestep
    delta = np.log(2) / (5.0 * ureg.min)  # half-life 5 min (E. coli mRNA)
    n = int(t / dt)
    out = np.empty(n)
    for i in range(n):
        mrna = mrna - delta * mrna * dt
        out[i] = mrna.to_base_units().magnitude
    return out * ureg.mol / ureg.m**3


def simulate_pint(t: Quantity) -> Quantity:
    mrna = 10.0 * ureg.nmol / ureg.L
    dt = 1.0 * ureg.s
    delta = np.log(2) / (5.0 * ureg.min)
    n = int(t / dt)
    out = np.empty(n)
    for i in range(n):
        mrna = mrna - delta * mrna * dt
        out[i] = mrna.to_base_units().magnitude
    return out * ureg.mol / ureg.m**3


if __name__ == "__main__":
    T, repeats = 10 * ureg.min, 300

    simulate_fast(T)  # first call: unit inference + compilation

    t0 = time.perf_counter()
    for _ in range(repeats):
        simulate_pint(T)
    t_pint = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(repeats):
        simulate_fast(T)
    t_fast = time.perf_counter() - t0

    print(f"plain Pint: {t_pint / repeats * 1e3:.2f} ms per call")
    print(f"unit_jit:   {t_fast / repeats * 1e3:.2f} ms per call  ({t_pint / t_fast:.0f}x vs Pint)")
