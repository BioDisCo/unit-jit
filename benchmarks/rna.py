"""Benchmark: unit_jit vs plain Pint for a scalar mRNA decay loop.

Both functions are identical in structure. simulate_pint runs with full Pint
overhead on every call; simulate_fast pays that cost only on the first call
and runs as plain floats thereafter. Run with:

    python benchmarks/rna.py
"""

import time

import numpy as np
from pint import Quantity

from unit_jit import unit_jit, ureg


@unit_jit
def simulate_fast(n: int) -> Quantity:
    mrna = 10.0 * ureg.mol / ureg.L
    dt   =  0.1 * ureg.s
    delta = 0.01 / ureg.s
    out = np.empty(n)
    for i in range(n):
        mrna = mrna - delta * mrna * dt
        out[i] = mrna.to_base_units().magnitude
    return out * ureg.mol / ureg.L


def simulate_pint(n: int) -> Quantity:
    mrna = 10.0 * ureg.mol / ureg.L
    dt   =  0.1 * ureg.s
    delta = 0.01 / ureg.s
    out = np.empty(n)
    for i in range(n):
        mrna = mrna - delta * mrna * dt
        out[i] = mrna.to_base_units().magnitude
    return out * ureg.mol / ureg.L


if __name__ == "__main__":
    N, repeats = 500, 300
    simulate_fast(N)  # warm-up

    t0 = time.perf_counter()
    for _ in range(repeats):
        simulate_fast(N)
    t_fast = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(repeats):
        simulate_pint(N)
    t_pint = time.perf_counter() - t0

    print(f"unit_jit:   {t_fast / repeats * 1e3:.2f} ms per call")
    print(f"plain Pint: {t_pint / repeats * 1e3:.2f} ms per call")
    print(f"speedup:    {t_pint / t_fast:.0f}x")
