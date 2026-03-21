"""Benchmark: unit_jit vs unit_jit + Numba for the mRNA decay loop.

unit_jit strips Pint overhead and runs plain Python floats. This benchmark
checks whether additionally compiling the inner loop with Numba gives a further
speedup. Run with:

    python benchmarks/rna_numba.py
"""

import time

import numpy as np
from pint import Quantity

from unit_jit import unit_jit, ureg


@unit_jit
def simulate_unitjit(t: Quantity) -> Quantity:
    mrna  = 10.0 * ureg.nmol / ureg.L        # 10 nM initial concentration
    dt    =  1.0 * ureg.s                     # 1 s timestep
    delta = np.log(2) / (5.0 * ureg.min)     # half-life 5 min (E. coli mRNA)
    n = int((t / dt).to_base_units().magnitude)
    out = np.empty(n)
    for i in range(n):
        mrna = mrna - delta * mrna * dt
        out[i] = mrna.to_base_units().magnitude
    return out * ureg.mol / ureg.m**3


@unit_jit(use_numba=True)
def simulate_numba(t: Quantity) -> Quantity:
    mrna  = 10.0 * ureg.nmol / ureg.L        # 10 nM initial concentration
    dt    =  1.0 * ureg.s                     # 1 s timestep
    delta = np.log(2) / (5.0 * ureg.min)     # half-life 5 min (E. coli mRNA)
    n = int((t / dt).to_base_units().magnitude)
    out = np.empty(n)
    for i in range(n):
        mrna = mrna - delta * mrna * dt
        out[i] = mrna.to_base_units().magnitude
    return out * ureg.mol / ureg.m**3


def simulate_pint(t: Quantity) -> Quantity:
    mrna  = 10.0 * ureg.nmol / ureg.L
    dt    =  1.0 * ureg.s
    delta = np.log(2) / (5.0 * ureg.min)
    n = int((t / dt).to_base_units().magnitude)
    out = np.empty(n)
    for i in range(n):
        mrna = mrna - delta * mrna * dt
        out[i] = mrna.to_base_units().magnitude
    return out * ureg.mol / ureg.m**3


if __name__ == "__main__":
    T, repeats = 10 * ureg.min, 300

    simulate_unitjit(T)   # warm-up: unit_jit compilation (Pint first call)
    simulate_unitjit(T)   # warm-up: unit_jit fast path
    simulate_numba(T)     # warm-up: Pint first call
    simulate_numba(T)     # warm-up: triggers Numba compilation
    simulate_numba(T)     # warm-up: Numba fast path
    simulate_pint(T)      # warm-up

    t0 = time.perf_counter()
    for _ in range(repeats):
        simulate_pint(T)
    t_pint = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(repeats):
        simulate_unitjit(T)
    t_unitjit = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(repeats):
        simulate_numba(T)
    t_numba = time.perf_counter() - t0

    print(f"plain Pint:       {t_pint    / repeats * 1e3:.2f} ms per call")
    print(f"unit_jit:         {t_unitjit / repeats * 1e3:.2f} ms per call  ({t_pint / t_unitjit:.0f}x vs Pint)")
    print(f"unit_jit + Numba: {t_numba   / repeats * 1e3:.2f} ms per call  ({t_pint / t_numba:.0f}x vs Pint)")
