import time

from unit_fast import unit_fast, ureg


@unit_fast
def simulate(n: int) -> float:
    v = 0.0 * ureg.cm / ureg.s
    for _ in range(n):
        d = 10 * ureg.cm
        t = 2 * ureg.s
        v = d / t
    return v.magnitude


def simulate_pint(n: int) -> float:
    v = 0.0 * ureg.cm / ureg.s
    for _ in range(n):
        d = 10 * ureg.cm
        t = 2 * ureg.s
        v = d / t
    return v.magnitude


N = 100
repeats = 1_000

# Warm-up
simulate(N)

t0 = time.perf_counter()
for _ in range(repeats):
    simulate(N)
t_fast = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(repeats):
    simulate_pint(N)
t_pint = time.perf_counter() - t0

print(f"unit_fast:  {t_fast:.3f} s")
print(f"plain Pint: {t_pint:.3f} s")
print(f"speedup:    {t_pint / t_fast:.1f}x")
