# unit-jit

We love explicit tracking of physical units in code, but do not want to pay the runtime overhead in hot loops. `unit-jit` solves this with a single decorator: write your functions against [Pint](https://pint.readthedocs.io) as usual, and let `unit-jit` strip the unit machinery at JIT compile time so subsequent calls run on plain floats.

```python
from pint import Quantity
from unit_jit import unit_jit, ureg

@unit_jit
def velocity(d: Quantity, t: Quantity) -> Quantity:
    return d / t

velocity(10 * ureg.m, 2 * ureg.s)   # warm-up (runs Pint)
velocity(10 * ureg.m, 2 * ureg.s)   # fast (pure float internally)
velocity(10 * ureg.cm, 2 * ureg.s)  # fast and fine: same dimension, different unit
velocity(10 * ureg.m, 2 * ureg.m)   # TypeError: wrong dimension for arg 1
```

The first call runs the original Pint function to infer return units and cache argument dimensions. All subsequent calls convert arguments to SI floats at the boundary, run a rewritten pure-float version of the function, and wrap the result back into a `Quantity` with the cached units.

## Benchmark

Both functions below are identical in structure. `simulate_pint` runs with full Pint overhead on every call; `simulate_fast` pays that cost only on the first call and runs as plain floats thereafter.

```python
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


N, repeats = 500, 300
simulate_fast(N)  # warm-up

t0 = time.perf_counter()
for _ in range(repeats): simulate_fast(N)
t_fast = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(repeats): simulate_pint(N)
t_pint = time.perf_counter() - t0

print(f"unit_jit:   {t_fast / repeats * 1e3:.2f} ms per call")
print(f"plain Pint: {t_pint / repeats * 1e3:.2f} ms per call")
print(f"speedup:    {t_pint / t_fast:.0f}x")
```

Result on an Apple M3 Pro (500 steps, 300 repetitions):

```
unit_jit:   0.03 ms per call
plain Pint: 11.29 ms per call
speedup:    327x
```

The speedup scales with loop length: the longer the loop, the more Pint overhead is avoided per call.

## How it works

1. **Module-level compilation**: on first call, all `@unit_jit` functions in the same module are rewritten together: `.magnitude`, `.to_base_units()`, and `cast("Quantity", x)` are stripped from the source.
2. **Eager snapshot**: Quantity attributes on objects (e.g. `self.params.alpha`) are pre-converted to SI floats once at boundary entry. Attribute access inside the loop is a plain dict lookup.
3. **Fast zone**: a thread-local flag marks the outermost `@unit_jit` frame. Inner `@unit_jit` calls skip boundary conversion entirely.
4. **Return wrapping**: the SI unit of the return value is inferred from the first call and used to wrap subsequent results back into `Quantity`.
5. **Dimension guard**: argument dimensions are cached from the first call; any later call with a different dimension raises `TypeError` immediately.

The right entry point is the **outermost function that owns the hot loop**, not the leaf functions it calls.

## Installation

```bash
uv add unit-jit
```

```bash
pip install unit-jit
```

From source:

```bash
git clone https://github.com/BioDisCo/unit-jit && cd unit-jit
uv sync --extra dev  # or: pip install -e ".[dev]"
```

## Usage

### Scalar loop

The primary use case is a tight loop over scalars. `unit_jit` rewrites the function body so that all Pint calls disappear: `ureg.mol / ureg.L` becomes the corresponding SI float, `.to_base_units()` is stripped, and arithmetic runs on plain floats. The returned ndarray is wrapped back into a `Quantity` with the units inferred from the first call.

```python
import numpy as np
from pint import Quantity
from unit_jit import unit_jit, ureg

@unit_jit
def simulate(n: int) -> Quantity:
    mrna = 10.0 * ureg.mol / ureg.L
    dt   =  0.1 * ureg.s
    delta = 0.01 / ureg.s
    out = np.empty(n)
    for i in range(n):
        mrna = mrna - delta * mrna * dt
        out[i] = mrna.to_base_units().magnitude
    return out * ureg.mol / ureg.L
```

### NumPy array argument

When the argument is a `Quantity` wrapping a NumPy array, `unit_jit` converts it to the underlying SI ndarray at the boundary. The function body then runs on plain NumPy, and the result is wrapped back.

```python
import numpy as np
from pint import Quantity
from unit_jit import unit_jit, ureg

@unit_jit
def path_total(path: Quantity) -> Quantity:
    return np.sum(path)

path = np.array([1.0, 2.0, 3.0]) * ureg.m
path_total(path)   # warm-up
path_total(path)   # fast; returns 6.0 m as Quantity
```

### Vectorized operations on Quantity arrays

Multiple `Quantity` array arguments work the same way: each is converted to its SI ndarray independently, and the operation runs without any Pint overhead.

```python
import numpy as np
from pint import Quantity
from unit_jit import unit_jit, ureg

@unit_jit
def speeds(distances: Quantity, times: Quantity) -> Quantity:
    return distances / times

d = np.array([10.0, 20.0, 30.0]) * ureg.m
t = np.array([2.0,  4.0,  5.0]) * ureg.s
speeds(d, t)   # warm-up
speeds(d, t)   # fast; returns [5., 5., 6.] m/s as Quantity
```

### Class with Quantity attributes

`unit_jit` can be applied to individual methods or to the whole class at once. When applied to a class, it decorates all non-dunder methods automatically.

`unit_jit` snapshots all `Quantity` attributes on `self` once at the outermost boundary entry, replacing them with SI floats. Inner methods called from within the fast zone skip boundary conversion entirely, so there is no double-conversion overhead.

```python
from dataclasses import dataclass

import numpy as np
from pint import Quantity
from unit_jit import unit_jit, ureg

@dataclass
class Params:
    alpha: Quantity   # [mol/L/s]
    delta: Quantity   # [1/s]

@unit_jit
class Model:
    def __init__(self, params: Params) -> None:
        self.params = params

    def rate(self, mrna: Quantity) -> Quantity:
        return self.params.alpha - self.params.delta * mrna

    def simulate(self, n: int) -> np.ndarray:  # entry point: owns the hot loop
        mrna = self.params.alpha / self.params.delta
        out = np.empty(n)
        for i in range(n):
            mrna = mrna + self.rate(mrna) * (0.1 * ureg.s)
            out[i] = mrna.to_base_units().magnitude
        return out
```

`simulate` is the entry point: it owns the hot loop and is where boundary conversion happens. `rate` is called from within the fast zone, so it receives plain floats directly and its rewritten body runs without any Pint calls.

## Debugging

To inspect what code actually runs in the fast zone, use `get_rewritten_source`. It triggers compilation if needed and returns the rewritten function source as a string.

```python
from pint import Quantity
from unit_jit import unit_jit, get_rewritten_source, ureg

@unit_jit
def simulate(n: int) -> Quantity:
    mrna = 10.0 * ureg.mol / ureg.L
    dt   =  0.1 * ureg.s
    delta = 0.01 / ureg.s
    out = np.empty(n)
    for i in range(n):
        mrna = mrna - delta * mrna * dt
        out[i] = mrna.to_base_units().magnitude
    return out * ureg.mol / ureg.L

simulate(1)  # trigger compilation
print(get_rewritten_source(simulate))
```

Output:

```python
def simulate(n: int) -> Quantity:
    mrna = 10.0 * 1.0 / 0.0010000000000000002
    dt   =  0.1 * 1.0
    delta = 0.01 / 1.0
    out = np.empty(n)
    for i in range(n):
        mrna = mrna - delta * mrna * dt
        out[i] = mrna
    return out * 1.0 / 0.0010000000000000002
```

All `ureg` unit references are replaced by their SI float values (`ureg.mol / ureg.L` becomes `1.0 / 0.001` since 1 mol/L = 1000 mol/m³), `.to_base_units().magnitude` is stripped, and the arithmetic is otherwise unchanged.

## Running tests

```bash
pytest
```

## Feedback

If you find this library useful, feel free to drop a message. Hearing about your experience would be very welcome. If you have any suggestions or run into an issue, don't hesitate to [open an issue](https://github.com/BioDisCo/unit-jit/issues).

## License

Apache-2.0
