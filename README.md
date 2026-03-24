# unit-jit

We love explicit tracking of physical units in code, but do not want to pay the runtime overhead in hot loops. `unit-jit` solves this with a single decorator: write your functions against [Pint](https://pint.readthedocs.io) as usual, and let `unit-jit` strip the unit machinery at JIT compile time so every call runs on plain floats.

```python
from pint import Quantity
from unit_jit import unit_jit, ureg

@unit_jit
def velocity(d: Quantity, t: Quantity) -> Quantity:
    return d / t

velocity(10 * ureg.m, 2 * ureg.s)   # first call: unit inference + fast
velocity(10 * ureg.m, 2 * ureg.s)   # fast (pure float internally)
velocity(10 * ureg.cm, 2 * ureg.s)  # fast and fine: same dimension, different unit
velocity(10 * ureg.m, 2 * ureg.m)   # TypeError: wrong dimension for arg 1
```

On the first call, `unit-jit` abstract-interprets the function body with the input units, checks dimensional correctness across all branches, infers return units, and caches a CST-rewritten version that operates on raw floats. All subsequent calls convert arguments to SI floats at the boundary, run the rewritten pure-float version, and wrap the result back into a `Quantity` with the cached units.

## Benchmark

Both functions below are identical in structure. `simulate_pint` runs with full Pint overhead on every call; `simulate_fast` runs as plain floats on every call.

```python
import time

import numpy as np
from pint import Quantity
from unit_jit import unit_jit, ureg


@unit_jit
def simulate_fast(t: Quantity) -> Quantity:
    mrna  = 10.0 * ureg.nmol / ureg.L        # 10 nM initial concentration
    dt    =  1.0 * ureg.s                    # 1 s timestep
    delta = np.log(2) / (5.0 * ureg.min)     # half-life 5 min (E. coli mRNA)
    n = int(t / dt)
    out = np.empty(n)
    for i in range(n):
        mrna = mrna - delta * mrna * dt
        out[i] = mrna.to_base_units().magnitude
    return out * ureg.mol / ureg.m**3


def simulate_pint(t: Quantity) -> Quantity:
    mrna  = 10.0 * ureg.nmol / ureg.L
    dt    =  1.0 * ureg.s
    delta = np.log(2) / (5.0 * ureg.min)
    n = int(t / dt)
    out = np.empty(n)
    for i in range(n):
        mrna = mrna - delta * mrna * dt
        out[i] = mrna.to_base_units().magnitude
    return out * ureg.mol / ureg.m**3


T, repeats = 10 * ureg.min, 300

t0 = time.perf_counter()
for _ in range(repeats): simulate_pint(T)
t_pint = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(repeats): simulate_fast(T)
t_fast = time.perf_counter() - t0

print(f"plain Pint: {t_pint / repeats * 1e3:.2f} ms per call")
print(f"unit_jit:   {t_fast / repeats * 1e3:.2f} ms per call  ({t_pint / t_fast:.0f}x vs Pint)")
```

Result on an Apple M3 Pro (600 steps, 300 repetitions):

```
plain Pint: 22.39 ms per call
unit_jit:    0.08 ms per call  (292x vs Pint)
```

The speedup scales with loop length: the longer the loop, the more Pint overhead is avoided per call.

## How it works

1. **Unit inference**: on the first call, all `@unit_jit` functions in the module are rewritten together. The function body is abstract-interpreted with the input units: dimensional errors (e.g. adding meters to seconds) are caught across all branches at this point, and return units are inferred. If source is unavailable, the function falls back to running as plain Pint on every call.
2. **Eager snapshot**: Quantity attributes on objects (e.g. `self.params.alpha`) are pre-converted to SI floats once at boundary entry. Attribute access inside the loop is a plain dict lookup.
3. **Fast zone**: a thread-local flag marks the outermost `@unit_jit` frame. Inner `@unit_jit` calls skip boundary conversion entirely.
4. **Return wrapping**: the SI unit of the return value is determined by abstract interpretation and cached. The registry is captured from the first call's arguments, so results always belong to the same registry that produced them, whether that is `unit_jit.ureg` or a user-supplied one.
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

The primary use case is a tight loop over scalars. `unit_jit` rewrites the function body so that all Pint calls disappear: `ureg.nmol / ureg.L` becomes the corresponding SI float, `.to_base_units()` is stripped, and arithmetic runs on plain floats. The result is wrapped back into a `Quantity` with the inferred units.

```python
import numpy as np
from pint import Quantity
from unit_jit import unit_jit, ureg

@unit_jit
def simulate(t: Quantity) -> Quantity:
    mrna  = 10.0 * ureg.nmol / ureg.L        # 10 nM initial concentration
    dt    =  1.0 * ureg.s                     # 1 s timestep
    delta = np.log(2) / (5.0 * ureg.min)     # half-life 5 min (E. coli mRNA)
    n = int(t / dt)
    out = np.empty(n)
    for i in range(n):
        mrna = mrna - delta * mrna * dt
        out[i] = mrna.to_base_units().magnitude
    return out * ureg.mol / ureg.m**3
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
path_total(path)   # first call: inference + fast; returns 6.0 m as Quantity
path_total(path)   # fast
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
speeds(d, t)   # first call: inference + fast; returns [5., 5., 6.] m/s as Quantity
```

### Class with Quantity attributes

`unit_jit` can be applied to individual methods or to the whole class at once. When applied to a class, it decorates all non-dunder methods automatically.

`unit_jit` snapshots all `Quantity` attributes on `self` once at the outermost boundary entry, replacing them with SI floats. Inner methods skip boundary conversion entirely, so there is no double-conversion overhead.

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

    def simulate(self, t: Quantity) -> Quantity:  # entry point: owns the hot loop
        dt   = 10.0 * ureg.s
        mrna = self.params.alpha / self.params.delta
        n    = int((t / dt).to_base_units().magnitude)
        out  = np.empty(n)
        for i in range(n):
            mrna = mrna + self.rate(mrna) * dt
            out[i] = mrna.to_base_units().magnitude
        return out * ureg.mol / ureg.m**3
```

`simulate` is the entry point: it owns the hot loop and is where boundary conversion happens. `rate` is an inner call, so it receives plain floats directly and its rewritten body runs without any Pint calls.

### Custom registry

You can use your own `UnitRegistry` instead of `unit_jit.ureg`. The registry is captured from the first call's input arguments, so results are wrapped in the same registry and interoperate naturally with the rest of your quantities.

```python
from pint import Quantity, UnitRegistry
from unit_jit import unit_jit

my_ureg = UnitRegistry()

@unit_jit
def scale(x: Quantity, factor: float) -> Quantity:
    return x * factor

x = 3.0 * my_ureg.meter
result = scale(x, 2.0)        # result is a Quantity in my_ureg
result + 1.0 * my_ureg.meter  # works: same registry
```

The module-level `ureg` exported by `unit_jit` remains available as a convenience default; there is no requirement to use it.

## Debugging

To inspect what code actually runs after rewriting, use `get_rewritten_source`. It triggers compilation if needed and returns the rewritten function source as a string.

```python
import numpy as np
from pint import Quantity
from unit_jit import unit_jit, get_rewritten_source, ureg

@unit_jit
def simulate(t: Quantity) -> Quantity:
    mrna  = 10.0 * ureg.nmol / ureg.L        # 10 nM initial concentration
    dt    =  1.0 * ureg.s                    # 1 s timestep
    delta = np.log(2) / (5.0 * ureg.min)     # half-life 5 min (E. coli mRNA)
    n = int(t / dt)
    out = np.empty(n)
    for i in range(n):
        mrna = mrna - delta * mrna * dt
        out[i] = mrna.to_base_units().magnitude
    return out * ureg.mol / ureg.m**3

simulate(10 * ureg.min)  # trigger inference and compilation
print(get_rewritten_source(simulate))
```

Output:

```python
def simulate(t: Quantity) -> Quantity:
    mrna  = 10.0 * 1e-09 / 0.0010000000000000002
    dt    =  1.0 * 1.0
    delta = np.log(2) / (5.0 * 60.0)
    n = int(t / dt)
    out = np.empty(n)
    for i in range(n):
        mrna = mrna - delta * mrna * dt
        out[i] = mrna
    return out * 1.0 / 1.0 ** 3
```

All `ureg` unit references are replaced by their SI float values (`ureg.nmol / ureg.L` becomes `1e-9 / 0.001`, `ureg.min` becomes `60.0`, `ureg.mol / ureg.m**3` becomes `1.0 / 1.0**3`), `.to_base_units().magnitude` is stripped, and the arithmetic is otherwise unchanged.

`get_rewritten_source` shows only what runs in the rewritten version. The boundary is not shown: arguments arrive as plain SI floats (so `t` is a float in seconds, not a `Quantity`), and the raw return value is wrapped back into a `Quantity` by the runtime using the inferred units.

## Numba integration

For functions with a pure float/NumPy inner loop, `use_numba=True` additionally compiles the rewritten function with [Numba](https://numba.readthedocs.io), giving a further speedup on top of the Pint stripping.

```python
import numpy as np
from pint import Quantity
from unit_jit import unit_jit, ureg

@unit_jit(use_numba=True)
def simulate(t: Quantity) -> Quantity:
    mrna  = 10.0 * ureg.nmol / ureg.L        # 10 nM initial concentration
    dt    =  1.0 * ureg.s                     # 1 s timestep
    delta = np.log(2) / (5.0 * ureg.min)     # half-life 5 min (E. coli mRNA)
    n = int(t / dt)
    out = np.empty(n)
    for i in range(n):
        mrna = mrna - delta * mrna * dt
        out[i] = mrna.to_base_units().magnitude
    return out * ureg.mol / ureg.m**3

simulate(10 * ureg.min)  # 1st call: unit inference + compilation
simulate(10 * ureg.min)  # 2nd call: triggers Numba compilation
simulate(10 * ureg.min)  # 3rd call onwards: Numba-compiled float loop
```

Two calls are needed before reaching full speed: the first runs unit inference and CST rewriting, and the second triggers Numba's own JIT compilation. From the third call on, the full pipeline runs at native speed.

On the same mRNA decay benchmark (Apple M3 Pro, 600 steps, 300 repetitions):

```
plain Pint:        23.10 ms per call
unit_jit:           0.08 ms per call   (291x vs Pint)
unit_jit + Numba:   0.01 ms per call  (1687x vs Pint)
```

The additional 5x on top of unit_jit comes from Numba compiling the inner loop to native code. The gain grows with loop complexity and body size. Numba is imported lazily and only required when `use_numba=True` is set.

`use_numba=True` is not suitable for functions that call other `@unit_jit`-decorated methods internally (e.g. `simulate_model` calling `self.rate`), as Numba cannot compile through the Python wrapper.

## Running tests

```bash
pytest
```

## Feedback

If you find this library useful, feel free to drop a message. Hearing about your experience would be very welcome. If you have any suggestions or run into an issue, don't hesitate to [open an issue](https://github.com/BioDisCo/unit-jit/issues).

## License

Apache-2.0
