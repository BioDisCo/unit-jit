# unit-jit

JIT unit-stripping decorator for [Pint](https://pint.readthedocs.io)-annotated Python. Write clean, unit-safe code; pay no Pint overhead in hot loops.

```python
from pint import Quantity
from unit_jit import unit_jit, ureg

@unit_jit
def velocity(d: Quantity, t: Quantity) -> Quantity:
    return d / t

velocity(10 * ureg.m, 2 * ureg.s)   # warm-up (runs Pint)
velocity(10 * ureg.m, 2 * ureg.s)   # fast (pure float internally)
velocity(10 * ureg.cm, 2 * ureg.s)  # fine — same dimension, different unit
velocity(10 * ureg.m, 2 * ureg.m)   # TypeError — wrong dimension for arg 1
```

First call runs the original Pint function (warm-up). All subsequent calls run a rewritten, pure-float version.

## How it works

1. **Module-level compilation** — on first call, all `@unit_jit` functions in the same module are rewritten together: `.magnitude`, `.to_base_units()`, and `cast("Quantity", x)` are stripped from the source.
2. **Eager snapshot** — Quantity attributes on objects (e.g. `self.params.alpha`) are pre-converted to SI floats once at boundary entry. Attribute access inside the loop is a plain dict lookup.
3. **Fast zone** — a thread-local flag marks the outermost `@unit_jit` frame. Inner `@unit_jit` calls skip boundary conversion entirely.
4. **Return wrapping** — the SI unit of the return value is inferred from the first call and used to wrap subsequent results back into `Quantity`.
5. **Dimension guard** — argument dimensions are cached from the first call; any later call with a different dimension raises `TypeError` immediately.

The right entry point is the **outermost function that owns the hot loop** — not the leaf functions it calls.

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

### Simple function

```python
from pint import Quantity
from unit_jit import unit_jit, ureg

@unit_jit
def velocity(d: Quantity, t: Quantity) -> Quantity:
    return d / t

velocity(10 * ureg.m, 2 * ureg.s)   # warm-up (runs Pint)
velocity(10 * ureg.m, 2 * ureg.s)   # fast (pure float internally)
velocity(10 * ureg.cm, 2 * ureg.s)  # fine — same dimension, different unit
velocity(10 * ureg.m, 2 * ureg.m)   # TypeError — wrong dimension for arg 1
```

### Loop with unit arithmetic

```python
import numpy as np
from unit_jit import unit_jit, ureg

@unit_jit
def simulate(n: int) -> np.ndarray:
    mrna = 10.0 * ureg.mol / ureg.L
    dt   =  0.1 * ureg.s
    delta = 0.01 / ureg.s
    out = np.empty(n)
    for i in range(n):
        mrna = mrna - delta * mrna * dt
        out[i] = mrna.to_base_units().magnitude
    return out
```

### Class with Quantity attributes

```python
from dataclasses import dataclass

import numpy as np
from pint import Quantity
from unit_jit import unit_jit, ureg

@dataclass
class Params:
    alpha: Quantity   # [mol/L/s]
    delta: Quantity   # [1/s]

class Model:
    def __init__(self, params: Params) -> None:
        self.params = params

    @unit_jit
    def rate(self, mrna: Quantity) -> Quantity:
        return self.params.alpha - self.params.delta * mrna

    @unit_jit                          # ← entry point: owns the hot loop
    def simulate(self, n: int) -> np.ndarray:
        mrna = self.params.alpha / self.params.delta
        out = np.empty(n)
        for i in range(n):
            mrna = mrna + self.rate(mrna) * (0.1 * ureg.s)   # rate() in fast zone
            out[i] = mrna.to_base_units().magnitude
        return out
```

`self.params.alpha` and all other Quantity attributes are converted to SI floats once when `simulate` is first called fast; `self.rate()` is called from inside the fast zone, so it skips boundary conversion entirely.

## Running tests

```bash
pytest
```

## License

Apache-2.0
