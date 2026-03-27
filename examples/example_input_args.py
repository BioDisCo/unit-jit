"""Pre-compilation with input_args.

By default, unit inference runs on the first call. Pass input_args to the
decorator to trigger it at decoration time instead, so every subsequent call
is immediately fast: no inference overhead on the first real call.

Two cases are shown:
  - scalar Quantity arguments
  - Quantity wrapping a NumPy array
"""

import numpy as np
from pint import Quantity, UnitRegistry

from unit_jit import unit_jit

ureg: UnitRegistry = UnitRegistry()


@unit_jit(input_args=(ureg.m, ureg.s))
def velocity(d: Quantity, t: Quantity) -> Quantity:
    return d / t


@unit_jit(input_args=(np.array([1.0, 2.0, 3.0]) * ureg.m,))
def path_total(path: Quantity) -> Quantity:
    return np.sum(path)


if __name__ == "__main__":
    # Inference already ran during decoration; these calls go straight to the fast path.
    v = velocity(20 * ureg.m, 4 * ureg.s)
    print(f"velocity:   {v}")  # 5.0 m/s

    total = path_total(np.array([10.0, 20.0, 30.0]) * ureg.m)
    print(f"path total: {total}")  # 60.0 m
