"""Tests for the fast_zone context manager."""

from typing import cast

import numpy as np
from pint import Quantity

from unit_jit import fast_zone, unit_jit, ureg


@unit_jit
class _DecayModel:
    def __init__(self, delta: Quantity) -> None:
        self.delta = delta

    def rate(self, x: Quantity) -> Quantity:
        return cast("Quantity", -self.delta * x)


def test_fast_zone_result_matches_pint():
    """fast_zone loop gives same trajectory as plain Pint."""
    model = _DecayModel(delta=0.01 * ureg.s**-1)
    x0 = 10.0 * ureg.mol / ureg.L
    dt = 0.1 * ureg.s
    n = 50

    x = x0
    pint_out = []
    model.rate(x0)  # warm-up
    for _ in range(n):
        x = x + model.rate(x) * dt  # type: ignore[assignment]
        pint_out.append(x.to_base_units().magnitude)

    x_si = x0.to_base_units().magnitude
    dt_si = dt.to_base_units().magnitude
    fast_out = []
    with fast_zone(model) as (fast_model,):
        for _ in range(n):
            x_si = x_si + fast_model.rate(x_si) * dt_si
            fast_out.append(x_si)

    np.testing.assert_allclose(fast_out, pint_out, rtol=1e-10)


def test_fast_zone_no_objects():
    """fast_zone() with no arguments still enters the fast zone."""
    with fast_zone() as proxies:
        assert proxies == ()


def test_fast_zone_nesting_is_noop():
    """Entering fast_zone from within fast_zone leaves objects unconverted."""
    model = _DecayModel(delta=0.01 * ureg.s**-1)
    model.rate(5.0 * ureg.mol / ureg.L)  # warm-up
    with fast_zone(model) as (_,):
        with fast_zone(model) as (inner,):
            assert inner is model  # no snapshot: already in fast zone


def test_fast_zone_multiple_objects():
    """fast_zone accepts and yields multiple objects."""
    m1 = _DecayModel(delta=0.01 * ureg.s**-1)
    m2 = _DecayModel(delta=0.05 * ureg.s**-1)
    m1.rate(1.0 * ureg.mol / ureg.L)  # warm-up
    m2.rate(1.0 * ureg.mol / ureg.L)
    x_si = 10.0
    with fast_zone(m1, m2) as (f1, f2):
        r1 = f1.rate(x_si)
        r2 = f2.rate(x_si)
    assert abs(r1 - (-0.01 * x_si)) < 1e-12
    assert abs(r2 - (-0.05 * x_si)) < 1e-12
