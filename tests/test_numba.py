"""Tests for unit_jit(use_numba=True)."""

import numpy as np
import pytest
from pint import Quantity

numba = pytest.importorskip("numba")

from unit_jit import unit_jit, ureg  # noqa: E402


@unit_jit(use_numba=True)
def simulate_nb(t: Quantity) -> Quantity:
    mrna = 10.0 * ureg.nmol / ureg.L  # 10 nM initial concentration
    dt = 1.0 * ureg.s  # 1 s timestep
    delta = np.log(2) / (5.0 * ureg.min)  # half-life 5 min (E. coli mRNA)
    n = int((t / dt).to_base_units().magnitude)
    out = np.empty(n)
    for i in range(n):
        mrna = mrna - delta * mrna * dt
        out[i] = mrna.to_base_units().magnitude
    return out * ureg.mol / ureg.m**3


def test_numba_returns_quantity():
    simulate_nb(5 * ureg.min)  # warm-up
    result = simulate_nb(10 * ureg.min)
    assert isinstance(result, Quantity)


def test_numba_shape_and_finite():
    simulate_nb(5 * ureg.min)  # warm-up
    result = simulate_nb(10 * ureg.min)
    assert result.magnitude.shape == (600,)
    assert np.all(np.isfinite(result.magnitude))


def test_numba_matches_plain_unitjit():
    """unit_jit(use_numba=True) gives the same values as plain unit_jit."""

    @unit_jit
    def simulate_plain(t: Quantity) -> Quantity:
        mrna = 10.0 * ureg.nmol / ureg.L
        dt = 1.0 * ureg.s
        delta = np.log(2) / (5.0 * ureg.min)
        n = int((t / dt).to_base_units().magnitude)
        out = np.empty(n)
        for i in range(n):
            mrna = mrna - delta * mrna * dt
            out[i] = mrna.to_base_units().magnitude
        return out * ureg.mol / ureg.m**3

    simulate_plain(5 * ureg.min)
    simulate_nb(5 * ureg.min)
    out_plain = simulate_plain(10 * ureg.min)
    out_nb = simulate_nb(10 * ureg.min)
    np.testing.assert_allclose(out_plain.magnitude, out_nb.magnitude)
