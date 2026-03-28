"""Tests for fractional-power unit inference (x ** 0.5, np.sqrt(x)).

These patterns appear in Chemical Langevin noise terms:

    sig = (f / V) ** 0.5 * np.sqrt(dt) * z

where f is a reaction rate density [1/(min·fL)], V is volume [fL], and dt is
time [min]. The result has units [1/fL], which is the correct noise increment
unit for mRNA concentration.

All tests have a plain-Pint reference result that the unit_jit result must match
within floating-point tolerance.
"""

from typing import cast

import numpy as np
import pytest
from pint import Quantity, UnitRegistry

from unit_jit import unit_jit

ureg = UnitRegistry()


# ---------------------------------------------------------------------------
# Scalar: (x ** 0.5)
# ---------------------------------------------------------------------------


@unit_jit
def _sqrt_quantity_scalar(x: Quantity) -> Quantity:
    """Square root of a scalar Quantity: x ** 0.5."""
    return cast("Quantity", x**0.5)


def test_sqrt_quantity_scalar_value():
    x = 4.0 * ureg.m**2
    result = _sqrt_quantity_scalar(x)
    expected = (4.0 * ureg.m**2) ** 0.5
    assert result.to("m").magnitude == pytest.approx(expected.to("m").magnitude)


def test_sqrt_quantity_scalar_unit():
    x = 9.0 * ureg.m**2
    result = _sqrt_quantity_scalar(x)
    assert result.dimensionality == {"[length]": 1}


# ---------------------------------------------------------------------------
# Scalar: np.sqrt(x) where x is a Quantity
# ---------------------------------------------------------------------------


@unit_jit
def _sqrt_np_scalar(x: Quantity) -> Quantity:
    """np.sqrt applied to a Quantity."""
    return cast("Quantity", np.sqrt(x))


def test_sqrt_np_scalar_value():
    x = 4.0 * ureg.m**2
    result = _sqrt_np_scalar(x)
    expected = np.sqrt(4.0 * ureg.m**2)
    assert result.to("m").magnitude == pytest.approx(expected.to("m").magnitude)


# ---------------------------------------------------------------------------
# CLE noise pattern: (f / V) ** 0.5 * np.sqrt(dt) * z
# This is the exact expression from ChemicalLangevinSystem / CLEOptSystem.
# ---------------------------------------------------------------------------


@unit_jit
def _cle_noise_amplitude(f: Quantity, V: Quantity, dt: Quantity, z: float) -> Quantity:
    """CLE noise increment: sqrt(f/V) * sqrt(dt) * z.

    Units: sqrt([rate/volume] * [time]) = sqrt(1/(fL^2)) = 1/fL.
    """
    return cast("Quantity", (f / V) ** 0.5 * np.sqrt(dt) * z)


def test_cle_noise_amplitude_value():
    f = 2.0 / ureg.minute / ureg.femtoliter  # rate density [1/(min·fL)]
    V = 1.0 * ureg.femtoliter
    dt = 0.1 * ureg.minute
    z = 1.5
    result = _cle_noise_amplitude(f, V, dt, z)
    expected = (f / V) ** 0.5 * np.sqrt(dt) * z
    assert result.to("1/femtoliter").magnitude == pytest.approx(
        expected.to("1/femtoliter").magnitude, rel=1e-9
    )


def test_cle_noise_amplitude_unit():
    f = 2.0 / ureg.minute / ureg.femtoliter
    V = 1.0 * ureg.femtoliter
    dt = 0.1 * ureg.minute
    result = _cle_noise_amplitude(f, V, dt, 1.0)
    assert result.dimensionality == {"[length]": -3}


# ---------------------------------------------------------------------------
# Batched version: magnitudes are numpy arrays (n,) as in simulate_sde_batch.
# ---------------------------------------------------------------------------


@unit_jit
def _cle_noise_amplitude_batch(f: Quantity, V: Quantity, dt: Quantity, z: np.ndarray) -> Quantity:
    """Batch CLE noise: z has shape (n,)."""
    return cast("Quantity", (f / V) ** 0.5 * np.sqrt(dt) * z)


def test_cle_noise_amplitude_batch_shape():
    n = 100
    f = 2.0 / ureg.minute / ureg.femtoliter
    V = 1.0 * ureg.femtoliter
    dt = 0.1 * ureg.minute
    z = np.random.default_rng(0).standard_normal(n)
    result = _cle_noise_amplitude_batch(f, V, dt, z)
    assert result.to("1/femtoliter").magnitude.shape == (n,)


def test_cle_noise_amplitude_batch_values():
    n = 50
    f = 3.0 / ureg.minute / ureg.femtoliter
    V = 1.0 * ureg.femtoliter
    dt = 0.1 * ureg.minute
    z = np.random.default_rng(1).standard_normal(n)
    result = _cle_noise_amplitude_batch(f, V, dt, z)
    expected = (f / V) ** 0.5 * np.sqrt(dt) * z
    np.testing.assert_allclose(
        result.to("1/femtoliter").magnitude,
        expected.to("1/femtoliter").magnitude,
        rtol=1e-9,
    )


# ---------------------------------------------------------------------------
# Two-component CLE noise (CLEOptSystem pattern):
# sig = ((f_birth + f_death) / V) ** 0.5 * np.sqrt(dt) * z
# ---------------------------------------------------------------------------


@unit_jit
def _cleopt_noise(
    f_birth: Quantity, f_death: Quantity, V: Quantity, dt: Quantity, z: float
) -> Quantity:
    """Combined CLE noise from birth and death reactions."""
    return cast("Quantity", ((f_birth + f_death) / V) ** 0.5 * np.sqrt(dt) * z)


def test_cleopt_noise_value():
    f_birth = 1.0 / ureg.minute / ureg.femtoliter
    f_death = 0.5 / ureg.minute / ureg.femtoliter
    V = 1.0 * ureg.femtoliter
    dt = 0.1 * ureg.minute
    z = 0.8
    result = _cleopt_noise(f_birth, f_death, V, dt, z)
    expected = ((f_birth + f_death) / V) ** 0.5 * np.sqrt(dt) * z
    assert result.to("1/femtoliter").magnitude == pytest.approx(
        expected.to("1/femtoliter").magnitude, rel=1e-9
    )


# ---------------------------------------------------------------------------
# list[Quantity] return: the noise methods return a list of Quantities.
# This tests that unit_jit handles list[Quantity] returns with ** 0.5 inside.
# ---------------------------------------------------------------------------


@unit_jit
def _cle_noise_list(
    f_birth: Quantity, f_death: Quantity, V: Quantity, dt: Quantity, z: float
) -> list[Quantity]:
    """Noise function returning list[Quantity], as in the simulation models."""
    sig = cast("Quantity", ((f_birth + f_death) / V) ** 0.5 * np.sqrt(dt) * z)
    return [sig]


def test_cle_noise_list_length():
    f_b = 1.0 / ureg.minute / ureg.femtoliter
    f_d = 0.5 / ureg.minute / ureg.femtoliter
    V = 1.0 * ureg.femtoliter
    dt = 0.1 * ureg.minute
    result = _cle_noise_list(f_b, f_d, V, dt, 1.0)
    assert len(result) == 1


def test_cle_noise_list_value():
    f_b = 1.0 / ureg.minute / ureg.femtoliter
    f_d = 0.5 / ureg.minute / ureg.femtoliter
    V = 1.0 * ureg.femtoliter
    dt = 0.1 * ureg.minute
    z = 1.3
    result = _cle_noise_list(f_b, f_d, V, dt, z)
    expected = ((f_b + f_d) / V) ** 0.5 * np.sqrt(dt) * z
    assert result[0].to("1/femtoliter").magnitude == pytest.approx(
        expected.to("1/femtoliter").magnitude, rel=1e-9
    )
