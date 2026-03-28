"""Tests for unit_jit inference with numpy RNG methods.

RNG calls such as rng.poisson(), rng.binomial(), and rng.geometric() return
plain int / ndarray values (dimensionless). The unit_jit abstract interpreter
must recognise them as dimensionless so that downstream arithmetic — e.g.
dividing Poisson counts by a volume — yields the correct units rather than
propagating an opaque 'object' type that breaks the wrapping step.

Patterns tested here are taken directly from the tau-leaping and geometric-burst
noise functions used in our SDE simulations:

    # tau-leaping net change
    total_rate = (alpha + delta * x) * dt * V   # [1/min/fL * min * fL] = dimensionless
    n_events   = rng.poisson(total_rate.magnitude)
    n_births   = rng.binomial(n_events, p)
    return (n_births - (n_events - n_births)) / V  # [1/fL]

    # geometric burst
    n_jumps = rng.poisson(rate.magnitude)
    sizes   = rng.geometric(p, size=n_jumps.sum()) - 1
    return total_burst / V  # [1/fL]
"""

from typing import cast

import numpy as np
import pytest
from pint import Quantity, UnitRegistry

from unit_jit import unit_jit

ureg = UnitRegistry()


# ---------------------------------------------------------------------------
# rng.poisson: dimensionless count / Quantity volume → Quantity
# ---------------------------------------------------------------------------


@unit_jit
def _poisson_counts_per_volume(
    rng: np.random.Generator,
    rate: Quantity,
    dt: Quantity,
    volume: Quantity,
) -> Quantity:
    """Poisson birth count divided by volume: canonical CLE birth term."""
    expected = (rate * dt * volume).magnitude  # dimensionless float
    n = rng.poisson(expected)
    return cast("Quantity", n / volume)


def test_poisson_counts_per_volume_unit():
    rng = np.random.default_rng(0)
    rate = 1.0 / ureg.minute / ureg.femtoliter
    dt = 0.1 * ureg.minute
    volume = 1.0 * ureg.femtoliter
    result = _poisson_counts_per_volume(rng, rate, dt, volume)
    assert result.dimensionality == {"[length]": -3}


def test_poisson_counts_per_volume_nonnegative():
    rng = np.random.default_rng(1)
    rate = 1.0 / ureg.minute / ureg.femtoliter
    dt = 0.1 * ureg.minute
    volume = 1.0 * ureg.femtoliter
    result = _poisson_counts_per_volume(rng, rate, dt, volume)
    assert result.magnitude >= 0


# ---------------------------------------------------------------------------
# rng.binomial: dimensionless count
# ---------------------------------------------------------------------------


@unit_jit
def _binomial_net_change(
    rng: np.random.Generator,
    total_events: int,
    birth_prob: float,
    volume: Quantity,
) -> Quantity:
    """Net mRNA change (births - deaths) / volume from binomial thinning."""
    n_births = rng.binomial(total_events, birth_prob)
    n_deaths = total_events - n_births
    return cast("Quantity", (n_births - n_deaths) / volume)


def test_binomial_net_change_unit():
    rng = np.random.default_rng(2)
    volume = 1.0 * ureg.femtoliter
    result = _binomial_net_change(rng, 10, 0.5, volume)
    assert result.dimensionality == {"[length]": -3}


def test_binomial_net_change_value_range():
    rng = np.random.default_rng(3)
    volume = 1.0 * ureg.femtoliter
    result = _binomial_net_change(rng, 10, 0.5, volume)
    assert abs(result.to("1/femtoliter").magnitude) <= 10.0


# ---------------------------------------------------------------------------
# Tau-leaping pattern: full net-change expression from _tl_net_change
# ---------------------------------------------------------------------------


@unit_jit
def _tl_net_change(
    rng: np.random.Generator,
    alpha_total: Quantity,
    delta: Quantity,
    mrna_now: Quantity,
    dt: Quantity,
    volume: Quantity,
) -> Quantity:
    """Net mRNA change via Poisson thinning (births - deaths) / volume."""
    total_rate = (alpha_total + delta * mrna_now) * dt * volume
    n_events = rng.poisson(total_rate.magnitude)
    birth_prob = (alpha_total / (alpha_total + delta * mrna_now)).magnitude
    n_births = rng.binomial(n_events, birth_prob)
    return cast("Quantity", (n_births - (n_events - n_births)) / volume)


def test_tl_net_change_unit():
    rng = np.random.default_rng(4)
    alpha = 2.0 / ureg.minute / ureg.femtoliter
    delta = 0.1 / ureg.minute
    mrna = 5.0 / ureg.femtoliter
    dt = 0.1 * ureg.minute
    V = 1.0 * ureg.femtoliter
    result = _tl_net_change(rng, alpha, delta, mrna, dt, V)
    assert result.dimensionality == {"[length]": -3}


def test_tl_net_change_reference_distribution():
    """Empirical mean of net change matches E[births-deaths] = (alpha - delta*x)*dt."""
    rng = np.random.default_rng(5)
    alpha = 2.0 / ureg.minute / ureg.femtoliter
    delta = 0.1 / ureg.minute
    mrna = 5.0 / ureg.femtoliter
    dt = 0.1 * ureg.minute
    V = 1.0 * ureg.femtoliter

    n_samples = 20_000
    samples = np.array(
        [
            _tl_net_change(np.random.default_rng(i), alpha, delta, mrna, dt, V)
            .to("1/femtoliter")
            .magnitude
            for i in range(n_samples)
        ]
    )
    expected_mean = ((alpha - delta * mrna) * dt).to("1/femtoliter").magnitude
    assert samples.mean() == pytest.approx(expected_mean, abs=0.05)


# ---------------------------------------------------------------------------
# rng.geometric: dimensionless burst sizes
# ---------------------------------------------------------------------------


@unit_jit
def _geometric_burst(
    rng: np.random.Generator,
    n_jumps: int,
    geom_p: float,
    volume: Quantity,
) -> Quantity:
    """Total geometric burst size / volume.  rng.geometric returns plain ints."""
    sizes = rng.geometric(geom_p, size=n_jumps) - 1
    return cast("Quantity", float(sizes.sum()) / volume)


def test_geometric_burst_unit():
    rng = np.random.default_rng(6)
    volume = 1.0 * ureg.femtoliter
    result = _geometric_burst(rng, 5, 0.5, volume)
    assert result.dimensionality == {"[length]": -3}


def test_geometric_burst_nonnegative():
    rng = np.random.default_rng(7)
    volume = 1.0 * ureg.femtoliter
    result = _geometric_burst(rng, 10, 0.3, volume)
    assert result.to("1/femtoliter").magnitude >= 0


# ---------------------------------------------------------------------------
# Batch (shape-(n,)) variant: rng.poisson with array-valued expected counts
# ---------------------------------------------------------------------------


@unit_jit
def _tl_net_change_batch(
    rng: np.random.Generator,
    alpha_total: Quantity,
    delta: Quantity,
    mrna_now: Quantity,
    dt: Quantity,
    volume: Quantity,
) -> Quantity:
    """Batch version: mrna_now.magnitude has shape (n,)."""
    total_rate = (alpha_total + delta * mrna_now) * dt * volume
    n_events = rng.poisson(total_rate.magnitude)
    birth_prob = (alpha_total / (alpha_total + delta * mrna_now)).magnitude
    n_births = rng.binomial(n_events, birth_prob)
    return cast("Quantity", (n_births - (n_events - n_births)) / volume)


def test_tl_net_change_batch_shape():
    n = 100
    rng = np.random.default_rng(8)
    alpha = 2.0 / ureg.minute / ureg.femtoliter
    delta = 0.1 / ureg.minute
    mrna = np.ones(n) * 5.0 / ureg.femtoliter
    dt = 0.1 * ureg.minute
    V = 1.0 * ureg.femtoliter
    result = _tl_net_change_batch(rng, alpha, delta, mrna, dt, V)
    assert result.to("1/femtoliter").magnitude.shape == (n,)


def test_tl_net_change_batch_unit():
    n = 50
    rng = np.random.default_rng(9)
    alpha = 2.0 / ureg.minute / ureg.femtoliter
    delta = 0.1 / ureg.minute
    mrna = np.ones(n) * 3.0 / ureg.femtoliter
    dt = 0.1 * ureg.minute
    V = 1.0 * ureg.femtoliter
    result = _tl_net_change_batch(rng, alpha, delta, mrna, dt, V)
    assert result.dimensionality == {"[length]": -3}
