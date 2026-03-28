"""Tests for unit_jit inference across call boundaries involving RNG.

The failing pattern from production code (stochastic-noise-in-microbial-cells):

    @unit_jit
    def _tl_net_change(rng, alpha_total, delta, mrna_now, dt, volume) -> Quantity:
        total_rate = (alpha_total + delta * mrna_now) * dt * volume
        n_events   = rng.poisson(total_rate.magnitude)
        n_births   = rng.binomial(n_events, birth_prob)
        return (n_births - (n_events - n_births)) / volume

    class TauLeapingSystem:
        @unit_jit
        def noise(self, rng, _t, y) -> list[Quantity]:
            return [_tl_net_change(rng, self.alpha, self.delta, y[-1], self.dt, self.volume)]

Calling sys.noise(...) raises:
    TypeError: units must be of type str, PlainQuantity or UnitsContainer; not <class 'object'>

Root cause: when unit_jit infers units for the method `noise`, it encounters the call
to `_tl_net_change` (also decorated with @unit_jit) and fails to resolve its return
unit, storing `object` instead of the correct unit (1/volume). The _wrap step then
crashes trying to construct a Quantity with unit=object.

The fix: unit_jit's inference engine must look up or re-evaluate the return unit of
a called @unit_jit function rather than returning `object`.
"""

from collections.abc import Sequence
from typing import cast

import numpy as np
from pint import Quantity, UnitRegistry

from unit_jit import unit_jit

ureg = UnitRegistry()


# ---------------------------------------------------------------------------
# Module-level helper (mirrors _tl_net_change in models.py)
# ---------------------------------------------------------------------------


@unit_jit
def _net_change_helper(
    rng: np.random.Generator,
    rate: Quantity,
    volume: Quantity,
    dt: Quantity,
) -> Quantity:
    """Poisson counts / volume — the simplest cross-call RNG pattern."""
    n = rng.poisson((rate * dt * volume).magnitude)
    return cast("Quantity", n / volume)


# ---------------------------------------------------------------------------
# Class whose method calls the module-level @unit_jit helper
# (mirrors TauLeapingSystem.noise calling _tl_net_change)
# ---------------------------------------------------------------------------


class _SimpleSystem:
    """Minimal stand-in for TauLeapingSystem."""

    def __init__(self, rate: Quantity, volume: Quantity, dt: Quantity) -> None:
        self.rate = rate
        self.volume = volume
        self.dt = dt

    @unit_jit
    def noise(
        self,
        rng: np.random.Generator,
        _t: Quantity,
        y: Sequence[Quantity],  # noqa: ARG002
    ) -> list[Quantity]:
        """Returns list[Quantity] by delegating to the module-level helper."""
        return [_net_change_helper(rng, self.rate, self.volume, self.dt)]


def test_cross_call_returns_list():
    sys = _SimpleSystem(
        rate=2.0 / ureg.minute / ureg.femtoliter,
        volume=1.0 * ureg.femtoliter,
        dt=0.1 * ureg.minute,
    )
    rng = np.random.default_rng(0)
    result = sys.noise(rng, 0.0 * ureg.minute, [5.0 / ureg.femtoliter])
    assert isinstance(result, list)
    assert len(result) == 1


def test_cross_call_element_is_quantity():
    sys = _SimpleSystem(
        rate=2.0 / ureg.minute / ureg.femtoliter,
        volume=1.0 * ureg.femtoliter,
        dt=0.1 * ureg.minute,
    )
    rng = np.random.default_rng(1)
    result = sys.noise(rng, 0.0 * ureg.minute, [5.0 / ureg.femtoliter])
    assert isinstance(result[0], Quantity)


def test_cross_call_unit():
    sys = _SimpleSystem(
        rate=2.0 / ureg.minute / ureg.femtoliter,
        volume=1.0 * ureg.femtoliter,
        dt=0.1 * ureg.minute,
    )
    rng = np.random.default_rng(2)
    result = sys.noise(rng, 0.0 * ureg.minute, [5.0 / ureg.femtoliter])
    assert result[0].dimensionality == {"[length]": -3}


# ---------------------------------------------------------------------------
# Full tau-leaping cross-call pattern (births - deaths split via binomial)
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
    """Exact replica of _tl_net_change from models.py."""
    total_rate = (alpha_total + delta * mrna_now) * dt * volume
    n_events = rng.poisson(total_rate.magnitude)
    birth_prob = (alpha_total / (alpha_total + delta * mrna_now)).magnitude
    n_births = rng.binomial(n_events, birth_prob)
    return cast("Quantity", (n_births - (n_events - n_births)) / volume)


class _TauLeapingSystem:
    """Minimal replica of TauLeapingSystem."""

    def __init__(
        self,
        alpha: Quantity,
        delta: Quantity,
        n_plasmids: int,
        volume: Quantity,
        dt: Quantity,
    ) -> None:
        self.alpha = alpha * n_plasmids
        self.delta = delta
        self.volume = volume
        self.dt = dt

    @unit_jit
    def noise(
        self,
        rng: np.random.Generator,
        _t: Quantity,
        y: Sequence[Quantity],
    ) -> list[Quantity]:
        """Mirror of TauLeapingSystem.noise."""
        return [_tl_net_change(rng, self.alpha, self.delta, y[-1], self.dt, self.volume)]


def test_tl_noise_unit():
    sys = _TauLeapingSystem(
        alpha=1.0 / ureg.minute / ureg.femtoliter,
        delta=0.1 / ureg.minute,
        n_plasmids=10,
        volume=1.0 * ureg.femtoliter,
        dt=0.1 * ureg.minute,
    )
    rng = np.random.default_rng(3)
    result = sys.noise(rng, 0.0 * ureg.minute, [5.0 / ureg.femtoliter])
    assert result[0].dimensionality == {"[length]": -3}


def test_tl_noise_returns_quantity():
    """Result is a Quantity with correct unit and a finite magnitude."""
    alpha = 1.0 / ureg.minute / ureg.femtoliter
    delta = 0.1 / ureg.minute
    mrna = 5.0 / ureg.femtoliter
    V = 1.0 * ureg.femtoliter
    dt = 0.1 * ureg.minute
    sys = _TauLeapingSystem(alpha=alpha, delta=delta, n_plasmids=10, volume=V, dt=dt)
    rng = np.random.default_rng(4)
    result = sys.noise(rng, 0.0 * ureg.minute, [mrna])
    assert isinstance(result[0], Quantity)
    assert result[0].dimensionality == {"[length]": -3}
    assert np.isfinite(result[0].to("1/femtoliter").magnitude)


def test_tl_noise_batch_shape():
    """Batch (shape-(n,)) magnitudes must be preserved through the cross-call."""
    n = 100
    sys = _TauLeapingSystem(
        alpha=2.0 / ureg.minute / ureg.femtoliter,
        delta=0.1 / ureg.minute,
        n_plasmids=5,
        volume=1.0 * ureg.femtoliter,
        dt=0.1 * ureg.minute,
    )
    rng = np.random.default_rng(5)
    mrna = np.ones(n) * 5.0 / ureg.femtoliter
    result = sys.noise(rng, 0.0 * ureg.minute, [mrna])
    assert result[0].to("1/femtoliter").magnitude.shape == (n,)
