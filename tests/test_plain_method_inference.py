"""Tests for lazy inference of plain (non-@unit_jit) callee methods.

Two related features are exercised here:

Feature A — plain method callee:
  Root cause: _call only triggered lazy inference for methods with
  __unit_jit_wrapped__ = True.  Plain helper methods (not decorated) fell
  through to _UNKNOWN.
  Fix: a second fallback path in _call tries lazy inference on any callable
  method that is not @unit_jit-decorated, using functools.partial to bind self.

Feature B — list[Quantity] dummy args in lazy callee inference:
  Root cause: when a parameter was inferred as _ListReturn (i.e. list[Quantity]),
  the dummy arg built for the callee was 1.0 (a float).  Indexing into that
  inside the callee (e.g. n_on = state[1]) returned None, propagating _UNKNOWN.
  Fix: _lazy_infer_callee builds a list of dummy Quantity objects matching the
  inferred element units of the _ListReturn.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from pint import Quantity, UnitRegistry

from unit_jit import unit_jit

ureg = UnitRegistry()


# ---------------------------------------------------------------------------
# Feature A: plain method callee
# ---------------------------------------------------------------------------


class _SystemPlainHelper:
    def __init__(self, alpha: Quantity, volume: Quantity) -> None:
        self.alpha = alpha
        self.volume = volume

    # NOT decorated with @unit_jit
    def propensity(self, n: Quantity) -> Quantity:
        return cast("Quantity", self.alpha * n)

    @unit_jit
    def total_rate(self, n: Quantity) -> Quantity:
        return cast("Quantity", self.propensity(n) * self.volume)


def test_plain_helper_result():
    sys = _SystemPlainHelper(
        alpha=2.0 / ureg.s / (ureg.mol / ureg.L),
        volume=1.0 * ureg.L,
    )
    n = 3.0 * ureg.mol / ureg.L
    result = sys.total_rate(n)
    expected = sys.propensity(n) * sys.volume
    assert isinstance(result, Quantity)
    assert abs(result.to_base_units().magnitude - expected.to_base_units().magnitude) < 1e-10


def test_plain_helper_returns_quantity():
    sys = _SystemPlainHelper(1.0 / ureg.s / (ureg.mol / ureg.L), 1.0 * ureg.L)
    assert isinstance(sys.total_rate(1.0 * ureg.mol / ureg.L), Quantity)


def test_plain_helper_dimension():
    sys = _SystemPlainHelper(1.0 / ureg.s / (ureg.mol / ureg.L), 1.0 * ureg.L)
    result = sys.total_rate(1.0 * ureg.mol / ureg.L)
    # (L/(mol·s)) * (mol/L) * L = L/s
    assert result.dimensionality == {"[length]": 3, "[time]": -1}


# ---------------------------------------------------------------------------
# Feature B: list[Quantity] dummy args
# ---------------------------------------------------------------------------


def _plain_rates_from_state(state: Sequence[Quantity], alpha: Quantity) -> list[Quantity]:
    """Undecorated helper that indexes into a list[Quantity] parameter."""
    n_on = state[1]
    return [alpha * n_on]


class _SystemListIndex:
    def __init__(self, alpha: Quantity) -> None:
        self.alpha = alpha

    @unit_jit
    def total_rate(self, state: Sequence[Quantity]) -> Quantity:
        rates = _plain_rates_from_state(state, self.alpha)
        return cast("Quantity", sum(rates))


def test_list_index_result():
    sys = _SystemListIndex(alpha=0.5 / ureg.s / (ureg.mol / ureg.L))
    state = [0.0 * ureg.mol / ureg.L, 2.0 * ureg.mol / ureg.L]
    result = sys.total_rate(state)
    expected = 0.5 / ureg.s / (ureg.mol / ureg.L) * 2.0 * ureg.mol / ureg.L
    assert isinstance(result, Quantity)
    assert abs(result.to_base_units().magnitude - expected.to_base_units().magnitude) < 1e-10


def test_list_index_returns_quantity():
    sys = _SystemListIndex(alpha=1.0 / ureg.s / (ureg.mol / ureg.L))
    state = [0.0 * ureg.mol / ureg.L, 1.0 * ureg.mol / ureg.L]
    assert isinstance(sys.total_rate(state), Quantity)


def test_list_index_dimension():
    sys = _SystemListIndex(alpha=1.0 / ureg.s / (ureg.mol / ureg.L))
    state = [0.0 * ureg.mol / ureg.L, 1.0 * ureg.mol / ureg.L]
    result = sys.total_rate(state)
    # (L/(mol·s)) * (mol/L) = 1/s
    assert result.dimensionality == {"[time]": -1}
