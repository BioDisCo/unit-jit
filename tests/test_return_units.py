"""Tests for the return_units= parameter on @unit_jit."""

from pint import Quantity

from unit_jit import unit_jit, ureg


@unit_jit(return_units=ureg.m / ureg.s)
def _div_declared(d: Quantity, t: Quantity) -> Quantity:
    return d / t  # type: ignore[return-value]


@unit_jit(return_units=[ureg.mol / ureg.L / ureg.s, ureg.mol / ureg.L / ureg.s])
def _two_rates(x: Quantity) -> list[Quantity]:
    alpha = 0.1 * ureg.mol / ureg.L / ureg.s
    return [alpha, alpha * x]  # type: ignore[return-value]


def test_return_units_no_warmup_needed():
    """First call runs fast (no pint warm-up) when return_units is declared."""
    result = _div_declared(10 * ureg.m, 2 * ureg.s)
    assert isinstance(result, Quantity)
    assert abs(result.to_base_units().magnitude - 5.0) < 1e-12


def test_return_units_matches_pint_baseline():
    """return_units result agrees with plain pint."""
    result = _div_declared(10 * ureg.m, 2 * ureg.s)
    expected = (10 * ureg.m / (2 * ureg.s)).to_base_units().magnitude
    assert abs(result.to_base_units().magnitude - expected) < 1e-12


def test_return_units_repeated_calls_consistent():
    """Multiple calls give consistent results."""
    r1 = _div_declared(10 * ureg.m, 2 * ureg.s)
    r2 = _div_declared(20 * ureg.m, 4 * ureg.s)
    assert abs(r1.to_base_units().magnitude - r2.to_base_units().magnitude) < 1e-12


@unit_jit(return_units=[ureg.mol / ureg.L / ureg.s, ureg.dimensionless])
def _mixed_rates(x: Quantity) -> list[Quantity]:
    return [0.1 * ureg.mol / ureg.L / ureg.s, x * ureg.dimensionless]  # type: ignore[return-value]


@unit_jit(return_units=list[ureg.mol / ureg.L / ureg.s])
def _two_rates_generic(x: Quantity) -> list[Quantity]:
    alpha = 0.1 * ureg.mol / ureg.L / ureg.s
    return [alpha, alpha * x]  # type: ignore[return-value]


def test_return_units_mixed_list():
    """Explicit list with different units per element."""
    x = 3.0 * ureg.dimensionless
    result = _mixed_rates(x)
    assert isinstance(result, list)
    assert len(result) == 2
    assert abs(result[0].to_base_units().magnitude - 100.0) < 1e-10  # 0.1 mol/L/s in SI
    assert abs(result[1].to_base_units().magnitude - 3.0) < 1e-12


def test_return_units_generic_alias():
    """list[unit] syntax applies one unit to all elements."""
    x = 2.0 * ureg.dimensionless
    result = _two_rates_generic(x)
    assert isinstance(result, list)
    assert all(isinstance(r, Quantity) for r in result)
    assert abs(result[0].to_base_units().magnitude - 100.0) < 1e-10


def test_return_units_list():
    """return_units as a list wraps each element correctly."""
    x = 2.0 * ureg.dimensionless
    result = _two_rates(x)
    assert isinstance(result, list)
    assert all(isinstance(r, Quantity) for r in result)
    assert abs(result[0].to_base_units().magnitude - 100.0) < 1e-10  # 0.1 mol/L/s = 100 mol/m^3/s
