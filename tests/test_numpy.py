"""Tests for unit_jit with NumPy array quantities."""

from typing import cast

import numpy as np
from pint import Quantity

from unit_jit import unit_jit, ureg


# Shared decorated functions


@unit_jit
def _vec_div(d: Quantity, t: Quantity) -> Quantity:
    """Element-wise divide a distance array by a time scalar."""
    return cast("Quantity", d / t)


@unit_jit
def _l2_norm(v: Quantity) -> float:
    """L2 norm of a Quantity array; returns SI magnitude as plain float."""
    mags = v.magnitude
    return float(np.sqrt(np.dot(mags, mags)))


@unit_jit
def _scale_plain(x: np.ndarray, factor: float) -> np.ndarray:
    """Scale a plain ndarray by a plain float; no Pint involved."""
    return x * factor


@unit_jit
def _weighted_sum(vals: Quantity, weights: np.ndarray) -> Quantity:
    """Dot product of a Quantity array with a plain weight vector."""
    return cast("Quantity", (vals * weights).sum())



# Array Quantity in, array Quantity out


def test_vec_div_returns_quantity():
    v = np.array([3.0, 4.0, 0.0]) * ureg.m
    result = _vec_div(v, 2.0 * ureg.s)
    assert isinstance(result, Quantity)


def test_vec_div_shape_preserved():
    v = np.array([3.0, 4.0, 0.0]) * ureg.m
    t = 2.0 * ureg.s
    _vec_div(v, t)  # warm-up
    result = _vec_div(v, t)
    assert result.to_base_units().magnitude.shape == (3,)


def test_vec_div_values_match_pint():
    v = np.array([3.0, 4.0, 0.0]) * ureg.m
    t = 2.0 * ureg.s
    _vec_div(v, t)  # warm-up
    result = _vec_div(v * 2, t)
    expected = (v * 2 / t).to_base_units().magnitude
    np.testing.assert_allclose(result.to_base_units().magnitude, expected)


def test_vec_div_non_si_units():
    """Input in non-SI units; fast path must still give correct SI result."""
    v_cm = np.array([300.0, 400.0]) * ureg.cm  # 3 m, 4 m
    t_ms = 2000.0 * ureg.ms  # 2 s
    _vec_div(v_cm, t_ms)  # warm-up
    result = _vec_div(v_cm, t_ms)
    np.testing.assert_allclose(result.to_base_units().magnitude, [1.5, 2.0], rtol=1e-12)


# Array Quantity in, scalar out


def test_l2_norm_returns_float():
    v = np.array([3.0, 4.0]) * ureg.m
    _l2_norm(v)  # warm-up
    result = _l2_norm(v)
    assert isinstance(result, float)


def test_l2_norm_value():
    v = np.array([3.0, 4.0]) * ureg.m
    _l2_norm(v)  # warm-up
    result = _l2_norm(v)
    assert abs(result - 5.0) < 1e-12


def test_l2_norm_non_si_units():
    """Norm of input in cm should equal SI magnitude norm (in metres)."""
    v_cm = np.array([300.0, 400.0]) * ureg.cm  # 3 m, 4 m
    _l2_norm(v_cm)  # warm-up
    result = _l2_norm(v_cm)
    assert abs(result - 5.0) < 1e-12


# Plain ndarray passthrough


def test_plain_ndarray_passthrough_type():
    x = np.array([1.0, 2.0, 3.0])
    _scale_plain(x, 1.0)  # warm-up
    result = _scale_plain(x, 2.0)
    assert isinstance(result, np.ndarray)


def test_plain_ndarray_passthrough_values():
    x = np.array([1.0, 2.0, 3.0])
    _scale_plain(x, 1.0)  # warm-up
    result = _scale_plain(x, 3.0)
    np.testing.assert_array_equal(result, x * 3.0)


# Mixed Quantity array + plain ndarray


def test_weighted_sum_returns_quantity():
    vals = np.array([1.0, 2.0, 3.0]) * ureg.m
    w = np.array([0.5, 0.3, 0.2])
    result = _weighted_sum(vals, w)
    assert isinstance(result, Quantity)


def test_weighted_sum_value():
    vals = np.array([1.0, 2.0, 3.0]) * ureg.m
    w = np.array([0.5, 0.3, 0.2])
    _weighted_sum(vals, w)  # warm-up
    result = _weighted_sum(vals, w)
    expected = float(np.dot(vals.to_base_units().magnitude, w))
    assert abs(result.to_base_units().magnitude - expected) < 1e-12


