"""Tests for the transcendental-function dimensionless-argument check."""

import math
from typing import cast

import numpy as np
import pytest
from pint import Quantity, UnitRegistry

from unit_jit import unit_jit

ureg = UnitRegistry()

def test_transcendental_dimensionless_ok():
    @unit_jit
    def decay(t: Quantity, tau: Quantity) -> Quantity:
        return cast("Quantity", np.exp(-(t / tau)))

    out = decay(1 * ureg.s, 2 * ureg.s)
    assert math.isclose(float(out), math.exp(-0.5))


def test_transcendental_dimensional_arg_raises():
    @unit_jit
    def bad(t: Quantity, length: Quantity) -> Quantity:
        return cast("Quantity", np.exp(t / length))  # s/m is not dimensionless

    with pytest.raises(TypeError, match="dimensionless argument"):
        bad(1 * ureg.s, 2 * ureg.m)
