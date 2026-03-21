"""Tests for get_rewritten_source."""

import numpy as np
from pint import Quantity

from unit_jit import get_rewritten_source, unit_jit, ureg


@unit_jit
def _decay(n: int) -> Quantity:
    mrna = 10.0 * ureg.mol / ureg.L
    dt = 0.1 * ureg.s
    delta = 0.01 / ureg.s
    out = np.empty(n)
    for i in range(n):
        mrna = mrna - delta * mrna * dt
        out[i] = mrna.to_base_units().magnitude
    return out * ureg.mol / ureg.L


@unit_jit
def _identity(x: Quantity) -> Quantity:
    return x


def test_get_rewritten_source_returns_string():
    src = get_rewritten_source(_decay)
    assert isinstance(src, str)
    assert len(src) > 0


def test_get_rewritten_source_triggers_compilation():
    """get_rewritten_source compiles the module even without a prior call."""
    src = get_rewritten_source(_identity)
    assert isinstance(src, str)


def test_ureg_units_replaced_by_floats():
    """ureg.s, ureg.mol, ureg.L should all be replaced by SI float literals."""
    src = get_rewritten_source(_decay)
    assert "ureg" not in src


def test_magnitude_stripped():
    src = get_rewritten_source(_decay)
    assert ".magnitude" not in src


def test_to_base_units_stripped():
    src = get_rewritten_source(_decay)
    assert ".to_base_units()" not in src


def test_function_body_still_valid_python():
    """The rewritten source must be parseable Python."""
    import ast

    src = get_rewritten_source(_decay)
    ast.parse(src)  # raises SyntaxError if invalid
