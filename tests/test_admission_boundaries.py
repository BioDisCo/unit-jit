"""Composition tests for shared value, callable and identity admission rules."""

import functools
from dataclasses import dataclass

import numpy as np
import pytest
from _jit_state import expect_execution
from pint import UnitRegistry

from unit_jit import unit_jit

ureg = UnitRegistry()
STEP = 2 * ureg.cm


def default_helper(x, y=STEP):
    return x + y


def keyword_helper(x, *, y=STEP):
    return x + y


def default_caller(x):
    return default_helper(x)


def keyword_caller(x):
    return keyword_helper(x)


def explicit_caller(x, y):
    return default_helper(x, y)


@pytest.mark.parametrize("function", [default_caller, keyword_caller])
def test_omitted_quantity_defaults_use_original_execution(function):
    wrapped = unit_jit(function)
    with expect_execution(wrapped, "fallback"):
        result = wrapped(1 * ureg.m)
    assert result == 1.02 * ureg.m


def test_explicit_operands_stay_fast_despite_unused_quantity_default():
    wrapped = unit_jit(explicit_caller)
    with expect_execution(wrapped):
        result = wrapped(1 * ureg.m, 2 * ureg.cm)
    assert result == 1.02 * ureg.m


def helper(x, y=2):
    return x * y


def keyword_numeric_helper(x, *, y=2):
    return x * y


def caller(x):
    return helper(x)


def keyword_numeric_caller(x):
    return keyword_numeric_helper(x)


def replacement(x, y=2):
    return x / x


@pytest.mark.parametrize("change", ["code", "default", "keyword_default"])
@pytest.mark.parametrize("direct", [False, True])
def test_callable_changes_invalidate_cached_semantics(monkeypatch, change, direct):
    target = keyword_numeric_helper if change == "keyword_default" else helper
    function = (
        target if direct else (keyword_numeric_caller if change == "keyword_default" else caller)
    )
    wrapped = unit_jit(function)
    with expect_execution(wrapped):
        assert wrapped(1 * ureg.m) == 2 * ureg.m
    with monkeypatch.context() as patch:
        if change == "code":
            patch.setattr(target, "__code__", replacement.__code__)
        elif change == "default":
            patch.setattr(target, "__defaults__", (2 * ureg.s,))
        else:
            patch.setitem(target.__kwdefaults__, "y", 2 * ureg.s)
        with expect_execution(wrapped, "fallback"):
            result = wrapped(1 * ureg.m)
        assert result == function(1 * ureg.m)
    with expect_execution(wrapped):
        assert wrapped(1 * ureg.m) == 2 * ureg.m


@dataclass
class Box:
    x: object


def read(x):
    return x.x


def indirect_read(x):
    return read(x)


def magnitude_getter(self, name):
    result = object.__getattribute__(self, name)
    return result.magnitude if name == "x" else result


@pytest.mark.parametrize("function", [read, indirect_read])
@pytest.mark.parametrize("warm", [False, True])
def test_object_access_is_verified_before_stripping(monkeypatch, function, warm):
    box = Box(2 * ureg.cm)
    wrapped = unit_jit(function)
    if warm:
        with expect_execution(wrapped):
            assert wrapped(box) == 2 * ureg.cm
    with monkeypatch.context() as patch:
        patch.setattr(Box, "__getattribute__", magnitude_getter)
        expected = function(box)
        with expect_execution(wrapped, "fallback"):
            assert wrapped(box) == expected
        assert vars(box)["x"] == 2 * ureg.cm
    if warm:
        with expect_execution(wrapped):
            assert wrapped(box) == 2 * ureg.cm


class Scale(float):
    def __mul__(self, other):
        return 7 * ureg.s


scale = Scale(2)


def global_scale(x):
    return scale * x


def explicit_scale(x, scale):
    return scale * x


@pytest.mark.parametrize("global_value", [False, True])
def test_custom_numbers_have_one_admission_policy(global_value):
    function = global_scale if global_value else explicit_scale
    args = (1 * ureg.m,) if global_value else (1 * ureg.m, scale)
    wrapped = unit_jit(function)
    with expect_execution(wrapped, "fallback"):
        assert wrapped(*args) == 7 * ureg.s


def equal(a, b):
    return a == b


def unequal(a, b):
    return a != b


def less(a, b):
    return a < b


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("function", [equal, unequal, less])
@pytest.mark.parametrize("array", [False, True])
def test_container_comparisons_preserve_implicit_identity(container, function, array):
    quantity = ureg.Quantity(np.array([1.0, 2.0]) if array else float("nan"), "cm")
    a, b = container([quantity]), container([quantity])
    expected = function(a, b)
    wrapped = unit_jit(function)
    with expect_execution(wrapped, "fallback"):
        assert wrapped(a, b) == expected


def test_quantity_scalar_comparison_remains_fast():
    wrapped = unit_jit(equal)
    with expect_execution(wrapped):
        assert wrapped(1 * ureg.m, 100 * ureg.cm)


# Partial application is another source of operands absent from the call syntax.
# It must obey the same preparation and dependency rules as omitted defaults.

partial_helper = functools.partial(helper, y=2)


def partial_caller(x):
    return partial_helper(x)


@pytest.mark.parametrize("warm", [False, True])
def test_bound_operands_share_default_admission_rules(monkeypatch, warm):
    wrapped = unit_jit(partial_caller)
    if warm:
        with expect_execution(wrapped):
            assert wrapped(1 * ureg.m) == 2 * ureg.m
    with monkeypatch.context() as patch:
        patch.setitem(partial_helper.keywords, "y", 2 * ureg.s)
        with expect_execution(wrapped, "fallback"):
            assert wrapped(1 * ureg.m) == 2 * ureg.m * ureg.s
    if warm:
        with expect_execution(wrapped):
            assert wrapped(1 * ureg.m) == 2 * ureg.m
