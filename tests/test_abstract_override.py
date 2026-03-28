"""Tests for lazy inference of overridden abstract methods and __init_subclass__ auto-compile.

Two related features tested here:

Feature A — abstract method override lazy inference:
  A @unit_jit base method calls self.abstract_method().  At inference time the
  abstract method on the base class has no body to analyse; the inferrer must
  look up the concrete override on type(self) and lazily infer its return unit.
  This works for both @unit_jit-decorated overrides (primary path) and plain
  undecorated overrides (fallback lazy-inference path).

Feature B — __init_subclass__ auto-compile:
  Mirrors the bcrnnoise BCRN pattern: __init_subclass__ wraps each concrete
  subclass __init__ to call unit_jit.compile(self) immediately after
  construction.  This pre-warms all @unit_jit methods on the instance so the
  first real simulation call pays zero lazy-inference overhead.
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from typing import cast

import unit_jit as _uj
from pint import Quantity, UnitRegistry
from unit_jit import compile as unit_jit_compile
from unit_jit import unit_jit

ureg = UnitRegistry()


# ---------------------------------------------------------------------------
# Feature A: abstract method override lazy inference
# ---------------------------------------------------------------------------


class _KineticBase(ABC):
    def __init__(self, volume: Quantity) -> None:
        self.volume = volume

    @abstractmethod
    def propensity(self, n: Quantity) -> Quantity: ...

    @unit_jit
    def total_rate(self, n: Quantity) -> Quantity:
        """Calls the abstract self.propensity() — unit inferred lazily from override."""
        return cast("Quantity", self.propensity(n) * self.volume)


# Override A: decorated with @unit_jit


class _ConcreteJit(_KineticBase):
    def __init__(self, alpha: Quantity, volume: Quantity) -> None:
        super().__init__(volume)
        self.alpha = alpha

    @unit_jit
    def propensity(self, n: Quantity) -> Quantity:
        return cast("Quantity", self.alpha * n)


# Override B: plain (undecorated) method — exercises the fallback path


class _ConcretePlain(_KineticBase):
    def __init__(self, alpha: Quantity, volume: Quantity) -> None:
        super().__init__(volume)
        self.alpha = alpha

    def propensity(self, n: Quantity) -> Quantity:
        return cast("Quantity", self.alpha * n)


def _make_sys(cls: type, alpha_val: float = 2.0, volume_val: float = 1.0) -> _KineticBase:
    return cls(
        alpha=alpha_val / ureg.s / (ureg.mol / ureg.L),
        volume=volume_val * ureg.L,
    )


# results


def test_jit_override_result():
    sys = _make_sys(_ConcreteJit)
    n = 3.0 * ureg.mol / ureg.L
    result = sys.total_rate(n)
    expected = sys.propensity(n) * sys.volume
    assert abs(result.to_base_units().magnitude - expected.to_base_units().magnitude) < 1e-10


def test_plain_override_result():
    sys = _make_sys(_ConcretePlain)
    n = 3.0 * ureg.mol / ureg.L
    result = sys.total_rate(n)
    expected = sys.propensity(n) * sys.volume
    assert abs(result.to_base_units().magnitude - expected.to_base_units().magnitude) < 1e-10


# both overrides give the same answer as the pure-pint baseline


def test_jit_override_matches_plain():
    n = 2.0 * ureg.mol / ureg.L
    jit_sys = _make_sys(_ConcreteJit)
    plain_sys = _make_sys(_ConcretePlain)
    r_jit = jit_sys.total_rate(n).to_base_units().magnitude
    r_plain = plain_sys.total_rate(n).to_base_units().magnitude
    assert abs(r_jit - r_plain) < 1e-10


# units


def test_jit_override_returns_quantity():
    sys = _make_sys(_ConcreteJit)
    assert isinstance(sys.total_rate(1.0 * ureg.mol / ureg.L), Quantity)


def test_plain_override_returns_quantity():
    sys = _make_sys(_ConcretePlain)
    assert isinstance(sys.total_rate(1.0 * ureg.mol / ureg.L), Quantity)


def test_jit_override_dimension():
    # (L/(mol·s)) * (mol/L) * L  =  L/s  →  [length]^3 [time]^-1
    sys = _make_sys(_ConcreteJit)
    result = sys.total_rate(1.0 * ureg.mol / ureg.L)
    assert result.dimensionality == {"[length]": 3, "[time]": -1}


def test_plain_override_dimension():
    sys = _make_sys(_ConcretePlain)
    result = sys.total_rate(1.0 * ureg.mol / ureg.L)
    assert result.dimensionality == {"[length]": 3, "[time]": -1}


# JIT active after first call


def test_jit_override_jit_active():
    sys = _make_sys(_ConcreteJit)
    sys.total_rate(1.0 * ureg.mol / ureg.L)
    qualname = sys.total_rate.__qualname__
    assert qualname in _uj._return_units
    assert qualname not in _uj._jit_disabled


# ---------------------------------------------------------------------------
# Feature B: __init_subclass__ auto-compile
# ---------------------------------------------------------------------------


class _AutoCompileBase:
    """Mirrors bcrnnoise.BCRN.__init_subclass__: compile(self) after construction."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if "__init__" in cls.__dict__ and not getattr(cls, "__abstractmethods__", None):
            orig_init = cls.__dict__["__init__"]

            @functools.wraps(orig_init)
            def _init_with_compile(self: "_AutoCompileBase", *args: object, **kw: object) -> None:
                orig_init(self, *args, **kw)  # type: ignore[misc]
                unit_jit_compile(self)

            cls.__init__ = _init_with_compile  # type: ignore[method-assign]

    @unit_jit
    def scale(self, x: Quantity) -> Quantity:
        """Simple method: uses only self.factor, which compile() can build a dummy for."""
        return cast("Quantity", x * self.factor)


class _AutoCompileConcrete(_AutoCompileBase):
    def __init__(self, factor: Quantity) -> None:
        self.factor = factor


def test_auto_compile_jit_active_before_first_call():
    """JIT pre-warmed by __init_subclass__: no call needed."""
    sys = _AutoCompileConcrete(factor=2.0 * ureg.s)
    qualname = sys.scale.__qualname__
    assert qualname in _uj._return_units, "scale() should be pre-compiled by __init_subclass__"
    assert qualname not in _uj._jit_disabled


def test_auto_compile_result_correct():
    sys = _AutoCompileConcrete(factor=3.0 * ureg.s)
    result = sys.scale(2.0 * ureg.s)
    assert isinstance(result, Quantity)
    assert abs(result.to("s^2").magnitude - 6.0) < 1e-12


def test_auto_compile_dimension():
    sys = _AutoCompileConcrete(factor=1.0 * ureg.s)
    result = sys.scale(1.0 * ureg.s)
    assert result.dimensionality == {"[time]": 2}
