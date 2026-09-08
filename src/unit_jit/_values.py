"""Abstract values and unit algebra shared by inference and operation specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import libcst as cst


class Atom(Enum):
    PLAIN = auto()
    UNKNOWN = auto()
    NO_RETURN = auto()


PLAIN = Atom.PLAIN
_UNKNOWN = Atom.UNKNOWN
_SENTINEL = Atom.NO_RETURN


@dataclass(frozen=True)
class QuantityValue:
    unit: Any

    @property
    def dimensionality(self):
        return self.unit.dimensionality

    @property
    def _REGISTRY(self):
        return self.unit._REGISTRY

    def __str__(self):
        return str(self.unit)


class ObjectValue(dict[str, "AbstractValue"]):
    """Abstract fields of an admitted object; distinct from interpreter environments."""


class _Unsupported(Exception):
    """The original computation must run before any quantities are stripped."""


def _same_unit(
    left: Any, right: Any, seen: tuple[dict[int, int], dict[int, int]] | None = None
) -> bool:
    if isinstance(left, (dict, SequenceValue)) and isinstance(right, type(left)):
        forward, reverse = seen if seen is not None else ({}, {})
        a, b = id(left), id(right)
        if a in forward or b in reverse:
            return forward.get(a) == b and reverse.get(b) == a
        forward[a], reverse[b] = b, a
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(
                _same_unit(left[k], right[k], (forward, reverse)) for k in left
            )
        return (
            left.kind == right.kind
            and left.cls is right.cls
            and left.repeated == right.repeated
            and len(left.units) == len(right.units)
            and all(_same_unit(a, b, (forward, reverse)) for a, b in zip(left.units, right.units))
        )
    if left is right:
        return True
    if isinstance(left, QuantityValue) and isinstance(right, QuantityValue):
        return left._REGISTRY is right._REGISTRY and left.dimensionality == right.dimensionality
    return False


def _join(left: Any, right: Any) -> Any:
    """Join reachable values without confusing unknown and dimensionless."""
    if left is _SENTINEL:
        return right
    if right is _SENTINEL:
        return left
    if isinstance(left, (dict, SequenceValue)) and left is not right:
        # Choosing one of two mutable values would invent an alias relationship.
        return _UNKNOWN
    return left if _same_unit(left, right) else _UNKNOWN


def _known(unit: Any) -> AbstractValue:
    if not isinstance(unit, (Atom, QuantityValue, ObjectValue, SequenceValue, LambdaValue)):
        raise _Unsupported("expression did not produce an abstract value")
    if unit is _UNKNOWN:
        raise _Unsupported("unit could not be established")
    if isinstance(unit, SequenceValue):
        for element in unit.units:
            _known(element)
    return unit


def _copy_env(env: dict[str, Any]) -> dict[str, Any]:
    """Copy abstract containers, preserving aliases and leaving registries alone."""
    memo: dict[int, Any] = {}

    def copy(value: Any) -> Any:
        if id(value) in memo:
            return memo[id(value)]
        if isinstance(value, dict):
            result: Any = type(value)()
            memo[id(value)] = result
            result.update({k: copy(v) for k, v in value.items()})
            return result
        if isinstance(value, SequenceValue):
            result = SequenceValue(
                value.kind, [], value.cls, repeated=value.repeated, external=value.external
            )
            memo[id(value)] = result
            result.units = [copy(v) for v in value.units]
            return result
        return value

    return copy(env)


@dataclass(slots=True, eq=False)
class SequenceValue:
    """Abstract elements, shape and alias policy of a supported sequence."""

    kind: str
    units: list[AbstractValue]
    cls: type | None = None
    repeated: bool = field(default=False, kw_only=True)
    external: bool = field(default=False, kw_only=True)

    def __eq__(self, other: object) -> bool:
        return _same_unit(self, other)


@dataclass(slots=True)
class LambdaValue:
    node: cst.Lambda


# ---------------------------------------------------------------------------
# Unit arithmetic helpers
# ---------------------------------------------------------------------------


def _unit_mul(u1: AbstractValue, u2: AbstractValue) -> AbstractValue:
    if u1 is _UNKNOWN or u2 is _UNKNOWN:
        return _UNKNOWN
    if u1 is PLAIN:
        return u2
    if u2 is PLAIN:
        return u1
    try:
        return QuantityValue(u1.unit * u2.unit)
    except Exception:  # noqa: BLE001 — Pint raises UndefinedUnitError, DimensionalityError, etc.
        return _UNKNOWN


def _unit_div(u1: AbstractValue, u2: AbstractValue) -> AbstractValue:
    if u1 is _UNKNOWN or u2 is _UNKNOWN:
        return _UNKNOWN
    if u1 is PLAIN and u2 is PLAIN:
        return PLAIN
    if u2 is PLAIN:
        return u1
    try:
        if u1 is PLAIN:
            reg = u2._REGISTRY  # noqa: SLF001
            return QuantityValue((reg.Quantity(1) / reg.Quantity(1, u2.unit)).to_base_units().units)
        return QuantityValue(u1.unit / u2.unit)
    except Exception:  # noqa: BLE001 — Pint raises UndefinedUnitError, DimensionalityError, etc.
        return _UNKNOWN


def _unit_pow(u: AbstractValue, exp: float) -> AbstractValue:
    if u is _UNKNOWN:
        return _UNKNOWN
    if u is PLAIN:
        return PLAIN
    try:
        return QuantityValue(u.unit**exp)
    except Exception:  # noqa: BLE001 — Pint raises UndefinedUnitError, DimensionalityError, etc.
        return _UNKNOWN


type AbstractValue = Atom | QuantityValue | ObjectValue | SequenceValue | LambdaValue


def contains_quantity(value: AbstractValue) -> bool:
    return isinstance(value, QuantityValue) or (
        isinstance(value, SequenceValue) and any(contains_quantity(item) for item in value.units)
    )
