"""Canonical, value-independent schemas for the outer JIT boundary."""

from typing import Any

import numpy as np

from ._admission import native_scalar, plain_storage
from ._dispatch import instance_overrides
from ._inferrer import _QUANTITY_TYPES, _Unsupported


def argument_schema(arguments: dict[str, Any]) -> dict[str, Any]:
    """Describe dimensions, representations and aliases after signature binding.

    Quantity magnitudes and unit scales are deliberately absent. Homogeneous
    sequences may change length; heterogeneous sequences retain positional units.
    Graph references use traversal indices, never the identity of mutable inputs.
    """
    seen: dict[int, int] = {}
    registries: set[int] = set()

    def visit(value: Any) -> Any:
        if isinstance(value, _QUANTITY_TYPES):
            if not getattr(value, "_is_multiplicative", True):
                raise _Unsupported("offset and logarithmic units require Pint")
            registries.add(id(value._REGISTRY))
            return (
                "quantity",
                id(value._REGISTRY),
                tuple(sorted(value.dimensionality.items())),
                isinstance(value.magnitude, np.ndarray),
                type(value),
                type(value.magnitude) if isinstance(value.magnitude, np.ndarray) else None,
                instance_overrides(value),
            )
        if native_scalar(value):
            return (
                ("number",)
                if not isinstance(value, (str, bytes, type(None)))
                else ("plain", type(value))
            )
        if isinstance(value, (np.ndarray, np.random.Generator)):
            if isinstance(value, np.ndarray) and (
                type(value) is not np.ndarray or value.dtype.hasobject
            ):
                raise _Unsupported("object arrays may contain quantities")
            return ("opaque", type(value))
        if callable(value):
            return ("callable", id(value))
        identity = id(value)
        if identity in seen:
            return ("ref", seen[identity])
        seen[identity] = len(seen)
        if isinstance(value, (list, tuple)):
            if type(value) not in (list, tuple) and not (
                isinstance(value, tuple)
                and hasattr(type(value), "_fields")
                and type(value).__getitem__ is tuple.__getitem__
                and type(value).__iter__ is tuple.__iter__
            ):
                raise _Unsupported("custom sequence protocols require original execution")
            elements = tuple(visit(item) for item in value)
            # Collapse only leaf schemas: collapsing graphs can erase aliasing.
            if elements and all(item == elements[0] for item in elements):
                if elements[0][0] in {"quantity", "number", "plain", "opaque", "callable"}:
                    if not hasattr(type(value), "_fields"):
                        return ("sequence", type(value), elements[0])
            return ("sequence-fixed", type(value), elements)
        if isinstance(value, dict):
            raise _Unsupported("mapping conversion is not supported")
        if hasattr(value, "__dict__"):
            if not plain_storage(value):
                raise _Unsupported("custom attribute protocols require original execution")
            return (
                "object",
                type(value),
                tuple((name, visit(item)) for name, item in sorted(vars(value).items())),
            )
        raise _Unsupported(f"unsupported argument type: {type(value).__name__}")

    schema = {name: visit(value) for name, value in arguments.items()}
    if len(registries) > 1:
        raise _Unsupported("mixed registries require Pint")
    return schema


def check_schema(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    """Reject dimensional changes; a different implementation requires Pint dispatch."""

    def check(left: Any, right: Any) -> bool:
        if left == right:
            return True
        if left[0] == right[0] == "quantity" and left[:3] == right[:3]:
            return False
        if left[0].startswith("sequence") and right[0].startswith("sequence"):
            if left[1] is right[1]:
                if left[2] == () or right[2] == ():
                    return False
                if left[0] == right[0] == "sequence":
                    return check(left[2], right[2])
                if left[0] == "sequence":
                    return all([check(left[2], item) for item in right[2]])
                if right[0] == "sequence":
                    return all([check(item, right[2]) for item in left[2]])
                if len(left[2]) == len(right[2]):
                    return all([check(a, b) for a, b in zip(left[2], right[2])])
        if left[0] == right[0] == "object":
            if left[1] is not right[1]:
                return False
            old, new = dict(left[2]), dict(right[2])
            changed = old.keys() ^ new.keys()
            if changed and all((old.get(name) or new[name])[0] == "callable" for name in changed):
                for name in old.keys() & new.keys():
                    check(old[name], new[name])
                return False
            if old.keys() == new.keys():
                results = [check(old[name], new[name]) for name in old]
                return all(results)
        if left[0] == right[0] == "callable":
            return False
        raise TypeError("argument dimensions or structure differ from the first compiled call")

    if expected.keys() != actual.keys():
        raise TypeError("bound parameters differ from the first compiled call")
    return all([check(expected[name], actual[name]) for name in expected])
