"""Implementation guards for quantity methods executed on stripped magnitudes."""

import inspect
from dataclasses import dataclass

import numpy as np

from ._calls import METHODS

_MISSING = object()
DISPATCH_HOOKS = (
    "__getattr__",
    "__getattribute__",
    "__array_function__",
    "__array_ufunc__",
    "_numpy_method_wrap",
    "to_base_units",
)
# Every operation on a stripped quantity relies on these protocols retaining
# backend semantics. Guard the whole protocol at plan admission, independently
# of which syntax or library entry point eventually invokes it.
OPERATORS = (
    "add",
    "sub",
    "mul",
    "truediv",
    "floordiv",
    "mod",
    "pow",
    "matmul",
    "and",
    "or",
    "xor",
    "lshift",
    "rshift",
)
PROTOCOL_NAMES = frozenset(
    (
        *DISPATCH_HOOKS,
        *(f"__{prefix}{name}__" for name in OPERATORS for prefix in ("", "r", "i")),
        *(
            f"__{name}__"
            for name in (
                "pos",
                "neg",
                "abs",
                "invert",
                "bool",
                "int",
                "float",
                "index",
                "eq",
                "ne",
                "lt",
                "le",
                "gt",
                "ge",
                "getitem",
                "setitem",
                "iter",
                "len",
                "contains",
                "array",
            )
        ),
    )
)
DISPATCH_NAMES = frozenset((*METHODS, *PROTOCOL_NAMES))


def instance_overrides(value):
    """Quantity schemas must distinguish per-instance dispatch overrides."""
    return (
        tuple(
            (name, id(item)) for name, item in sorted(vars(value).items()) if name in DISPATCH_NAMES
        )
        if hasattr(value, "__dict__")
        else ()
    )


@dataclass(frozen=True)
class DispatchBinding:
    owner: type
    implementations: tuple[tuple[str, object], ...]

    def unchanged(self):
        # Resolve all descriptors in one MRO walk. Repeating getattr_static for
        # every operator at every small-function entry is prohibitively costly.
        descriptors = {}
        for base in reversed(type.__getattribute__(type(self.owner), "__mro__")):
            descriptors.update(type.__getattribute__(base, "__dict__"))
        for base in reversed(type.__getattribute__(self.owner, "__mro__")):
            descriptors.update(type.__getattribute__(base, "__dict__"))
        return all(
            descriptors.get(name, _MISSING) is expected for name, expected in self.implementations
        )


class QuantityDispatch:
    def __init__(self, quantity_types):
        # Capture implementations once, not after a user's override on first call.
        self.baselines = {
            cls: {name: inspect.getattr_static(cls, name, _MISSING) for name in DISPATCH_NAMES}
            for cls in quantity_types
        }
        self.ndarray_methods = {
            name: inspect.getattr_static(np.ndarray, name, _MISSING) for name in METHODS
        }

    def guard(self, name, sources):
        keys = (name, *DISPATCH_HOOKS) if name is not None else PROTOCOL_NAMES
        bindings = {}
        families = set()

        def require(cls, key, expected):
            if inspect.getattr_static(cls, key, _MISSING) is not expected:
                raise ValueError("quantity method implementation has been overridden")
            bindings[cls, key] = expected

        for source in sources:
            cls = type(source)
            # Use real MRO membership, not facade metaclass instance/subclass hooks.
            base = next(
                (candidate for candidate in cls.__mro__ if candidate in self.baselines), None
            )
            if base is None or instance_overrides(source):
                raise ValueError("quantity receiver has unverified method dispatch")
            families.add(base.__module__.split(".")[0])
            for key in keys:
                require(cls, key, self.baselines[base][key])
            if name is not None and (
                not callable(getattr(source, name, None)) or self.ndarray_methods[name] is _MISSING
            ):
                raise ValueError("method is unavailable on the backend or stripped array")
            magnitude = source.magnitude
            if isinstance(magnitude, np.ndarray):
                if name is None:
                    if type(magnitude) is not np.ndarray:
                        raise ValueError("custom magnitude arrays require backend execution")
                else:
                    require(type(magnitude), name, self.ndarray_methods[name])

        if not families:
            raise ValueError("quantity method lacks concrete backend provenance")
        # Arithmetic may produce another concrete quantity class in the same
        # backend (e.g. pintrs scalar -> array). Guard those implementations too.
        for cls, expected in self.baselines.items():
            if cls.__module__.split(".")[0] in families:
                for key in keys:
                    require(cls, key, expected[key])
        grouped = {}
        for (cls, key), expected in bindings.items():
            grouped.setdefault(cls, []).append((key, expected))
        return [DispatchBinding(cls, tuple(items)) for cls, items in grouped.items()]
