"""Shared representation and callable assumptions for fast-path admission."""

import functools
import inspect
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

_NATIVE_NUMBERS = frozenset((bool, int, float, complex)) | frozenset(
    cls for cls in np.sctypeDict.values() if issubclass(cls, (np.number, np.bool_))
)


def native_scalar(value):
    return type(value) in _NATIVE_NUMBERS or type(value) in (str, bytes, type(None))


def immutable_default(value):
    return native_scalar(value) or (
        type(value) is tuple and all(immutable_default(item) for item in value)
    )


def plain_storage(value):
    """Dictionary fields describe reads/writes only under ordinary object access."""
    cls = type(value)
    for name in ("__getattribute__", "__setattr__", "__delattr__"):
        actual = inspect.getattr_static(cls, name, None)
        if actual not in (
            inspect.getattr_static(object, name),
            inspect.getattr_static(SimpleNamespace, name),
        ):
            return False
    if inspect.getattr_static(cls, "__getattr__", None) is not None:
        return False
    return all(
        not inspect.isdatadescriptor(inspect.getattr_static(cls, name, None))
        for name in vars(value)
    )


def callable_state(function):
    # Retain referents, rather than just their ids, to prevent identity reuse.
    if isinstance(function, functools.partial):
        return (function.func, function.args, tuple(sorted(function.keywords.items())))
    return (
        function.__code__,
        function.__defaults__,
        tuple(sorted((function.__kwdefaults__ or {}).items())),
    )


@dataclass(frozen=True)
class CallableBinding:
    function: object
    state: tuple

    @classmethod
    def capture(cls, function):
        return cls(function, callable_state(function))

    def unchanged(self):
        code, defaults, keywords = callable_state(self.function)
        old_code, old_defaults, old_keywords = self.state
        return (
            code is old_code
            and defaults is old_defaults
            and len(keywords) == len(old_keywords)
            and all(
                k == old_k and v is old_v for (k, v), (old_k, old_v) in zip(keywords, old_keywords)
            )
        )
