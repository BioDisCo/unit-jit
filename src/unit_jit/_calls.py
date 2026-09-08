"""Operation specifications: binding, unit rules and trusted callable identities.

Only declared data operands feed a unit rule. Control operands must be plain;
every other supplied option requires original-backend execution. This includes
scale-dependent values, dtype conversions, output storage and callbacks.
"""

import builtins
import inspect
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._values import (
    _UNKNOWN,
    PLAIN,
    AbstractValue,
    QuantityValue,
    SequenceValue,
    _join,
    _unit_div,
    _unit_mul,
    _unit_pow,
    _Unsupported,
    contains_quantity,
)

_OPTIONAL = object()


def signature(required, optional=(), *, variadic=None, keywords=(), positional_only=()):
    params = [
        inspect.Parameter(
            n,
            inspect.Parameter.POSITIONAL_ONLY
            if n in positional_only
            else inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        for n in required
    ]
    params += [
        inspect.Parameter(
            n,
            inspect.Parameter.POSITIONAL_ONLY
            if n in positional_only
            else inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=_OPTIONAL,
        )
        for n in optional
    ]
    if variadic:
        params.append(inspect.Parameter(variadic, inspect.Parameter.VAR_POSITIONAL))
    params += [
        inspect.Parameter(n, inspect.Parameter.KEYWORD_ONLY, default=_OPTIONAL) for n in keywords
    ]
    return inspect.Signature(params)


@dataclass(frozen=True)
class CallContract:
    signature: inspect.Signature
    data: tuple[str, ...]
    controls: frozenset[str] = frozenset()
    allow_sequences: bool = False

    def bind(self, arguments):
        positional, keywords = [], {}
        for name, unit in arguments:
            if name is None:
                positional.append(unit)
            else:
                if name in keywords:
                    raise ValueError("duplicate keyword")
                keywords[name] = unit
        bound = self.signature.bind(*positional, **keywords)
        for name, unit in bound.arguments.items():
            if name in self.data:
                continue
            values = (
                unit
                if self.signature.parameters[name].kind is inspect.Parameter.VAR_POSITIONAL
                else (unit,)
            )
            if name not in self.controls or any(value is not PLAIN for value in values):
                raise ValueError(f"argument {name} requires original-backend semantics")
        data = []
        for name in self.data:
            if name not in bound.arguments:
                continue
            value = bound.arguments[name]
            if self.signature.parameters[name].kind is inspect.Parameter.VAR_POSITIONAL:
                data.extend(value)
            else:
                data.append(value)
        return data


def contract(
    required,
    optional=(),
    *,
    controls=(),
    variadic=None,
    keywords=(),
    data=None,
    allow_sequences=False,
    positional_only=(),
):
    return CallContract(
        signature(
            required,
            optional,
            variadic=variadic,
            keywords=keywords,
            positional_only=positional_only,
        ),
        tuple(required) if data is None else tuple(data),
        frozenset(controls),
        allow_sequences,
    )


_log = logging.getLogger("unit_jit")


def _p(us: list[AbstractValue]) -> AbstractValue:
    return us[0] if us else PLAIN  # preserve first arg unit


def _d(_: list[AbstractValue]) -> AbstractValue:
    return PLAIN  # dimensionless


def _dimensionless_in(us: list[AbstractValue]) -> AbstractValue:
    """Transcendental functions: argument must be dimensionless, result is dimensionless.

    Unlike _d, this verifies the argument carries no dimensions (e.g. catches
    math.exp(length / time)), then reports a dimensionless result.
    """
    for u in us:
        if isinstance(u, QuantityValue) and u.dimensionality:
            msg = f"transcendental function requires a dimensionless argument, got '{u}'"
            _log.warning("dimension mismatch: %s", msg)
            raise TypeError(msg)
    return PLAIN


def _numpy_dimensionless_in(us: list[AbstractValue]) -> AbstractValue:
    _dimensionless_in(us)
    return (
        QuantityValue(us[0]._REGISTRY.dimensionless)
        if us and isinstance(us[0], QuantityValue)
        else PLAIN
    )


def _sqrt(us: list[AbstractValue]) -> AbstractValue:
    return _unit_pow(us[0], 0.5) if us else PLAIN


def _sq(us: list[AbstractValue]) -> AbstractValue:
    return _unit_pow(us[0], 2) if us else PLAIN


def _cbrt(us: list[AbstractValue]) -> AbstractValue:
    return _unit_pow(us[0], 1 / 3) if us else PLAIN


def _mul2(us: list[AbstractValue]) -> AbstractValue:
    return _unit_mul(us[0], us[1]) if len(us) > 1 else _p(us)


def _reduce(us: list[AbstractValue]) -> AbstractValue:
    """Return element unit when reducing a list (sum/min/max), else first arg unit."""
    result = us[0] if us else PLAIN
    if isinstance(result, SequenceValue):
        # sum/min/max collapses a list to its common element unit
        if not result.units:
            return PLAIN
        unit = result.units[0]
        for element in result.units[1:]:
            unit = _join(unit, element)
        # Local sequences can become empty at runtime. Nonempty input sequences
        # are guarded at the boundary; an empty input uses per-call fallback.
        if result.repeated and not result.external and unit is not PLAIN:
            return _UNKNOWN
        return unit
    return result


def _select(us: list[AbstractValue]) -> AbstractValue:
    if len(us) <= 1:
        return _reduce(us)
    result = us[0]
    for operand in us[1:]:
        result = _join(result, operand)
    return result


def resolve_builtin(name):
    parts = name.split(".")
    value = {"np": np, "math": math}.get(parts[0], getattr(builtins, parts[0], None))
    for part in parts[1:]:
        value = getattr(value, part, None)
    return value


@dataclass(frozen=True)
class Operation:
    rule: Callable[[list[AbstractValue]], AbstractValue]
    contract: CallContract | None = None
    implementation: Any = None

    def infer(self, arguments: list[tuple[str | None, AbstractValue]]) -> AbstractValue:
        """Bind admitted operands and apply the operation's unit rule."""
        values = [value for _, value in arguments]
        quantity_call = any(contains_quantity(value) for value in values)
        if quantity_call:
            if self.contract is None:
                raise _Unsupported("no verified quantity call contract")
            try:
                operands = self.contract.bind(arguments)
            except (TypeError, ValueError) as exc:
                raise _Unsupported(str(exc)) from exc
            if not self.contract.allow_sequences and any(
                isinstance(value, SequenceValue) for value in operands
            ):
                raise _Unsupported("quantity sequence coercion requires the original backend")
        else:
            operands = [
                PLAIN if isinstance(value, SequenceValue) else value
                for name, value in arguments
                if name is None
            ]
        if self.rule is _d and any(value is not PLAIN for value in values):
            raise _Unsupported("operation requires plain operands")
        if (
            not quantity_call
            and self.rule is _p
            and any(value is not PLAIN for value in values[1:])
        ):
            raise _Unsupported("additional operands require an explicit unit rule")
        try:
            return self.rule(operands)
        except TypeError:
            raise
        except Exception:
            return _UNKNOWN


def numpy_arguments(name, controls):
    return CallContract(inspect.signature(resolve_builtin(name)), ("a",), frozenset(controls))


_UFUNC_ARGUMENTS = contract(
    ("x",),
    ("out",),
    controls=("order", "subok", "where"),
    keywords=("where", "casting", "order", "dtype", "subok", "signature"),
)

# A missing contract explicitly restricts an operation to plain operands.
FUNCTIONS = {
    "abs": Operation(_p, contract(("x",), positional_only=("x",))),
    "round": Operation(_p),
    "float": Operation(
        _dimensionless_in, contract((), ("x",), positional_only=("x",), data=("x",))
    ),
    "int": Operation(_dimensionless_in, contract(("x",))),
    "bool": Operation(_d),
    "len": Operation(_d),
    "sum": Operation(
        _reduce,
        contract(("iterable",), ("start",), positional_only=("iterable",), allow_sequences=True),
    ),
    "min": Operation(
        _select,
        contract(
            (),
            variadic="values",
            keywords=("key", "default"),
            data=("values",),
            allow_sequences=True,
        ),
    ),
    "max": Operation(
        _select,
        contract(
            (),
            variadic="values",
            keywords=("key", "default"),
            data=("values",),
            allow_sequences=True,
        ),
    ),
    "math.sqrt": Operation(_dimensionless_in, contract(("x",))),
    "math.exp": Operation(_dimensionless_in, contract(("x",))),
    "math.log": Operation(_dimensionless_in, contract(("x",))),
    "math.log2": Operation(_dimensionless_in, contract(("x",))),
    "math.log10": Operation(_dimensionless_in, contract(("x",))),
    "math.log1p": Operation(_dimensionless_in, contract(("x",))),
    "math.expm1": Operation(_dimensionless_in, contract(("x",))),
    "math.sin": Operation(_dimensionless_in, contract(("x",))),
    "math.cos": Operation(_dimensionless_in, contract(("x",))),
    "math.tan": Operation(_dimensionless_in, contract(("x",))),
    "math.asin": Operation(_dimensionless_in, contract(("x",))),
    "math.acos": Operation(_dimensionless_in, contract(("x",))),
    "math.atan": Operation(_dimensionless_in, contract(("x",))),
    "math.atan2": Operation(_d),
    "math.sinh": Operation(_dimensionless_in, contract(("x",))),
    "math.cosh": Operation(_dimensionless_in, contract(("x",))),
    "math.tanh": Operation(_dimensionless_in, contract(("x",))),
    "math.fabs": Operation(_p),
    "math.hypot": Operation(_p),
    "math.floor": Operation(_d),
    "math.ceil": Operation(_d),
    "math.trunc": Operation(_d),
    "np.exp": Operation(_numpy_dimensionless_in, _UFUNC_ARGUMENTS),
    "np.expm1": Operation(_numpy_dimensionless_in, _UFUNC_ARGUMENTS),
    "np.log": Operation(_numpy_dimensionless_in, _UFUNC_ARGUMENTS),
    "np.log2": Operation(_numpy_dimensionless_in, _UFUNC_ARGUMENTS),
    "np.log10": Operation(_numpy_dimensionless_in, _UFUNC_ARGUMENTS),
    "np.log1p": Operation(_numpy_dimensionless_in, _UFUNC_ARGUMENTS),
    "np.sin": Operation(_numpy_dimensionless_in, _UFUNC_ARGUMENTS),
    "np.cos": Operation(_numpy_dimensionless_in, _UFUNC_ARGUMENTS),
    "np.tan": Operation(_numpy_dimensionless_in, _UFUNC_ARGUMENTS),
    "np.arcsin": Operation(_numpy_dimensionless_in, _UFUNC_ARGUMENTS),
    "np.arccos": Operation(_numpy_dimensionless_in, _UFUNC_ARGUMENTS),
    "np.arctan": Operation(_numpy_dimensionless_in, _UFUNC_ARGUMENTS),
    "np.arctan2": Operation(_d),
    "np.sinh": Operation(_numpy_dimensionless_in, _UFUNC_ARGUMENTS),
    "np.cosh": Operation(_numpy_dimensionless_in, _UFUNC_ARGUMENTS),
    "np.tanh": Operation(_numpy_dimensionless_in, _UFUNC_ARGUMENTS),
    "np.arcsinh": Operation(_numpy_dimensionless_in, _UFUNC_ARGUMENTS),
    "np.arccosh": Operation(_numpy_dimensionless_in, _UFUNC_ARGUMENTS),
    "np.arctanh": Operation(_numpy_dimensionless_in, _UFUNC_ARGUMENTS),
    "np.sign": Operation(_d),
    "np.isfinite": Operation(_d),
    "np.isnan": Operation(_d),
    "np.isinf": Operation(_d),
    "np.isreal": Operation(_d),
    "np.iscomplex": Operation(_d),
    "np.any": Operation(_d),
    "np.all": Operation(_d),
    "np.argmin": Operation(_d),
    "np.argmax": Operation(_d),
    "np.argsort": Operation(_d),
    "np.nonzero": Operation(_d),
    "np.linspace": Operation(_d),
    "np.arange": Operation(_d),
    "np.logspace": Operation(_d),
    "np.geomspace": Operation(_d),
    "np.abs": Operation(_p, _UFUNC_ARGUMENTS),
    "np.fabs": Operation(_p, _UFUNC_ARGUMENTS),
    "np.sum": Operation(_p, numpy_arguments("np.sum", ("axis", "keepdims", "where"))),
    "np.nansum": Operation(_p, numpy_arguments("np.nansum", ("axis", "keepdims", "where"))),
    "np.min": Operation(_p, numpy_arguments("np.min", ("axis", "keepdims", "where"))),
    "np.nanmin": Operation(_p, numpy_arguments("np.nanmin", ("axis", "keepdims", "where"))),
    "np.max": Operation(_p, numpy_arguments("np.max", ("axis", "keepdims", "where"))),
    "np.nanmax": Operation(_p, numpy_arguments("np.nanmax", ("axis", "keepdims", "where"))),
    "np.mean": Operation(_p, numpy_arguments("np.mean", ("axis", "keepdims", "where"))),
    "np.nanmean": Operation(_p, numpy_arguments("np.nanmean", ("axis", "keepdims", "where"))),
    "np.std": Operation(
        _p, numpy_arguments("np.std", ("axis", "correction", "ddof", "keepdims", "where"))
    ),
    "np.nanstd": Operation(
        _p, numpy_arguments("np.nanstd", ("axis", "correction", "ddof", "keepdims", "where"))
    ),
    "np.median": Operation(_p, numpy_arguments("np.median", ("axis", "keepdims"))),
    "np.nanmedian": Operation(_p, numpy_arguments("np.nanmedian", ("axis", "keepdims"))),
    "np.cumsum": Operation(_p, numpy_arguments("np.cumsum", ("axis",))),
    "np.diff": Operation(_p, numpy_arguments("np.diff", ("axis", "n"))),
    "np.clip": Operation(_p),
    "np.sort": Operation(_p),
    "np.real": Operation(_p),
    "np.imag": Operation(_p),
    "np.conj": Operation(_p),
    "np.floor": Operation(_p),
    "np.ceil": Operation(_p),
    "np.around": Operation(_p),
    "np.round_": Operation(_p),
    "np.trunc": Operation(_p),
    "np.fix": Operation(_p),
    "np.roll": Operation(_p),
    "np.flip": Operation(_p),
    "np.rot90": Operation(_p),
    "np.tile": Operation(_p),
    "np.repeat": Operation(_p),
    "np.squeeze": Operation(_p),
    "np.expand_dims": Operation(_p),
    "np.transpose": Operation(_p),
    "np.reshape": Operation(_p),
    "np.ravel": Operation(_p),
    "np.broadcast_to": Operation(_p),
    "np.pad": Operation(_p),
    "np.append": Operation(_p),
    "np.concatenate": Operation(_p),
    "np.stack": Operation(_p),
    "np.hstack": Operation(_p),
    "np.vstack": Operation(_p),
    "np.dstack": Operation(_p),
    "np.array": Operation(_p),
    "np.asarray": Operation(_p),
    "np.ascontiguousarray": Operation(_p),
    "np.copy": Operation(_p, numpy_arguments("np.copy", ("order", "subok"))),
    "np.hypot": Operation(_p),
    "np.linalg.norm": Operation(_p),
    "np.empty": Operation(_d),
    "np.zeros": Operation(_d),
    "np.ones": Operation(_d),
    "np.empty_like": Operation(_d),
    "np.zeros_like": Operation(_d),
    "np.ones_like": Operation(_d),
    "np.full": Operation(lambda us: us[1] if len(us) > 1 else PLAIN),
    "np.full_like": Operation(lambda us: us[1] if len(us) > 1 else PLAIN),
    "np.dot": Operation(_mul2),
    "np.inner": Operation(_mul2),
    "np.outer": Operation(_mul2),
    "np.matmul": Operation(_mul2),
    "np.cross": Operation(_mul2),
    "np.tensordot": Operation(_mul2),
    "np.kron": Operation(_mul2),
    "np.sqrt": Operation(_sqrt, _UFUNC_ARGUMENTS),
    "np.square": Operation(_sq, _UFUNC_ARGUMENTS),
    "np.cbrt": Operation(_cbrt, _UFUNC_ARGUMENTS),
    "np.linalg.solve": Operation(lambda us: _unit_div(us[1], us[0]) if len(us) > 1 else PLAIN),
    "np.shape": Operation(_d),
    "np.ndim": Operation(_d),
    "np.size": Operation(_d),
    "np.atleast_1d": Operation(_p),
    "np.atleast_2d": Operation(_p),
    "np.atleast_3d": Operation(_p),
    "np.add.reduceat": Operation(_p),
    "np.multiply.reduceat": Operation(_p),
    "np.maximum.reduceat": Operation(_p),
    "np.minimum.reduceat": Operation(_p),
    "np.searchsorted": Operation(_d),
    "np.unravel_index": Operation(_d),
    "np.ravel_index": Operation(_d),
    "np.unique": Operation(_p),
    "np.where": Operation(lambda us: us[1] if len(us) > 1 else PLAIN),
    "np.select": Operation(lambda us: us[0] if us else PLAIN),
    "np.piecewise": Operation(_p),
    "np.frompyfunc": Operation(_d),
    "np.fromiter": Operation(_d),
    "np.fromfunction": Operation(_d),
}

FUNCTIONS = {
    name: Operation(op.rule, op.contract, resolve_builtin(name)) for name, op in FUNCTIONS.items()
}

METHODS = {
    "abs": Operation(_p, contract(("a",))),
    "astype": Operation(_p, None),
    "ceil": Operation(_p, None),
    "clip": Operation(_p, None),
    "conj": Operation(_p, contract(("a",))),
    "copy": Operation(_p, contract(("a",), ("order",), controls=("order",))),
    "cumsum": Operation(_p, FUNCTIONS["np.cumsum"].contract),
    "diagonal": Operation(
        _p, contract(("a",), ("offset", "axis1", "axis2"), controls=("axis1", "axis2", "offset"))
    ),
    "flatten": Operation(_p, contract(("a",), ("order",), controls=("order",))),
    "floor": Operation(_p, None),
    "imag": Operation(_p, None),
    "max": Operation(_p, FUNCTIONS["np.max"].contract),
    "mean": Operation(_p, FUNCTIONS["np.mean"].contract),
    "min": Operation(_p, FUNCTIONS["np.min"].contract),
    "ravel": Operation(_p, contract(("a",), ("order",), controls=("order",))),
    "real": Operation(_p, None),
    "reshape": Operation(
        _p,
        contract(
            ("a",),
            controls=("copy", "order", "shape"),
            variadic="shape",
            keywords=("order", "copy"),
        ),
    ),
    "round": Operation(_p, None),
    "sort": Operation(_p, None),
    "squeeze": Operation(_p, contract(("a",), ("axis",), controls=("axis",))),
    "std": Operation(_p, FUNCTIONS["np.std"].contract),
    "sum": Operation(_p, FUNCTIONS["np.sum"].contract),
    "trace": Operation(
        _p,
        contract(
            ("a",),
            ("offset", "axis1", "axis2", "dtype", "out"),
            controls=("axis1", "axis2", "offset"),
        ),
    ),
}
