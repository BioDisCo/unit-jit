"""Differential execution coverage for the complete advertised call catalogue.

Cases are real source-backed functions so a passing fallback cannot masquerade
as successful inference. New catalogue entries must acquire a test domain here.
"""

import importlib.util
import sys

import numpy as np
import pytest

from unit_jit import QuantitySnapshot, trace_execution, unit_jit
from unit_jit._calls import FUNCTIONS
from unit_jit._calls import METHODS as METHOD_SPECS

# Domains are independent of inference rules: all operations receive finite,
# positive data; inverse trigonometric functions remain inside their domain.
CALLS = {
    "int",
    "float",
    "abs",
    "min",
    "max",
    "sum",
    *(
        f"np.{name}"
        for name in (
            "sum",
            "nansum",
            "min",
            "nanmin",
            "max",
            "nanmax",
            "mean",
            "nanmean",
            "std",
            "nanstd",
            "median",
            "nanmedian",
            "cumsum",
            "diff",
            "copy",
            "abs",
            "fabs",
            "sqrt",
            "square",
            "cbrt",
            "exp",
            "expm1",
            "log",
            "log2",
            "log10",
            "log1p",
            "sin",
            "cos",
            "tan",
            "arcsin",
            "arccos",
            "arctan",
            "sinh",
            "cosh",
            "tanh",
            "arcsinh",
            "arccosh",
            "arctanh",
        )
    ),
    *(
        f"math.{name}"
        for name in (
            "sqrt",
            "exp",
            "expm1",
            "log",
            "log2",
            "log10",
            "log1p",
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "sinh",
            "cosh",
            "tanh",
        )
    ),
}
METHODS = {
    "sum",
    "min",
    "max",
    "mean",
    "std",
    "cumsum",
    "flatten",
    "ravel",
    "copy",
    "squeeze",
    "diagonal",
    "trace",
    "conj",
    "abs",
    "reshape",
}


def test_catalogue_requires_explicit_domain_coverage():
    assert CALLS == {n for n, op in FUNCTIONS.items() if op.contract is not None}
    assert METHODS == {n for n, op in METHOD_SPECS.items() if op.contract is not None}


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    directory = tmp_path_factory.mktemp("contracts")
    result = {}
    for i, entry in enumerate(sorted(CALLS | {f"method.{n}" for n in METHODS})):
        path = directory / f"unit_contract_case_{i}.py"
        if entry.startswith("method."):
            method = entry.split(".")[1]
            expression = f"x.{method}({'-1' if method == 'reshape' else ''})"
        else:
            expression = f"{entry}(x)"
        path.write_text(f"import numpy as np\nimport math\ndef case(x):\n    return {expression}\n")
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        result[entry] = module.case
    yield result
    for function in result.values():
        sys.modules.pop(function.__module__)


@pytest.fixture(scope="module", params=["pint", "pintrs"])
def registry(request):
    return pytest.importorskip(request.param).UnitRegistry()


def assert_same(actual, expected):
    if hasattr(expected, "units"):
        assert hasattr(actual, "units")
        actual = actual.to(expected.units).magnitude
        expected = expected.magnitude
    else:
        assert not hasattr(actual, "units")
    assert np.shape(actual) == np.shape(expected)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("entry", sorted(CALLS | {f"method.{n}" for n in METHODS}))
@pytest.mark.parametrize("units", ["cm", "percent"])
def test_catalogue_matches_backend_and_observes_execution(cases, registry, entry, units):
    original = cases[entry]
    wrapped = unit_jit(original)
    scalar = entry.startswith("math.") or entry in {"int", "float"}
    # Exercise a fresh plan and then different values/scales through its cache.
    for factor in (1, 2):
        data = 0.25 * factor if scalar else np.array([[0.2, 0.4], [0.3, 0.5]]) * factor
        if entry in {"sum", "min", "max"}:
            data = np.array([0.2, 0.4]) * factor
        if entry == "np.arccosh":
            data = data + 200
        value = registry.Quantity(data, units)
        with np.errstate(all="ignore"):
            try:
                expected = original(value)
            except Exception as error:
                with trace_execution() as trace, pytest.raises((type(error), TypeError)):
                    wrapped(value)
                # Dimensional errors may be rejected statically, before entry.
                if trace.calls:
                    assert [call.path for call in trace.calls] == ["fallback"]
                else:
                    assert units == "cm"
                    assert entry in {"float", "int"} or entry.startswith(("math.", "np."))
            else:
                with trace_execution() as trace:
                    actual = wrapped(value)
                assert_same(actual, expected)
                assert len(trace.calls) == 1
                assert trace.calls[0].finished
                assert trace.calls[0].path == "fast"
                if trace.calls[0].path == "fast":
                    assert not isinstance(trace.calls[0].arguments["x"], QuantitySnapshot)
                    assert not isinstance(trace.calls[0].result, QuantitySnapshot)


OPTION_CASES = [
    (entry, name)
    for catalogue, prefix in ((FUNCTIONS, ""), (METHOD_SPECS, "method."))
    for entry, operation in catalogue.items()
    if (contract := operation.contract) is not None
    for name, parameter in contract.signature.parameters.items()
    if name not in contract.data
    and parameter.kind.name not in {"VAR_POSITIONAL", "VAR_KEYWORD", "POSITIONAL_ONLY"}
]


@pytest.mark.parametrize(("entry", "option"), OPTION_CASES)
def test_every_nondata_option_rejects_quantity_controls(tmp_path, registry, entry, option):
    """No non-data parameter may smuggle units into a stripped call."""
    path = tmp_path / "unit_option_case.py"
    callee = f"x.{entry.split('.')[1]}" if entry.startswith("method.") else entry
    operands = "" if entry.startswith("method.") else "x, "
    path.write_text(
        f"import numpy as np\nimport math\ndef case(x, option):\n"
        f"    return {callee}({operands}{option}=option)\n"
    )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    value = registry.Quantity(np.array([0.2, 0.4]), "cm")
    control = registry.Quantity(1, "cm")
    wrapped = unit_jit(module.case)
    try:
        try:
            expected = module.case(value, control)
        except Exception as error:
            with trace_execution() as trace, pytest.raises(type(error)):
                wrapped(value, control)
        else:
            with trace_execution() as trace:
                actual = wrapped(value, control)
            assert_same(actual, expected)
        assert [call.path for call in trace.calls] == ["fallback"]
    finally:
        sys.modules.pop(spec.name)
