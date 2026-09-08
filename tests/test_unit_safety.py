"""Regression coverage for the safety rules in docs/unit-safety-plan.md.

Each test starts with fresh inference state for this module, independent of order.
"""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from _jit_state import expect_execution
from pint import DimensionalityError, Quantity, UnitRegistry

import unit_jit as runtime
from unit_jit import is_jit_disabled, unit_jit

ureg = UnitRegistry()


@pytest.fixture(autouse=True)
def fresh_inference(monkeypatch):
    prefix = f"{__name__}::"
    monkeypatch.setattr(
        runtime, "_states", {k: v for k, v in runtime._states.items() if not k.startswith(prefix)}
    )


def assert_quantity(actual, expected):
    assert isinstance(actual, Quantity)
    assert actual.dimensionality == expected.dimensionality
    assert actual.to(expected.units).magnitude == pytest.approx(expected.magnitude)


@unit_jit
def velocity(d, t=2 * ureg.s):
    return d / t


@unit_jit
def first(values):
    return values[0]


class Box:
    def __init__(self, x):
        self.x = x

    @unit_jit
    def add(self, dx):
        return self.x + dx

    @unit_jit
    def replace(self, value):
        self.x = value
        return self.x


@pytest.mark.parametrize("warm_style", ["positional", "keyword", "default"])
def test_bound_parameter_dimensions_are_guarded(warm_style):
    if warm_style == "positional":
        velocity(6 * ureg.m, 2 * ureg.s)
        args, kwargs = (), {"d": 6 * ureg.s, "t": 2 * ureg.m}
    elif warm_style == "keyword":
        velocity(d=6 * ureg.m, t=2 * ureg.s)
        args, kwargs = (6 * ureg.s, 2 * ureg.m), {}
    else:
        velocity(6 * ureg.m)
        args, kwargs = (6 * ureg.m,), {"t": 2 * ureg.m}
    with pytest.raises(TypeError):
        velocity(*args, **kwargs)


@pytest.mark.parametrize("quantity_first", [True, False])
def test_quantity_presence_is_part_of_signature(quantity_first):
    initial = 6 * ureg.m if quantity_first else 6
    changed = 6 if quantity_first else 6 * ureg.m
    velocity(initial, 2 * ureg.s)
    with pytest.raises(TypeError):
        velocity(changed, 2 * ureg.s)


@pytest.mark.parametrize("container", [list, tuple])
def test_container_element_dimensions_are_guarded(container):
    first(container([1 * ureg.m]))
    with pytest.raises(TypeError):
        first(container([1 * ureg.s]))


@pytest.mark.parametrize("new_instance", [False, True])
def test_object_attribute_dimensions_are_guarded(new_instance):
    box = Box(1 * ureg.m)
    box.add(2 * ureg.m)
    if new_instance:
        box = Box(1 * ureg.s)
    else:
        box.x = 1 * ureg.s
    with pytest.raises(TypeError):
        box.add(2 * ureg.m)
    assert_quantity(box.x, 1 * ureg.s)


def test_assignment_can_change_attribute_units_without_relabeling():
    box = Box(1 * ureg.m)
    reference = Box(1 * ureg.m)
    expected = Box.replace.__wrapped__(reference, 3 * ureg.s)
    result = box.replace(3 * ureg.s)
    assert_quantity(result, expected)
    assert_quantity(box.x, reference.x)


@unit_jit
def add_scalar(x, scalar):
    return x + scalar


@unit_jit
def branch_add(x, t, flag):
    if flag:
        y = x
    else:
        y = t
    return y + x


@unit_jit
def choose(x, t, flag):
    return x if flag else t


@unit_jit
def compare(x, t):
    return x < t


@unit_jit
def compare_condition(x, t):
    if x < t:
        return 1
    return 0


@pytest.mark.parametrize(
    "func,args",
    [
        (add_scalar, (2 * ureg.m, 1)),
        (branch_add, (2 * ureg.m, 3 * ureg.s, False)),
        (compare, (2 * ureg.m, 3 * ureg.s)),
        (compare_condition, (2 * ureg.m, 3 * ureg.s)),
    ],
    ids=["dimensional-plus-number", "conflicting-branches", "comparison", "condition"],
)
def test_invalid_operations_do_not_execute_as_float_arithmetic(func, args):
    with pytest.raises(DimensionalityError):
        func.__wrapped__(*args)
    with pytest.raises((TypeError, DimensionalityError)):
        func(*args)


@pytest.mark.parametrize("flag", [True, False])
def test_valid_branch_dependent_returns_preserve_units(flag):
    args = (2 * ureg.m, 3 * ureg.s, flag)
    assert_quantity(choose(*args), choose.__wrapped__(*args))


@unit_jit
def power(x, n):
    return x**n


@unit_jit
def multiply_loop(x, t, n):
    y = x
    for _ in range(n):
        y = y * t
    return y


@unit_jit
def multiply_while(x, t, n):
    y = x
    i = 0
    while i < n:
        y = y * t
        i += 1
    return y


@pytest.mark.parametrize("exponent", [2, 3, 0, -1, 0.5])
def test_dynamic_power_tracks_changing_exponent(exponent):
    power(4 * ureg.m, 1)  # Establish cached state with a different exponent.
    args = (4 * ureg.m, exponent)
    assert_quantity(power(*args), power.__wrapped__(*args))


@pytest.mark.parametrize("func", [multiply_loop, multiply_while])
@pytest.mark.parametrize("count", [0, 2, 3])
def test_loop_units_account_for_zero_and_multiple_iterations(func, count):
    args = (2 * ureg.m, 3 * ureg.s, count)
    assert_quantity(func(*args), func.__wrapped__(*args))


@unit_jit
def append_values(x, t):
    values = [x]
    values.append(t)
    return values


@unit_jit
def extend_values(x, t):
    values = [x]
    values.extend([t])
    return values


@unit_jit
def replace_element(x, t):
    values = [x, x]
    values[1] = t
    return values


@pytest.mark.parametrize("func", [append_values, extend_values, replace_element])
def test_heterogeneous_list_mutation_preserves_each_element(func):
    args = (1 * ureg.m, 2 * ureg.s)
    result = func(*args)
    expected = func.__wrapped__(*args)
    assert isinstance(result, list)
    assert len(result) == len(expected)
    for actual, reference in zip(result, expected, strict=True):
        assert_quantity(actual, reference)


@unit_jit
def magnitude(x):
    return x.magnitude


@unit_jit
def base_magnitude(x):
    return x.to_base_units().magnitude


@unit_jit
def explicit_magnitude(x):
    return x.to(ureg.cm).magnitude


@pytest.mark.parametrize("unit", [ureg.cm, ureg.mm])
def test_bare_magnitude_preserves_input_scale_across_calls(unit):
    with expect_execution(magnitude, "fallback"):
        assert magnitude(2 * ureg.m) == 2  # Warm with base units before changing scale.
    value = 2 * unit
    with expect_execution(magnitude, "fallback"):
        assert magnitude(value) == pytest.approx(magnitude.__wrapped__(value))


# Positive controls: safety improvements must retain ordinary supported fast paths.


def test_equivalent_calls_and_unit_scales_stay_fast():
    with expect_execution(velocity):
        assert_quantity(velocity(6 * ureg.m, 2 * ureg.s), 3 * ureg.m / ureg.s)
        assert_quantity(velocity(d=600 * ureg.cm, t=2000 * ureg.ms), 3 * ureg.m / ureg.s)


def test_same_style_wrong_dimensions_already_raise():
    velocity(6 * ureg.m, 2 * ureg.s)
    with pytest.raises(TypeError):
        velocity(6 * ureg.m, 2 * ureg.m)


def test_object_value_and_scale_changes_stay_fast():
    box = Box(1 * ureg.m)
    with expect_execution(box.add):
        assert_quantity(box.add(2 * ureg.m), 3 * ureg.m)
        box.x = 200 * ureg.cm
        assert_quantity(box.add(3 * ureg.m), 5 * ureg.m)


@pytest.mark.parametrize("scalar", [0, 1])
def test_dimensionless_addition_matches_pint(scalar):
    value = 2 * ureg.dimensionless
    with expect_execution(add_scalar):
        assert_quantity(add_scalar(value, scalar), add_scalar.__wrapped__(value, scalar))


def test_zero_operand_can_be_added_to_dimensional_quantity():
    value = 2 * ureg.m
    assert_quantity(add_scalar(value, 0), add_scalar.__wrapped__(value, 0))


@pytest.mark.parametrize("func", [base_magnitude, explicit_magnitude])
def test_explicit_magnitude_conversion_stays_fast(func):
    with expect_execution(func):
        for value in (2 * ureg.cm, 2 * ureg.m):
            assert func(value) == pytest.approx(func.__wrapped__(value))


def test_compatible_comparison_stays_fast():
    with expect_execution(compare):
        assert compare(2 * ureg.m, 300 * ureg.cm) == compare.__wrapped__(2 * ureg.m, 300 * ureg.cm)


@pytest.mark.parametrize("func", [append_values, extend_values, replace_element])
def test_homogeneous_list_mutation_stays_fast(func):
    with expect_execution(func):
        result = func(1 * ureg.m, 200 * ureg.cm)
        for actual, expected in zip(result, [1 * ureg.m, 2 * ureg.m], strict=True):
            assert_quantity(actual, expected)


@unit_jit
def accumulate(x, dx, n):
    for _ in range(n):
        x += dx
    return x


@pytest.mark.parametrize("count", [0, 1, 3])
def test_unit_invariant_loop_stays_fast(count):
    args = (2 * ureg.m, 30 * ureg.cm, count)
    with expect_execution(accumulate):
        assert_quantity(accumulate(*args), accumulate.__wrapped__(*args))


@unit_jit
def nested_power(counter, x, exponent):
    counter.count += 1
    return power(x, exponent)


class Counter:
    def __init__(self):
        self.count = 0


def test_unsupported_callee_falls_back_before_any_side_effect():
    counter = Counter()
    with expect_execution(nested_power, "fallback"):
        assert_quantity(nested_power(counter, 2 * ureg.m, 3), 8 * ureg.m**3)
    assert counter.count == 1
    assert is_jit_disabled(nested_power)
    assert is_jit_disabled(power)


@unit_jit
def identity(x):
    return x


@unit_jit
def nested_identity(x):
    return identity(x)


def test_callee_is_inferred_for_call_site_units_not_previous_cache():
    assert_quantity(identity(1 * ureg.m), 1 * ureg.m)
    with expect_execution(nested_identity):
        assert_quantity(nested_identity(2 * ureg.s), 2 * ureg.s)


class Helper:
    def rate(self, x):
        return x * 2

    @unit_jit
    def run(self, x):
        return self.rate(x)


def test_changed_method_implementation_invalidates_plan(monkeypatch):
    helper = Helper()
    with expect_execution(helper.run):
        assert_quantity(helper.run(3 * ureg.m), 6 * ureg.m)

    def changed(self, x):
        return x / ureg.s

    monkeypatch.setattr(Helper, "rate", changed)
    with expect_execution(helper.run, "fallback"):
        assert_quantity(helper.run(3 * ureg.m), 3 * ureg.m / ureg.s)


class Pair:
    def __init__(self):
        self.x = 1 * ureg.m
        self.y = 2 * ureg.m


@unit_jit
def pair_sum(pair):
    return pair.x + pair.y


def test_preparation_failure_restores_previously_stripped_fields(monkeypatch):
    pair = Pair()
    pair_sum(pair)  # Inference succeeds before injecting a conversion failure.
    original_conversion = runtime._base_quantity
    failing_value = pair.y

    def conversion(value):
        if value is failing_value:
            raise RuntimeError("conversion failed")
        return original_conversion(value)

    # Backend overrides now invalidate the plan before preparation. Inject at
    # the conversion boundary to continue testing partial preparation rollback.
    monkeypatch.setattr(runtime, "_base_quantity", conversion)
    with pytest.raises(RuntimeError, match="conversion failed"):
        pair_sum(pair)
    assert_quantity(pair.x, 1 * ureg.m)
    assert pair.y is failing_value
    assert not runtime._in_fast_zone()


@unit_jit
def change_then_divide(pair, divisor):
    pair.x += pair.y
    return 1 / divisor


def test_fast_execution_failure_restores_updated_state_without_retry():
    pair = Pair()
    with pytest.raises(ZeroDivisionError):
        change_then_divide(pair, 0)
    assert_quantity(pair.x, 3 * ureg.m)
    assert_quantity(pair.y, 2 * ureg.m)
    assert not runtime._in_fast_zone()


_default_calls = 0


def make_default():
    global _default_calls
    _default_calls += 1
    return 2 * ureg.cm


@unit_jit
def default_value(x=make_default()):
    return x * 2


def test_rewriting_does_not_reexecute_default_expressions(monkeypatch):
    previous = _default_calls
    monkeypatch.delitem(runtime._compiled, __name__, raising=False)
    with expect_execution(default_value):
        assert_quantity(default_value(), 4 * ureg.cm)
        assert _default_calls == previous


@unit_jit
def branch_with_implicit_none(x, flag):
    if flag:
        return x


def test_implicit_none_return_uses_fallback():
    with expect_execution(branch_with_implicit_none, "fallback"):
        assert_quantity(branch_with_implicit_none(2 * ureg.m, True), 2 * ureg.m)
    with expect_execution(branch_with_implicit_none, "fallback"):
        assert branch_with_implicit_none(2 * ureg.m, False) is None
    assert is_jit_disabled(branch_with_implicit_none)


@unit_jit
def break_before_restoring_units(x, t, flag):
    for _ in range(2):
        x *= t
        if flag:
            break
        x /= t
    return x


@pytest.mark.parametrize("flag", [True, False])
def test_loop_exit_units_are_not_taken_from_back_edge_only(flag):
    args = (2 * ureg.m, 3 * ureg.s, flag)
    with expect_execution(break_before_restoring_units, "fallback"):
        assert_quantity(
            break_before_restoring_units(*args), break_before_restoring_units.__wrapped__(*args)
        )
    assert is_jit_disabled(break_before_restoring_units)


@unit_jit
def aliased_append(x, t):
    values = [x]
    alias = values
    alias.append(t)
    return values


def test_local_list_aliases_share_mutation_schema():
    with expect_execution(aliased_append):
        result = aliased_append(1 * ureg.m, 2 * ureg.s)
        assert len(result) == 2
        assert_quantity(result[0], 1 * ureg.m)
        assert_quantity(result[1], 2 * ureg.s)


@unit_jit
def mutate_input_list(values, t):
    values.append(t)
    return values


def test_input_list_mutation_preserves_original_identity_and_effects():
    values = [1 * ureg.m]
    with expect_execution(mutate_input_list, "fallback"):
        result = mutate_input_list(values, 2 * ureg.s)
    assert result is values
    assert len(values) == 2
    assert_quantity(values[0], 1 * ureg.m)
    assert_quantity(values[1], 2 * ureg.s)
    assert is_jit_disabled(mutate_input_list)


@unit_jit
def filtered_values(x, t):
    return [x for _ in range(1) if x < t]


def test_comprehension_conditions_are_checked():
    with pytest.raises((TypeError, DimensionalityError)):
        filtered_values(1 * ureg.m, 2 * ureg.s)


def test_registry_switch_is_rejected_before_stripping():
    identity(1 * ureg.m)
    other = UnitRegistry()
    with pytest.raises(TypeError):
        identity(1 * other.m)


@unit_jit
def add_values(x, y):
    return x + y


def test_mixed_registries_retain_pint_error():
    other = UnitRegistry()
    args = (1 * ureg.m, 2 * other.m)
    with pytest.raises(ValueError):
        add_values.__wrapped__(*args)
    with pytest.raises(ValueError):
        add_values(*args)


@unit_jit
def invalid_conversion(x):
    return x.to(ureg.s).magnitude


def test_explicit_conversion_checks_source_dimensions():
    with pytest.raises((TypeError, DimensionalityError)):
        invalid_conversion(1 * ureg.m)


@unit_jit
def unknown_then_known(x, t):
    _value = x**t
    return x


def test_unknown_intermediate_cannot_be_hidden_by_known_return():
    with expect_execution(unknown_then_known, "fallback"):
        assert_quantity(unknown_then_known(2 * ureg.m, 3), 2 * ureg.m)
    assert is_jit_disabled(unknown_then_known)


@unit_jit
def choose_alias(x, t, flag):
    left = [x]
    right = [x]
    if flag:
        alias = left
    else:
        alias = right
    alias.append(t)
    return left


@pytest.mark.parametrize("flag", [True, False])
def test_branch_aliases_are_not_merged_just_because_units_match(flag):
    args = (1 * ureg.m, 2 * ureg.s, flag)
    actual = choose_alias(*args)
    expected = choose_alias.__wrapped__(*args)
    assert len(actual) == len(expected)
    for result, reference in zip(actual, expected, strict=True):
        assert_quantity(result, reference)


@unit_jit
def repeat_heterogeneous(x, t, count):
    return [x, t] * count


@pytest.mark.parametrize("count", [0, 2])
def test_repetition_does_not_extend_last_unit_of_heterogeneous_list(count):
    args = (1 * ureg.m, 2 * ureg.s, count)
    actual = repeat_heterogeneous(*args)
    expected = repeat_heterogeneous.__wrapped__(*args)
    assert len(actual) == len(expected)
    for result, reference in zip(actual, expected, strict=True):
        assert_quantity(result, reference)


def test_concurrent_first_calls_cannot_publish_incompatible_plans(monkeypatch):
    entered, release, attempted = Event(), Event(), Event()
    infer = runtime.infer_return_units

    def delayed_inference(func, *args, **kwargs):
        if func is identity.__wrapped__:
            entered.set()
            assert release.wait(5)
        return infer(func, *args, **kwargs)

    def second_call():
        attempted.set()
        return identity(2 * ureg.s)

    monkeypatch.setattr(runtime, "infer_return_units", delayed_inference)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_call = pool.submit(identity, 1 * ureg.m)
        try:
            assert entered.wait(5)
            second = pool.submit(second_call)
            assert attempted.wait(5)
        finally:
            release.set()
        assert_quantity(first_call.result(timeout=5), 1 * ureg.m)
        with pytest.raises(TypeError):
            second.result(timeout=5)
