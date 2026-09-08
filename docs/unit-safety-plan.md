# Unit safety implementation

The regression cases identified on commit `91711f9` are now covered by ordinary
passing tests in `tests/test_unit_safety.py`. The implementation follows three
shared mechanisms rather than special cases for individual failing examples.

## Current architecture

- Runtime specialization has one source of truth: `_states` maps a function key
  to either `_Plan` (compiled callable, return value, registry, boundary schema
  and dependencies) or `_Fallback` (reason). Public status predicates derive
  their answers from this mapping. Per-call fallback does not replace a valid
  cached plan.
- `_calls.py` holds operation specifications. Each entry owns its argument
  contract, unit rule and trusted callable identity; methods use the same operand
  binding and rule application. Shared NumPy signatures and unary contracts avoid
  repeating argument policies. A missing quantity contract explicitly restricts
  an operation to plain operands.
- `_values.py` defines the abstract domain, graph-preserving copies/joins and
  unit algebra. `_inferrer.py` handles syntax, lexical resolution, control flow
  and dependency collection. Every expression must produce an explicit domain
  value; the independent syntax audit additionally rejects missed subexpressions.
- `_prepared()` owns the conversion memo and restoration journal for both inner
  and outer invocations. Only the outer invocation establishes the fast execution
  context and wraps the final result. The public execution observer remains at
  actual body entry, after preparation.

The obsolete parallel return-unit/registry/schema caches, unused inference-cache
parameter and legacy tuple return representation have been removed. Internal tests
now use the single state model; the public decorator and tracing interfaces are
unchanged. Representation fault-injection tests and the existing syntax-audit
fault-injection tests exercise the two independent admission checks.

## S1: Canonical boundary schemas

Calls are bound with the original `inspect.Signature.bind` and `apply_defaults`.
Inference and conversion therefore see the same arguments, including defaults,
regardless of positional/keyword spelling. Converted defaults are explicitly
passed to the rewritten function. Compilation does not re-evaluate default
expressions or annotations.

`_schema.py` describes quantity dimensions and registry identity, plain values,
container elements and object attributes. It preserves graph references so aliases
and cycles are not mistaken for dimensionless values. Homogeneous sequences may
change length; heterogeneous sequences retain their positional schemas. Quantity
magnitudes and unit scales are not part of the schema.

A cached fast path rejects changes in dimensions, quantity presence or required
structure before stripping any fields. A different concrete object implementation
or callable uses the original function. Mixed registries, unsupported objects,
object arrays, and offset/logarithmic quantities use Pint fallback.

## S2: Shared abstract values and joins

The inferrer uses an explicit `AbstractValue` domain: `PLAIN`, `QuantityValue`,
`ObjectValue`, `SequenceValue`, `LambdaValue`, `UNKNOWN`, and `NO_RETURN`. Runtime
`None` is distinct from the abstract `PLAIN` marker. Quantity-valued dimensionless
results remain quantities, and raw unit objects are not valid abstract results.
Structured values contain per-element schemas. A shared join retains compatible
schemas and produces unknown otherwise; unknown is never treated as dimensionless.
Unknown operations trigger fallback even when their result is later discarded.

- Addition/subtraction requires compatible quantities or valid plain arithmetic.
  Adding a plain operand to a dimensional quantity uses Pint because validity can
  depend on whether that operand is zero.
- Comparisons, conditions and comprehension filters are visited. Incompatible
  operands fall back, preserving Pint's distinct equality and ordering semantics.
- Branch-dependent units and possible implicit `None` returns fall back.
- Loop entry and back-edge schemas must agree. Unit-changing loops and loops with
  early exits fall back. Unit-invariant scalar accumulation remains accelerated.
- A quantity raised to a nonliteral exponent falls back; literal powers retain
  inferred units.
- Local list append/extend/index writes update the affected element schemas.
  Aliases share those updates. Input-container mutations and uncertain container
  shapes fall back so mutations retain their original identity and effects.
- Heterogeneous outputs are wrapped position by position. Only explicitly
  homogeneous variable-length outputs repeat an element schema.

Bare `.magnitude` uses Pint fallback because it observes the original unit scale.
Explicit `.to_base_units().magnitude` and supported multiplicative
`.to(unit).magnitude` remain accelerated, with conversion compatibility checked.
Scale-sensitive operations such as rounding quantities also fall back. Known
numeric functions need an approved quantity signature to operate on stripped
quantities; recognizing a function name alone is insufficient.

## S3: Verified plans and restoration

A compilation plan is published only after inference and rewriting succeed for
its decorated dependencies. Guard checks and plan publication are serialized so
concurrent first calls cannot publish incompatible specializations. Callees are inferred for their actual abstract call
arguments, independently of an earlier boundary specialization. Recursive or
unresolved dependencies fall back at the outer boundary. Plain helpers can run in
the fast computation only if their analyzed body needs no unit-specific rewrite.
Inner decorated calls must belong to the outer plan.

Plans retain dependencies on referenced global values and method implementations.
If those bindings change, or a callee is subsequently disabled, the call runs the
original function before stripping any arguments.

Attribute writes must preserve their known unit schema. Assigning seconds to a
previously metre-valued attribute therefore uses Pint, preserving the valid Python
assignment instead of reattaching metre units.

Argument conversion uses one graph memo and restoration journal per boundary.
Preparation is inside `try/finally`, so a later conversion failure restores fields
already stripped. After execution, restored fields retain supported state updates.
Restoration attempts every journal entry and reports failures instead of silently
leaving raw floats. There is no retry after fast execution starts, avoiding duplicate
mutations, random draws or other effects.

## Validation and remaining scope

The safety suite includes every originally reported failure, plus checks for
fallback before side effects, callee specialization, changed implementations,
preparation/execution failures, default-expression evaluation, implicit returns,
loop exits, aliasing, input-container mutation, mixed registries and unknown
intermediates, as well as concurrent first-call compilation. Positive controls
require supported cases to remain accelerated.

```sh
.venv/bin/python -m pytest tests/test_unit_safety.py -q
.venv/bin/python -m pytest -q
ruff check src/unit_jit tests/test_unit_safety.py
```

This is a conservative compiler for its supported operations, not a complete
Python/Pint equivalence proof. Unsupported control flow and signatures can lose
acceleration. Registry definitions must remain stable after compilation. In-place
stripping of a shared object is still not safe for concurrent access by another
thread; a thread-local fast-zone flag does not provide object isolation.


## Additional boundary coverage

Inference records the expressions it checks. A separate CST visitor audits the
function body afterward, allowing only checked expressions and explicit syntactic
names (attribute names, keyword labels and parameter names). Assignment targets
and resolved callable references are accounted for by their corresponding rules.
Any remaining expression disables compilation before conversion. The audit is
deliberately conservative for unreachable statements and unused lambda bodies.
Fault-injection tests omit whole subtrees from otherwise successful inference and
require fallback; this protects against traversal omissions in future handlers.

Both reads and writes inspect every index and slice bound. Quantity-valued
indices retain original backend semantics through fallback. Quantity-array
assignment, sorting and augmented assignment involving quantity-array inputs also
fall back, preserving storage sharing and mutations across views and aliases.
Scalar/array representation changes invalidate a cached specialization.

Empty/nonempty container transitions use the original function when no shared
element-unit evidence exists. Instance callable overrides invalidate object
dispatch. Plain callees are checked as plain callees even when their original
function was separately decorated elsewhere; registration alone does not mean a
particular call goes through a wrapper. Quantity keyword operands without a bound
unit signature fall back. Numba dispatch is keyed by function identity and enables
bounds checking.

The added edge suite covers these cases, offset/logarithmic/scaled dimensionless
quantities, nested and empty outputs, signature binding and defaults, and failures
during restoration (including preservation of an execution exception). Mutation
and index tests run against Pint and pintrs, with and without Numba. Positive
controls require read-only quantity arithmetic and checked indexing to stay fast.

For the optional backend matrix, install the dev and numba extras and run:

```sh
PYTHONPATH=src python -m pytest -q
ruff check src tests
```


## Execution evidence

The public `trace_execution()` context manager records actual body dispatch,
including nested decorated calls, instead of inferring execution from a cached
plan. Liveness tests assert fast entry, SI-valued argument snapshots, raw results
and final quantity correctness. The Numba matrix also requires populated
nopython signatures. Fallback tests require original-scale quantity snapshots;
one test exercises fast, fallback and fast again while a cached plan stays active.

The liveness checks exposed a conservative reduction rule that unnecessarily
disabled nonempty input-list sums. Input schemas already guard empty transitions,
so these reductions can use their known element units; potentially empty local
sequences still require fallback. Input units with common registry provenance
and dimensions share an abstract unit during inference. This avoids accidental
fallback when pintrs clones registries while producing Unit objects, while keeping
the original-quantity registry guard. A converted quantity that itself belongs
to a different registry remains subject to that guard.

Trace tests cover body exceptions, entry-time visibility, nested scopes, thread
isolation, mutable snapshots, cycles, and absence of snapshot work when disabled.


Existing tests that promise acceleration now use `expect_execution`, a shared
test assertion built on the public trace interface. It requires at least one
actual invocation, the expected path for every invocation in its scope, and
quantity-free argument/result snapshots for fast calls, including nested calls.
Correctness assertions remain in place. State predicates remain where tests
specifically check cached-plan or disabled-state behavior; construction-time
and decoration-time compilation tests also observe their implicit calls.

The migration covers NumPy/Numba hot paths, unit conversion, plain and overridden
helpers, mutation/restoration, cyclic and namedtuple objects, and safety-suite
positive controls. Invalidation and deliberate fallback paths have explicit
fallback observations. The pintrs literal-loop regression exposed registry clones
in derived units; inference now carries verified registry provenance through
arithmetic and interns equivalent abstract units without merging distinct
registry origins. Regression tests cover zero and nonzero loop counts and
rejection of distinct origins.


## Lexical bindings and complete call contracts

`LexicalBindings` uses LibCST's scope analysis and is shared by inference and
rewriting. It recognizes parameter bindings, assignments throughout a function,
lambda scopes and comprehension targets. Function free-variable names also
prevent captured closure bindings from being mistaken for globals. Global helpers, builtins, modules and
registry literals are considered only when their reference is external to a
local scope. Reassignment also invalidates the concrete argument object retained
for method inference. Literal conversion targets use the same lexical resolver
and unit algebra; a quantity target contributes its units, not its magnitude.
Conversion rewriting emits a parenthesized numeric expression without inserting
a helper name that user code could shadow.

`CallContract` binds positional and keyword operands through an inspect.Signature.
Declared data operands feed the unit rule. Declared controls must be plain;
undeclared or unsupported supplied parameters force fallback. NumPy contracts
take their signatures from the installed implementations. Methods reuse the
corresponding contracts and, when options are supplied, also require a concrete
receiver whose backend signature accepts them. Known library implementations are
checked against captured callable identities.

Variadic min/max join every data operand. Sum's explicit start, numeric
initial/prepend/append values, dtype conversions, output storage and callbacks
retain original-backend semantics through fallback. Quantity sequences require
an explicit sequence-data contract; numeric NumPy coercion of quantity lists is
not assumed equivalent to stripping them first. Plain sequences produce ordinary
numeric library results without erroneous container wrapping.

Regression tests reproduce all six reported failures, then exercise local
assignments/lambdas, comprehension shadowing, use before assignment, module and
builtin parameters, conversion targets, positional/keyword options, extra selector
operands, defaults, callbacks, output mutation, and replaced library callables.
Trace assertions distinguish error-preserving fallback from genuine stripped
execution. Compatible min/max operands, bound data keywords, scale-independent
controls and supported Numba calls remain fast.


## Operator semantics and quantity-method implementation guards

Floor division has a separate inference rule: plain operands retain fast
execution, while quantity operands use backend execution for both `//` and
`//=`. Identity comparisons use the original function before conversion,
including chained comparisons and identity-dependent branches. Dimensional
equality comparisons remain supported.

Quantity method contracts now verify implementations, not only argument shapes.
The supported quantity classes' methods and dispatch hooks are captured when
unit-jit imports the backends. Static descriptor checks reject overrides before
first compilation, and equivalent bindings guard cached plans. This covers
methods on derived expressions as well as direct arguments. The concrete pintrs
classes are registered alongside its public facades so metaclass instance checks
cannot stand in for implementation evidence.

Quantity schemas include concrete quantity/array classes and instance dispatch
overrides. Recursive sequence checks distinguish an implementation change
(fallback) from a dimensional change (rejection), including a homogeneous sequence
becoming heterogeneous solely because one element overrides a method. Plans
retain class/descriptor dependencies rather than the original quantity data.

Regression tests compare results/errors against both backends, check that
side effects execute once, cover class/instance/custom-array overrides before
and after compilation, and require fast execution to resume after an override is
removed from an otherwise valid cached plan. Trace-based positive controls cover
ordinary floor division, dimensional equality and standard quantity methods,
including Numba execution and invalidation of a warmed Numba plan.

## Admission inventory and catalogue-wide tests

Fast execution has several independent obligations. Checking a function name or
matching dimensions alone does not discharge them:

| Layer | Admission obligation | Executable coverage |
| --- | --- | --- |
| Syntax and control flow | Every expression is analyzed; branches and loops preserve supported schemas | Expression-audit fault injection; branch/loop and index tests |
| Operators | Unit rule preserves backend semantics; identity and quantity floor division retain original execution | Operator/mutation matrices and positive raw-execution controls |
| Library calls and methods | Callable identity, backend availability, data operands and option roles are verified | Complete call/method catalogue matrix; every non-data keyword receives a quantity control |
| Quantity protocols | Arithmetic, reflected/in-place operations, comparisons, coercion, indexing, iteration and NumPy dispatch retain trusted implementations | Every guarded protocol overridden before first call and after warmup |
| Representations | Numeric and array subclasses cannot inherit a proof of plain numeric behavior | Custom number, array and sequence regressions, plus native numeric fast controls |
| Cached plans | Bound schemas and implementation dependencies still hold | Repeated catalogue calls; changed dispatch and restored-dispatch tests |
| Preparation and execution | Units are stripped before fast entry; failures restore state without replay | Public execution traces and mutation/restoration fault injection |

`test_contract_matrix.py` contains an independent, explicit domain inventory and
asserts equality with the production catalogue. Adding a quantity call contract
without adding a domain therefore fails the suite. Each baseline runs against
Pint and pintrs, with centimeter and percentage inputs and changed values on a
second call. Successful baselines must enter the fast function with raw arguments
and results; unsupported backend methods must execute the original function.
Dimensional errors may instead be rejected by inference before either body runs.
This distinction is checked explicitly, rather than treating a populated cache as
proof of acceleration.

Protocol verification now applies to the entire plan, including arithmetic,
builtins and NumPy calls. It is not attached solely to individual method handlers.
A method must also exist on the original backend and the stripped representation:
NumPy providing a method does not authorize inventing it on a quantity backend.

This is a finite executable contract for the advertised subset, not a proof of
equivalence for arbitrary Python, arbitrary backend versions or every floating
point input. Existing requirements concerning stable registry definitions and
exclusive access to objects during in-place stripping still apply. New syntax,
representations and library capabilities need an admission argument and coverage;
they must not inherit support merely because an abstract operand looks unitless.

## Shared admission assumptions for composition

`_admission.py` centralizes native scalar classification, ordinary object field
access, immutable unprepared operands, and callable-state dependencies. These
rules apply across explicit arguments, globals, plain helpers and cached plans:

- Numeric subclasses cannot obtain a unitless proof through a global name after
  being rejected at the argument boundary. Object fields are inferred from their
  storage only when ordinary attribute read/write protocols are in force and no
  stored field is intercepted by a data descriptor.
- Plain helper defaults and partial arguments do not cross the wrapper's
  conversion boundary. They must be immutable native values, or, for a bound
  method receiver, an object already prepared by the caller. Otherwise the entire
  call uses original execution. Explicitly supplied quantity operands continue
  to accelerate even when they replace an unused quantity default.
- Plans retain callable code/default identities and keyword-default entries,
  including partial keyword bindings. Invalid callable assumptions cause fallback
  before checking a stale dimensional schema. The outer wrapper also binds against
  the current signature when defaults change. Restoring a cached implementation
  restores eligibility for fast execution.
- Container comparisons use original execution: Python equality and ordering can
  depend on element identity even without an explicit `is` expression. Compatible
  scalar quantity comparisons remain accelerated.

`test_admission_boundaries.py` tests these assumptions through direct and helper
calls, positional and keyword defaults, partial application, cold and warmed
execution, restoration of implementations, and list/tuple comparisons with
shared scalar/array quantities. Positive controls require observed raw execution;
fallback assertions require entry into the original function.
