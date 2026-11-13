# Testing and validation

The test suite is organized primarily by **solver method** and, for broad reference regressions, by **mathematical input type**. This makes a failure point directly to the method or input family that needs attention.

```text
tests/
├── methods/
│   ├── algebraization/
│   ├── critical_points/
│   ├── kkt/
│   ├── linear_programming/
│   ├── quadratic_programming/
│   ├── sdp/
│   ├── sos/
│   ├── moments/
│   ├── intervals/
│   ├── semialgebraic/
│   ├── pipeline/
│   └── certification/
├── reference/
│   ├── by_input_type/
│   │   ├── test_polynomial.py
│   │   ├── test_radical.py
│   │   ├── test_rational.py
│   │   ├── test_nonsmooth.py
│   │   ├── test_linear_programming.py
│   │   ├── test_semialgebraic.py
│   │   ├── test_wrappers.py
│   │   └── ...
│   ├── corpus/
│   │   ├── cases.json
│   │   ├── checks.py
│   │   └── test_cases.py
│   └── test_reference_case_inventory.py
├── api/
├── infrastructure/
├── validation/
└── tooling/
```

## Runtime guard

The test environment includes `pytest-timeout` with a **60-second per-test limit**. A test that exceeds one minute is treated as a test-design or performance regression and fails rather than hanging the suite. Slow benchmark work belongs in `benchmarks/`, not in ordinary pytest cases.

## Organization

**Method tests** Live under `tests/methods/<method>/`. Unit, integration, regression, and property-based checks for the same solver family stay together. Tests that require optional backends retain the `integration` marker.

**Reference tests** Live under `tests/reference/by_input_type/` and are grouped by the mathematical form of the problem. The complete external corpus lives in `tests/reference/corpus/`, with executable assertions and isolated solver budgets. See {doc}`../reference/test-corpus` for its mathematical translation and capability statuses.

**API and infrastructure tests** Cover result models, diagnostics, shared caches, normalization helpers, and other code that is not itself an optimization method.

**Validation and tooling tests** Keep the independent numerical validation battery, documentation checks, public-docstring checks, and benchmark infrastructure separate from solver correctness tests.

## Pytest markers

The suite declares these markers:

```text
integration
validation
property
reference
reference_corpus
unsupported
capability_probe
slow
```

Useful development commands include:

```bash
python -m pytest -q -m "not unsupported and not reference_corpus"
python -m pytest -q -m integration
python -m pytest -q -m property
python -m pytest -q -m reference
python -m pytest -q -m validation
python -m pytest -q -m "not slow and not unsupported"
```

To inspect the documented unsupported cases without running them:

```bash
python -m pytest --collect-only -q -m unsupported
```

## Backend requirements

The test extra supplies pytest, Hypothesis and time limits. It does not install optional analysis backends. The OS/Python suite installs the semialg extra explicitly; optional-backend jobs install only their selected backend. Tests needing funcprops or semialg skip when that backend is absent. Arb is a required runtime dependency.

## Numerical validation battery

The package exposes:

```python
from symbopt.validation_suite import run_validation_suite

results = run_validation_suite()
```

The validation test verifies that the known SDP/SOS numerical cases remain green independently of ordinary unit tests.

## Documentation validation

Documentation CI builds Sphinx with warnings treated as errors:

```bash
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

The repository also verifies required pages, MyST targets, and stable public API documentation coverage.

## Expectations for new changes

- A solver route should have success, infeasible/unsupported, and regression cases.
- A correctness bug should receive a minimal regression test.
- A new backend call should have both adapter-level and live integration tests.
- A new rigorous certificate route should verify that heuristic evidence cannot trigger rigorous status.
- A new public API symbol should be documented and covered by API tests.
- A newly discovered unsupported but valuable problem can be added to the `unsupported` reference suite with a precise reason.
## Complete reference corpus

The 779-case corpus has one executable pytest parameter per mathematical problem. `cases.json` stores the problem expressions, expected values, assumptions, wrapper choice, precision requests and capability status. `manifest.json` records case-identity checksums and mathematical category counts. All expressions use SymPy constructors; no external parser or input file is needed.

The default OS/Python matrix runs the core profile. The separate reference job executes the complete bounded corpus with analysis backends installed. Unsupported mathematical targets use executable expected failures. Known resource-limited cases can pass when a backend completes; a timeout records an expected failure instead of claiming the mathematical problem is impossible.

See {doc}`../reference/test-corpus` for commands, representation rules and maintenance requirements.

## Backend matrix

`.github/workflows/optional-backends.yml` exercises backend-sensitive tests in separate environments for core, semialg, CVXPY+Clarabel, CVXPY+SCS, and the full combination. Optional packages matter here: optional packages can change routing and exact recovery behavior even when their own adapter tests pass.

## Certificate-adversarial tests

Rigorous routes must include negative tests. Perturbing an exact residual, introducing a non-PSD Gram matrix, or leaving PSD status unknown must prevent a certificate from becoming verified. External SDP tests compare final exact certificates rather than requiring identical floating-point matrices.

## Result and normalization invariants

Property tests cover mathematically equivalent input forms (`And`, lists, tuples, sets, bounds mappings, and scalar/list variable syntax) and stable result invariants such as mutual exclusion of infeasible/unbounded states and exact objective evaluation at represented certified optimizers.

## Performance guards

Known pathological transitions receive structural performance tests rather than fragile wall-clock benchmarks. In particular, large SOS reconstruction vectors must not enter common-number-field primitive-element construction. General timing benchmarks remain in `benchmarks/`.

## Executable capability probes

Each full-corpus parameter executes a mathematical test targeting the relevant optimization API; the region-construction check verifies the defining predicate directly. A resource budget can expire during expression reconstruction or solver work, which records a budget limitation. Solver work is isolated in a child process with a per-case budget. Unexpected worker exits and exception types fail the test. A static expected failure that starts satisfying its mathematical assertions produces strict XPASS and must be promoted. Budget-limited cases may become ordinary passes without a hardware-dependent XPASS failure.

The smaller `capability_probe` profile remains useful for selected scheduled checks. The complete corpus is a separate profile so its runtime does not multiply across every core matrix entry.

## Installed-wheel smoke test

The optional-backend workflow builds the wheel, installs that wheel into a clean virtual environment, and exercises the compact API, detailed results, Arb, semialg, and one live external-SDP SOS certificate. Source-tree success therefore does not substitute for packaging validation.

## Mathematical test priorities

Use exact expected values and feasible coordinates, including open endpoints and excluded boundary points. Compare simplex routes against independent small rational LP oracles; scale and permute rows to exercise basis changes. Certificate mutations should classify each field as a theorem premise, a derived summary or diagnostic metadata, and include coherent changes to multiple fields. Endpoint generators should cover cancellation near rational and algebraic bounds.

Run advertised examples from installed wheels with minimal and optional dependency sets. Keep result-contract assertions separate from route-selection tests: a faster exact route can change a method label while preserving the mathematical answer. A disabled expected failure is an inventory entry; executed bounded probes provide evidence about coverage.

## Core-runtime wheel checks

The `core-wheel` job installs a built wheel into a fresh environment with only runtime requirements, excluding funcprops and semialg. `tests/packaging/core_runtime.py` checks import provenance, compact results, simplex, SOS replay, exact endpoints, quadratic proofs and safe refusal of unsupported real-output analysis. Optional-analysis helpers load their backend when called, so missing optional packages do not prevent core import.

## Exact LP and QP oracles

`tests/methods/test_polyhedral_oracles.py` enumerates rational stationary points and vertices on small compact polytopes using independent matrix equations. Numerical LP and convex-QP drivers are compared with those exact values, and primal feasibility is checked separately. Equivalent cases scale and reorder constraints, reverse coordinates and scale the objective by a positive factor. Equality slices, redundant rows, singular PSD matrices and zero quadratic forms are included. Native nonconvex face solutions must equal the oracle; limited spatial bounds must enclose it and pass replay.

## Dependency floors and installed wheels

`.github/workflows/dependencies.yml` tests the runtime floors in `tests/constraints-minimum.txt` on Python 3.11/Linux. A metadata check keeps those pins equal to the declared lower bounds. Latest-dependency core jobs cover Python 3.11 and 3.14 on Linux, macOS and Windows. A separate latest optional-analysis job runs the supported suite. Python 3.11 is used for the floor profile because older NumPy and SciPy releases do not provide wheels for every newer Python.

The workflow builds a wheel, installs it without an editable checkout, checks the resolved dependency environment and records versions plus JUnit results. `tools/test_wheel.py` stages tests, examples and documentation into a temporary directory without `src`, clears inherited `PYTHONPATH`, disables pytest's source-path insertion, and verifies package provenance before testing. The core profile includes the LP/QP oracle suite, certificate mutations, interval arithmetic, SOS, native quadratics and executable Markdown examples. The supported profile includes optional function analysis and omits formatting-only convention checks. Run either profile in the environment where the wheel is installed:

```bash
python tools/test_wheel.py --suite core --junitxml=wheel-core.xml
python tools/test_wheel.py --suite supported --junitxml=wheel-supported.xml
```

## Proof-field roles and mutations

Field inventories are grouped by owning family: SOS, univariate, native quadratic, and the nested/preordering/enclosure records in `tests/methods/certification/test_replay_families.py`. Inventories distinguish statement, domain premise, witness, coverage, derived summary, provenance and administrative data. A new unclassified field fails with the omitted field name. The new cases mutate source premises and several fields coherently, so matching flags or matching fabricated endpoints cannot replace replay of the mathematical obligations.

`VerifiedBound` and `IntervalResult` are producer-generated enclosures with rigor metadata, not self-contained replay certificates. Their inventory marks that trust boundary explicitly. Replayable native spatial certificates instead include the domain coverage, node partition and rational leaf evidence. Pipeline summaries collect evidence; editing their status fields is not an independent proof.
