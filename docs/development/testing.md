# Testing and validation

The test suite is organized primarily by **solver method** and, for broad
reference regressions, by **mathematical input type**. This makes a failure point
directly to the method or input family that needs attention.

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
│   │   ├── test_future_transcendental.py
│   │   ├── test_future_integer.py
│   │   ├── test_future_algebraic.py
│   │   └── ...
│   └── test_reference_case_inventory.py
├── api/
├── infrastructure/
├── validation/
└── tooling/
```

## Runtime guard

The test environment includes `pytest-timeout` with a **60-second per-test
limit**. A test that exceeds one minute is treated as a test-design or
performance regression and fails rather than hanging the suite. Slow benchmark
work belongs in `benchmarks/`, not in ordinary pytest cases.

## Organization

**Method tests**
: Live under `tests/methods/<method>/`. Unit, integration, regression, and
  property-based checks for the same solver family stay together. Tests that
  require optional backends retain the `integration` marker.

**Reference tests**
: Live under `tests/reference/by_input_type/` and are grouped by the mathematical
  form of the problem rather than by where the case originated. Supported cases
  execute normally; unsupported cases retain `future` + `xfail(run=False)`.

**API and infrastructure tests**
: Cover result models, diagnostics, shared caches, normalization helpers, and
  other code that is not itself an optimization method.

**Validation and tooling tests**
: Keep the independent numerical validation battery, documentation checks,
  public-docstring checks, and benchmark infrastructure separate from solver
  correctness tests.

## Pytest markers

The suite declares these markers:

```text
integration
validation
property
reference
future
slow
```

Useful development commands include:

```bash
python -m pytest -q -m "not future"
python -m pytest -q -m integration
python -m pytest -q -m property
python -m pytest -q -m reference
python -m pytest -q -m validation
python -m pytest -q -m "not slow and not future"
```

To inspect the documented future cases without running them:

```bash
python -m pytest --collect-only -q -m future
```

## Backend requirements

Release CI installs the package's test extras. Arb-backed and semialg-backed
checks are integration tests, so missing optional/runtime backends are visible
as integration-environment problems rather than being hidden among unit tests.

## Numerical validation battery

The package exposes:

```python
from symbopt import run_validation_suite

results = run_validation_suite()
```

The validation test verifies that the known SDP/SOS numerical cases remain
green independently of ordinary unit tests.

## Documentation validation

Documentation CI builds Sphinx with warnings treated as errors:

```bash
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

The repository also verifies required pages, MyST targets, and stable public API
documentation coverage.

## Expectations for new changes

- A solver route should have success, infeasible/unsupported, and regression
  cases.
- A correctness bug should receive a minimal regression test.
- A new backend call should have both adapter-level and live integration tests.
- A new rigorous certificate route should verify that heuristic evidence cannot
  trigger rigorous status.
- A new public API symbol should be documented and covered by API tests.
- A newly discovered unsupported but valuable problem can be added to the
  `future` reference suite with a precise reason.
## Large reference-case corpus

The reference suite also maintains a case-level inventory for a 291-case
external optimization regression corpus. Every case number is represented in
`tests/reference/by_input_type/reference_case_inventory.json`. Cases that match a supported,
fast mathematical core point to the ordinary regression that exercises them.
Cases that are not yet both supported and suitable for the normal runtime budget
are collected individually under the `future` marker with `xfail(run=False)`.

Precision-only and value/argument-wrapper variants retain a `variant_of` mapping
so they remain traceable without pretending they add an independent solver
capability. This keeps the capability backlog complete while ensuring unsupported
or very expensive cases cannot hang the default test suite.

