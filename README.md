# symbopt

[GitHub](https://github.com/BhuvaneshBhatt/symbopt) ·
[Documentation](https://symbopt.readthedocs.io/) ·
[Tutorial notebook](https://github.com/BhuvaneshBhatt/symbopt/blob/main/notebooks/symbopt_tutorial.ipynb)

`symbopt` is a symbolic, certifying optimization package built on top of SymPy. It has exact symbolic reasoning alongwith numerical optimization, polynomial SOS/moment methods, semidefinite programming, algebraization, and verified interval branch-and-bound. It currently supports only continuous optimization, no discrete optimization.

Central design principle: **exploit mathematical structure first, use numerical methods to generate evidence when useful, and distinguish that evidence from rigorous global certification.**

## Installation

```bash
python -m pip install symbopt
```

Install the optional exact semialgebraic backend with:

```bash
python -m pip install "symbopt[semialg]"
```

For development:

```bash
git clone https://github.com/BhuvaneshBhatt/symbopt.git
cd symbopt
python -m pip install -e ".[test,docs,notebook,release]"
```

`symbopt` supports CPython 3.11–3.14. `python-flint` is a required dependency
and supplies the Arb backend used for rigorous transcendental enclosures.

## Quick start

```python
import sympy as sp
from symbopt import argmin, argmax, maximize, maxval, minimize, minval

x, y = sp.symbols("x y", real=True)

result = minimize(
    x**2 + y**2 - 2*x - 4*y,
    [
        sp.Le(x + y, 3),
        sp.Ge(x, 0),
        sp.Ge(y, 0),
    ],
    variables=[x, y],
)

print(result.optimum_value)
print(result.attained)
print(result.certified)
print(result.best_candidate)
```

For concise queries:

```python
minval((x - 2)**2, variables=[x])
argmin((x - 2)**2, variables=[x])
maxval(-(x - 1)**2, variables=[x])
argmax(-(x - 1)**2, variables=[x])
```

`minval()`/ `maxval()` return a certified infimum/supremum when available.
They do not require attainment. `argmin()` and `argmax()` return represented
optimizer mappings and return `()` for a certified but unattained extremum.

For example:

```python
minval(x, [x > 0], variables=[x])   # 0
argmin(x, [x > 0], variables=[x])   # ()
```

`minimize()` and `maximize()` return full `PipelineResult` containing the
certified value when known, attainment status, candidates, rigorous bounds,
solver diagnostics, and provenance.

## What symbopt can solve

| Problem family | Main route |
|---|---|
| Smooth symbolic problems | critical points, KKT / Karush--John, local classification |
| Linear programs | LP driver with exact postprocessing |
| Convex quadratic programs | convex-QP route |
| Linear-fractional programs | Charnes--Cooper with denominator-sign splitting |
| Polynomial global optimization | exact candidates, SOS/moments, intervals, optional semialg |
| Algebraic expressions | exact algebraization/lifting, including fixed algebraic numbers |
| `Abs` / `Piecewise` | branch decomposition |
| Monotone analytic compositions | reduction of `exp`, `log`, roots/powers, positive-base exponentials |
| Rational-frequency trig polynomials | exact tangent-half-angle reduction |
| Common univariate rational/log/exp problems | domain, pole, limit, convexity, and stationary-point analysis |
| Bounded supported transcendental problems | verified interval branch-and-bound with Arb |
| Independent-variable problems | certified separability decomposition |
| Sparse polynomial/SOS structure | support-aware bases and overlapping chordal SOS/SDP decomposition |
| Semidefinite subproblems | internal SDP backend |
| Exact semialgebraic certification | optional `semialg` CAD/QE/RUR backend |

Integer optimization and general parameter-stratified optimization are outside
current package scope.

## Decision variables, precision, and semialg policy

Use `variables=[...]` to distinguish optimization variables from parameters:

```python
a, x = sp.symbols("a x", positive=True)
result = minimize(a * (x - 2)**2, variables=[x])
```

`working_precision=N` records and propagates a decimal precision request.
High-precision inexact input is not silently routed through float64-only
LP/QP/SOS backends; `precision_provenance` records the routing decision and any
backend precision limits.

When `semialg` is installed, `semialg_certification` accepts `"auto"`,
`"candidate"`, and `"complete"`. In `"auto"` mode symbopt uses a cost-aware
policy rather than always invoking CAD first: cheap structural/LP/QP routes run
first, and small polynomial systems may use symbolic KKT/elimination before an
expensive exact CAD range computation. Explicit `"candidate"` or `"complete"`
requests force the corresponding backend policy when the installed semialg API
supports it.

## Certification semantics

A candidate is not automatically a proof. `symbopt` distinguishes:

- exact symbolic certification,
- rigorous interval/ball bounds,
- verified exact SOS identities/PSD recovery,
- heuristic or floating-point candidate evidence.

`result.optimum_value` is the preferred high-level value field. A
`best_candidate` may be absent when the exact infimum/supremum is unattained.
See the documentation for complete result and certificate semantics.

## Documentation

The full manual is hosted at <https://symbopt.readthedocs.io/>. Useful entry
points include:

- [Quickstart](https://symbopt.readthedocs.io/en/latest/getting-started/quickstart.html)
- [Understanding results](https://symbopt.readthedocs.io/en/latest/getting-started/understanding-results.html)
- [Solver selection](https://symbopt.readthedocs.io/en/latest/guide/solver-selection.html)
- [Structural optimization routes](https://symbopt.readthedocs.io/en/latest/methods/structural-routes.html)
- [Certification semantics](https://symbopt.readthedocs.io/en/latest/guide/certification.html)
- [Capability matrix](https://symbopt.readthedocs.io/en/latest/reference/capability-matrix.html)
- [Limitations](https://symbopt.readthedocs.io/en/latest/reference/limitations.html)

The executable long-form tutorial is
[`notebooks/symbopt_tutorial.ipynb`](https://github.com/BhuvaneshBhatt/symbopt/blob/main/notebooks/symbopt_tutorial.ipynb).

Build the documentation locally with:

```bash
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

## Testing

```bash
python -m pytest -q
```

The repository also contains numerical validation tests, property/regression
tests, reference cases, benchmarks, and an end-to-end notebook execution job.

## License

`symbopt` is licensed under GPL-3.0-only. See [LICENSE](LICENSE).

## Contributing

See the [contributing guide](https://symbopt.readthedocs.io/en/latest/development/contributing.html).
