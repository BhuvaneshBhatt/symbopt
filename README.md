# symbopt

`symbopt` finds global extrema of SymPy expressions under real constraints. Use it when you need symbolic values, exact optimizer coordinates, open-boundary behavior or a global bound. A local search can miss another basin; `symbopt` combines structural rules, algebraic methods, numerical proposals and proof checks to decide what each result establishes. It currently supports only continuous optimization, no discrete optimization.

Typical applications include fitting polynomial models, allocating resources with linear or quadratic costs, bounding nonlinear designs and checking proposed polynomial inequalities. The [application tutorial](https://github.com/BhuvaneshBhatt/symbopt/blob/main/docs/examples/applications.md) develops these tasks with executable examples.

## Installation

```sh
python -m pip install symbopt
```

Python 3.11 or newer is required. SymPy provides symbolic arithmetic, SciPy and NumPy provide numerical proposals, and python-flint provides Arb enclosures. Optional backends add function analysis, semialgebraic reasoning and numerical mixed-integer quadratic optimization:

```sh
python -m pip install 'symbopt[funcprops,semialg]'
python -m pip install 'symbopt[scip]'
```

## Values and optimizing points

```python
import sympy as sp
from symbopt import minimize, maximize, minimize_result, minval, argmin

x, y = sp.symbols("x y", real=True)
assert minimize((x - 2) ** 2, x >= 0, variables=x) == [0, {x: 2}]
assert maximize(
    3 * x + 2 * y, [x >= 0, y >= 0, x + y <= 4, x <= 2], variables=(x, y)
) == [10, {x: 2, y: 2}]
assert minval(x**4 - 2 * x**2, variables=x) == -1
assert argmin(x**4 - 2 * x**2, variables=x) == ({x: -1}, {x: 1})
```

`minimize` and `maximize` return plain `[value, optimizer]` lists. The optimizer is a coordinate mapping or a tuple of mappings for represented isolated optima. Omitted coordinates in a represented partial mapping are free. An empty mapping can also indicate that no optimizer was represented; detailed attainment distinguishes these cases. Value-only helpers return a proved infimum or supremum when available; `argmin` and `argmax` return represented attained optimizers. A complete description of every positive-dimensional optimizer set is not always available.

## An infimum can be unattained

`minval()`/ `maxval()` return a certified infimum/supremum when available.
They do not require attainment. `argmin()` and `argmax()` return represented
optimizer mappings and return `()` for a certified but unattained extremum.

```python
result = minimize_result(x, x > 0, variables=x)
assert result.certified and result.optimum_value == 0
assert result.attained is False
assert argmin(x, x > 0, variables=x) == ()
assert minimize(x, x > 0, variables=x) == [0, {x: 0}]
```

The last compact mapping identifies a limiting coordinate, not a feasible minimizer. Detailed results distinguish attainment, infeasibility, unboundedness and unfinished work. Use `minimize_result`, `maximize_result` or `full_output=True` to inspect candidates, bounds, proof objects, diagnostics and the solver plan. If the pipeline cannot prove a global value, `minval` returns `None`; a numerical best candidate can still be inspected in the detailed result.

Constraints may be relations, `And`, `Or` or finite iterables. `Or` denotes alternative feasible regions. Supply `variables=x` or `variables=(x, y)` to distinguish decisions from parameters. `bounds={x: (0, 1)}` adds actual domain constraints; `box_bounds` controls interval search. Symbol sign assumptions are used when `use_symbol_assumptions=True`. Integer decision symbols require the explicitly selected optional quadratic SCIP backend.

## Capabilities and limits

| Task | Supported methods | Boundary of the claim |
| --- | --- | --- |
| Fixed univariate polynomial and rational extrema | Exact critical points, poles, exclusions, boundary values and limits | Exact coefficient isolation and comparisons must complete |
| Linear and convex quadratic programs | Numerical LP/QP proposals with exact feasibility and KKT reconstruction | Numerical solver success alone is not an exact proof |
| Nonconvex quadratic models | Exact stationary faces, rational spatial bounds and optional SCIP | Native methods need supported rational continuous domains; SCIP bounds are numerical |
| Parametrised quadratics | Selected PSD stationary forms and scalar affine intervals | General parameter-region stratification is unsupported |
| Polynomial global optimization | Structural bounds, SOS, moments, finite sublevel boxes and optional `semialg` | The portfolio is not a complete solver for every polynomial problem |
| Algebraic, Abs and Piecewise expressions | Source-preserving graph lifts, owned branches and exact reductions | Unsupported branch geometry or unresolved feasibility can leave partial results |
| Periodic and selected elementary expressions | Rational trigonometric reductions, independent periodic components, domain and limit analysis | General transcendental global optimization is unsupported |
| Finite-box analytic optimization | Verified interval branch-and-bound with Arb | Function grammar, finite domain, precision and search budgets limit completion |
| Local optimization and critical points | Stationarity, KKT, critical cones and numerical local search | Local classification / candidates do not establish a global optimum, obviously |
| Integer optimization | Explicit optional SCIP for quadratic models | No general exact integer optimization route |

The [full capability matrix](https://github.com/BhuvaneshBhatt/symbopt/blob/main/docs/reference/capability-matrix.md) and [limitations](https://github.com/BhuvaneshBhatt/symbopt/blob/main/docs/reference/limitations.md) state the supported cases. A finite work budget can leave a supported problem unresolved. An unsupported or unknown result does not establish infeasibility.

## Evidence and package roles

`symbopt` owns optimization strategy, value and attainment decisions, and optimizer certificates. [funcprops](https://github.com/BhuvaneshBhatt/funcprops/blob/main/docs/ranges.md) supplies domains, roots, shape and image information, while semialg supplies supported real-algebraic projection and reasoning. `funcprops` does not require `symbopt`. An exact image contains attainable values, while an interval enclosure may contain missing values. See [ranges and extrema](https://github.com/BhuvaneshBhatt/symbopt/blob/main/docs/guide/ranges-and-extrema.md).

Numerical candidates, rigorous interval bounds and exact global certificates have separate statuses. An SOS certificate replays its polynomial identity, Gram matrices, basis, domain multipliers and bound; cached flags cannot establish verification. Exact interval endpoints are rounded outward before numerical search. A tolerance-screened proposal cannot establish an exact feasible witness; original constraints and the objective value are checked again before primal proof claims. A positive optimization gap is a bound, not an exact optimum. The [certification guide](https://github.com/BhuvaneshBhatt/symbopt/blob/main/docs/guide/certification.md) explains these contracts.

`working_precision` records a precision request and backend provenance. High-precision inexact input bypasses float64-only fast paths; it does not make every numerical backend arbitrary precision. Optional `semialg` policy accepts `auto`, `candidate` and `complete`, with cost-aware dispatch in automatic mode. See [precision](https://github.com/BhuvaneshBhatt/symbopt/blob/main/docs/reference/precision.md) and [solver selection](https://github.com/BhuvaneshBhatt/symbopt/blob/main/docs/guide/solver-selection.md).

## Learn and look up

- [Quickstart](https://github.com/BhuvaneshBhatt/symbopt/blob/main/docs/getting-started/quickstart.md), [problem formulation](https://github.com/BhuvaneshBhatt/symbopt/blob/main/docs/getting-started/problem-formulation.md) and [understanding results](https://github.com/BhuvaneshBhatt/symbopt/blob/main/docs/getting-started/understanding-results.md).
- [Application examples](https://github.com/BhuvaneshBhatt/symbopt/blob/main/docs/examples/applications.md), [tutorial notebook](https://github.com/BhuvaneshBhatt/symbopt/blob/main/notebooks/symbopt_tutorial.ipynb) and [documentation index](https://github.com/BhuvaneshBhatt/symbopt/blob/main/docs/index.md).
- [Optimization over regions](https://github.com/BhuvaneshBhatt/symbopt/blob/main/docs/guide/regions.md): pass `region=` for SymPy sets, Boolean formulas or optional `semialg` geometry; combine membership with constraints and bounds, reuse prepared regions, reduce covering parametrisations and compute exact optimizer-set formulas or parameter-family relations.
- [Polynomial and rational methods](https://github.com/BhuvaneshBhatt/symbopt/blob/main/docs/methods/polynomial-rational.md), [global quadratics](https://github.com/BhuvaneshBhatt/symbopt/blob/main/docs/methods/global-quadratics.md), [univariate analysis](https://github.com/BhuvaneshBhatt/symbopt/blob/main/docs/guide/univariate-analysis.md) and [SOS proofs](https://github.com/BhuvaneshBhatt/symbopt/blob/main/docs/methods/sum-of-squares.md)
- [Public API](https://github.com/BhuvaneshBhatt/symbopt/blob/main/docs/api/index.md), [known issues](https://github.com/BhuvaneshBhatt/symbopt/blob/main/docs/reference/known-issues.md), [testing](https://github.com/BhuvaneshBhatt/symbopt/blob/main/docs/development/testing.md)

## Development

```sh
python -m pip install -e '.[test,docs,lint,release]'
python -m pytest -q -m 'not unsupported and not reference_corpus'
python -m pytest -q tests/reference/corpus
ruff check .
ruff format --check .
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

The test suite includes exact mathematical oracles, certificate mutations, numerical differential checks, executable documentation and a bounded reference corpus. The [testing guide](https://github.com/BhuvaneshBhatt/symbopt/blob/main/docs/development/testing.md) distinguishes supported checks, expected failures and resource-limited cases.

## License

GNU General Public License, GPL-3.0-only. See [LICENSE](https://github.com/BhuvaneshBhatt/symbopt/blob/main/LICENSE).
