# symbopt documentation

`symbopt` is a symbolic and certifying continuous-optimization package built on
SymPy. Its high-level `minimize()` function combines symbolic critical-point
analysis, constrained KKT/Karush--John reasoning, linear and convex quadratic
optimization, polynomial SOS/moment methods, algebraization of selected
nonpolynomial expressions, and verified interval branch-and-bound.

The package distinguishes **candidate generation** from **global
certification**. A numerical solution is useful evidence, but it is not called
a proof unless the package has a rigorous proof path: for example an exact SOS
identity or a verified interval enclosure.

## Start here

- {doc}`getting-started/installation` — install the package and verify the Arb backend.
- {doc}`getting-started/quickstart` — solve a first problem with `minimize()`.
- {doc}`getting-started/problem-formulation` — formulate objectives, constraints, and bounds.
- {doc}`getting-started/understanding-results` — interpret candidates, bounds, and certificates.
- {doc}`guide/solver-selection` — understand how `minimize()` chooses a method.
- {doc}`guide/certification` — understand exact, rigorous numerical, and heuristic results.
- {doc}`reference/capability-matrix` — see what problem classes are supported.
- {doc}`reference/limitations` — understand the current boundaries of the package.

The executable tutorial notebook is kept in
[`notebooks/symbopt_tutorial.ipynb`](https://github.com/BhuvaneshBhatt/symbopt/blob/main/notebooks/symbopt_tutorial.ipynb).

```{toctree}
:maxdepth: 2
:caption: Getting started

getting-started/installation
getting-started/quickstart
getting-started/problem-formulation
getting-started/understanding-results
```

```{toctree}
:maxdepth: 2
:caption: User guide

guide/solver-selection
guide/certification
guide/exact-vs-numerical
guide/assumptions-and-domains
guide/performance
```

```{toctree}
:maxdepth: 2
:caption: Methods

methods/overview
methods/critical-points
methods/kkt-karush-john
methods/algebraization
methods/elimination
methods/linear-programming
methods/quadratic-programming
methods/structural-routes
methods/semidefinite-programming
methods/sum-of-squares
methods/moment-methods
methods/sparse-structure
methods/degeneracy
methods/verified-intervals
```

```{toctree}
:maxdepth: 2
:caption: Worked examples

examples/index
examples/high-level-api
examples/unconstrained
examples/equality-constrained
examples/inequality-constrained
examples/linear-programs
examples/quadratic-programs
examples/polynomial-global
examples/sos-certificates
examples/moment-extraction
examples/algebraic-functions
examples/piecewise-and-abs
examples/transcendental
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/capability-matrix
reference/limitations
```


```{toctree}
:maxdepth: 2
:caption: API reference

api/index
```

```{toctree}
:maxdepth: 2
:caption: Detailed reference

reference/numerical-tolerances
reference/precision
reference/performance
reference/references
```

```{toctree}
:maxdepth: 2
:caption: Development

development/architecture
development/testing
development/contributing
development/documentation
```
