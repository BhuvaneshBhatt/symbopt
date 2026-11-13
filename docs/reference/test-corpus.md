# Executable reference corpus

This corpus is used to check mathematical coverage across ordinary functions, special functions, multivariate constraints, branches, geometric regions and the public extremum wrappers. A passing core suite alone does not establish coverage of this collection.

The corpus has **779 distinct cases**: 424 univariate problems, 291 main optimization cases, four piecewise problems and 60 documentation examples. Each mathematical problem has a descriptive, pytest parameter ID.

## Run the cases

```bash
python -m pip install -e ".[test,funcprops]"
python -m pytest -q tests/reference/corpus
```

The smaller core profile is:

```bash
python -m pytest -q -m "not unsupported and not reference_corpus"
```

The dedicated CI corpus job runs the full collection. The repository includes its JSON data. The tests need the installed package and its dependencies, and reconstruct the problem using SymPy expressions.

## Mathematical assertions

| Mathematical behavior | SymPy test representation |
| --- | --- |
| Minimum or maximum | Objective, original constraints, explicit decision variables and expected value; exact references require certification |
| Multiple acceptable optimizers | Compare the optimum and verify a returned real feasible witness; equivalent optimizers are accepted |
| Value-only query | Call `minval` or `maxval` and compare the mathematical value |
| Argument-only query | Call `argmin` or `argmax`, verify feasibility and objective value; nonexistent optimizers use the empty tuple contract |
| Strict boundary or unattained limit | Check attainment separately from the extended-real infimum or supremum |
| Infeasible problem | Check infeasibility and value convention separately from unboundedness |
| Numerical reference | Compare numerical values with a stated tolerance and verify primal residuals |
| Fixed algebraic branch | Use `CRootOf` with the corresponding root index |
| Root given by an equation and numerical isolating hint | Retain both the defining equation and the high-precision hint; matching numerical witnesses must satisfy the equation |
| Parameter-dependent or triangular algebraic branch | Retain the polynomial lambdas and branch indices as formal `algebraic_root` or `triangular_algebraic_root` expressions |
| Function absent from SymPy | Keep a named symbolic function with its mathematical arguments; this records an executable capability target, without adding a numerical implementation |
| Region membership or distance | Expand disks, spheres, simplices, rectangles, cylinders and cones into scalar coordinate constraints |
| Region bounds | Solve the corresponding coordinate minimum and maximum queries |
| Seeded local optimization | Call `minimize_local` from each starting point and check the local value and stationarity |
| Diagnostic formatting and execution budgets | Check the mathematical value, feasibility, attainment or refusal behavior |

The geometric translations flatten vector variables and preserve radii, segment endpoints and boundary conditions. Signed distance inside a rectangle or triangle is the negative minimum distance to its boundary. A circumsphere example becomes equal-distance equations with a nonnegative radius. The region-construction example checks the defining predicate at interior, boundary and exterior points.

The rounding expressions use nearest-integer ties to even. Signed fractional part uses truncation toward zero, including negative inputs. Two-argument error functions, incomplete and regularized beta/gamma functions, generalized polynomial functions and hypergeometric functions retain their mathematical argument order.

Numerical reference comparisons use relative-scaled tolerance `1e-9`; primal equalities use absolute tolerance `1e-8`, and strict inequalities retain their strict sign. Root equations are checked using relative-scaled residual tolerance `1e-6`. Numerical agreement is recorded separately from exact global certification. High-precision requests are forwarded to the solver; decimal precision metadata alone is not treated as a portable certificate.

## Solver profile

The bounded regression profile disables SOS search, limits active sets to 16 and interval boxes to 12, and requests interval tolerance `1e-7`. Precision requests stored with a case are forwarded. This tests a reproducible portfolio configuration; an expected failure does not prove that every other solver configuration fails on the problem.

## Capability status

Supported references execute their mathematical assertions. Known unsupported references execute the same assertions under strict expected-failure marks. A new pass then fails CI as XPASS so the capability record must be updated.

Resource-limited cases also execute. Their solver process is terminated after the case budget: three seconds for capability probes and ten seconds for supported regressions. A budget expiry establishes only that the configured run did not finish. Such a case can pass on a faster or improved backend without causing a strict timing-dependent XPASS. Unknown exception types and worker failures remain visible as test failures.

All statuses live alongside the problem in `tests/reference/corpus/cases.json`; there is no separate covered-case map to drift out of sync. The manifest and inventory tests check every repository-owned case identity, reconstructed mathematical expression and category count.

Integer and mixed-integer test cases retain their mathematical targets and discrete variables. The default continuous profile exercises the explicit `NotImplementedError` rejection under expected-failure marks. Dedicated API tests require this rejection across continuous global, local and constrained-local entry points. Separate optional SCIP tests exercise integer and parity domains on quadratic models with numerical result semantics.

See {doc}`expected-failures` for the distinction between unsupported families, budget limits and oracle repairs, with the next coverage targets.
