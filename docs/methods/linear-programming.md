# Linear programming

For a linear objective and linear constraints, `minimize()` can route directly to the LP driver.

## General LP driver

`solve_lp()` supports:

- linear inequalities,
- linear equalities,
- variable bounds,
- free variables when bounds `(None, None)` are supplied,
- infeasible/unbounded status reporting.

It uses SciPy's HiGHS backend for robust feasibility initialization and general continuous LP handling.

## Package simplex routines

`symbopt.simplex()` and `symbopt.revised_simplex()` expose low-level educational standard-form solvers for

\[
Ax\le b,\qquad x\ge0,
\]

assuming the initial slack basis is feasible. Both low-level routines minimize the supplied objective coefficients, matching `solve_lp()`. If it is not, they return `needs_phase_one` rather than pretending the basis is valid.

## Example

```python
import sympy as sp
from symbopt import minimize

x, y = sp.symbols("x y", real=True)
result = minimize(
    2 * x + y,
    [
        sp.Ge(x + y, 1),
        sp.Ge(x, 0),
        sp.Ge(y, 0),
    ],
)
```

The high-level router converts the symbolic linear system into the LP matrix representation while preserving the original constraint semantics.

## Further reading

See Bertsimas–Tsitsiklis and Schrijver in {doc}`../reference/references` for the mathematical background.

## Exact postprocessing

After a numerical LP optimum is found, the symbolic router identifies active constraints and attempts to solve the corresponding KKT system exactly. When this succeeds, the returned candidate uses exact coordinates and the certificate records the active constraints, exact feasibility checks, multipliers, and stationarity residuals. The direct route is globally rigorous only when exact reconstruction and exact original-constraint feasibility both succeed. A numerical feasibility fallback can support a numerical candidate but not an exact certificate. A failed reconstruction is recorded as a typed certificate diagnostics.
