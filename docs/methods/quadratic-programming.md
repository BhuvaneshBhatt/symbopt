# Quadratic programming

A convex QP has a quadratic objective

\[
\frac12x^TQx+c^Tx
\]

with \(Q\succeq0\), together with linear equalities/inequalities and optional bounds.

`symbopt.solve_convex_qp()` symmetrizes \(Q\), checks positive semidefiniteness, and rejects a genuinely nonconvex matrix rather than routing it through the convex solver.

Variables are free by default.

## Example

```python
import sympy as sp
from symbopt import minimize

x = sp.symbols("x", real=True)
result = minimize((x + 2) ** 2)
```

The unconstrained minimizer is \(x=-2\), demonstrating that the direct QP route does not impose an artificial \(x\ge0\) bound.

## Limitations

For nonconvex objectives and quadratic constraints use {doc}`global-quadratics`: exact stationary-face enumeration for bounded rational QPs, native spatial branch-and-bound for rational QCQPs with a proved finite domain, and optional numerical SCIP for broader continuous and mixed-integer models.

## Further reading

See Boyd–Vandenberghe and Nocedal–Wright in {doc}`../reference/references` for the mathematical background.

## Exact postprocessing

The numerical convex-QP solution is followed by symbolic active-set KKT reconstruction. This can recover exact rational/algebraic coordinates and multipliers for problems whose numerical optimizer is only approximate. The proof artifact records exact feasibility and stationarity information when recovery succeeds. Exact global certification additionally requires exact feasibility of the original constraints, provably nonnegative inequality multipliers, and an exact convexity check; approximate numerical feasibility alone is not sufficient.
