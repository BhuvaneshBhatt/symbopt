# Convex quadratic programs

Consider the unconstrained problem

\[
\min_x (x+2)^2.
\]

```python
import sympy as sp
from symbopt import minimize

x = sp.symbols("x", real=True)
result = minimize((x + 2)**2)
```

The direct QP route checks that the Hessian is positive semidefinite and keeps
the variable free, so the minimizer is

\[
x=-2,\qquad f^\star=0.
\]

For a linearly constrained example:

```python
x, y = sp.symbols("x y", real=True)
result = minimize(
    (x - 1)**2 + (y - 2)**2,
    [sp.Eq(x + y, 1)],
)
```

A nonconvex quadratic matrix is not sent to `solve_convex_qp()`.
