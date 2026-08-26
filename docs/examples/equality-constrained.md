# Equality-constrained optimization

Minimize

\[
x^2+y^2
\]

subject to

\[
x+y=1.
\]

```python
import sympy as sp
from symbopt import minimize, critical_points

x, y = sp.symbols("x y", real=True)
constraints = [sp.Eq(x + y, 1)]

points = critical_points((x**2 + y**2, constraints), [x, y])
result = minimize(x**2 + y**2, constraints)
```

The constrained stationary point is

\[
x=y=\frac12.
\]

The tangent space to \(x+y=1\) is one-dimensional. The reduced Hessian is
positive, so the point is a constrained local minimum.

For this convex quadratic problem the high-level pipeline can also recognize
convex structure.
