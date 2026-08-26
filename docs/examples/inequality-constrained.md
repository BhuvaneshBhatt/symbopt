# Inequality-constrained optimization

Minimize

\[
f(x)=x
\]

subject to

\[
x\ge1.
\]

```python
import sympy as sp
from symbopt import critical_points, minimize

x = sp.symbols("x", real=True)
constraints = [sp.Ge(x, 1)]

points = critical_points((x, constraints), [x])
result = minimize(x, constraints)
```

The solution lies on the boundary \(x=1\). Under the package convention

\[
h(x)=x-1\ge0,
\]

the Lagrangian is

\[
L=x-\mu(x-1),\qquad \mu\ge0.
\]

Stationarity gives \(\mu=1\), and complementary slackness is satisfied.
