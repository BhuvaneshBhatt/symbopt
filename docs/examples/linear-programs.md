# Linear programs

Minimize

\[
2x+y
\]

subject to

\[
x+y\ge1,\qquad x\ge0,\qquad y\ge0.
\]

```python
import sympy as sp
from symbopt import minimize

x, y = sp.symbols("x y", real=True)

result = minimize(
    2*x + y,
    [
        sp.Ge(x + y, 1),
        sp.Ge(x, 0),
        sp.Ge(y, 0),
    ],
)
```

`minimize()` recognizes a linear objective and linear constraints and routes to
the general LP driver.

A separate equality example demonstrates that equalities are preserved:

```python
result = minimize(
    x,
    [sp.Eq(x, -2)],
)
```

The free symbolic variable can take the negative value \(-2\); no artificial
nonnegativity bound is imposed by the router.
