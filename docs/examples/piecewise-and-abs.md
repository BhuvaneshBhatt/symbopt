# Piecewise and absolute-value problems

## Absolute value

```python
import sympy as sp
from symbopt import minimize

x = sp.symbols("x", real=True)
result = minimize(sp.Abs(x - 2))
```

The branch machinery can represent the sign cases explicitly.

## Piecewise

```python
objective = sp.Piecewise(
    (x**2, x <= 0),
    ((x - 1)**2 + 1, True),
)

result = minimize(
    objective,
    [sp.Ge(x, -2), sp.Le(x, 2)],
)
```

Each branch is algebraized/solved with its branch condition. The default
`True` branch is preserved, and the best feasible branch result is returned.
