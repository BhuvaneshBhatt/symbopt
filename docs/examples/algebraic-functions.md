# Algebraic functions

Minimize a problem containing a square root:

\[
f(x)=\sqrt{x+1}
\]

on \(x\ge-1\).

```python
import sympy as sp
from symbopt import algebraize, minimize

x = sp.symbols("x", real=True)

lifted = algebraize(
    sp.sqrt(x + 1),
    [sp.Ge(x, -1)],
)

print(lifted.lifted_objective)
print(lifted.all_constraints())

result = minimize(
    sp.sqrt(x + 1),
    [sp.Ge(x, -1), sp.Le(x, 3)],
)
```

The lift introduces an auxiliary variable representing the square root together
with the polynomial graph equation and correct real-branch condition.
