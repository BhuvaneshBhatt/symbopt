# Unconstrained optimization

Consider

\[
f(x)=x^4-2x^2.
\]

```python
import sympy as sp
from symbopt import minimize_result, critical_points

x = sp.symbols("x", real=True)

points = critical_points(x**4 - 2 * x**2, [x])
result = minimize_result(x**4 - 2 * x**2)
```

The stationary equation is

\[
4x^3-4x=0,
\]

so \(x\in\{-1,0,1\}\). The points \(\pm1\) are minima with value \(-1\); \(x=0\) is a local maximum.

Inspect:

```python
result.optimum_value
result.attained
result.best_candidate.original_variable_values
result.certificate
```

Polynomial lower-bound methods may additionally certify global optimality.
