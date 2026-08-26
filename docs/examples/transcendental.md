# Verified transcendental optimization

Minimize

\[
\sin x
\]

on a bounded interval.

```python
import sympy as sp
from symbopt import interval_optimize

x = sp.symbols("x", real=True)

result = interval_optimize(
    sp.sin(x),
    [
        sp.Ge(x, 0),
        sp.Le(x, 2 * sp.pi),
    ],
    {x: (0.0, float(2 * sp.pi))},
    max_boxes=200,
    tol=1e-4,
)

print(result.best_lower_bound)
print(result.best_upper_bound)
print(result.lower_bound_rigorous)
print(result.interval_backend)
```

With Arb available, supported transcendental enclosures are rigorous.

A singular example such as

\[
1/x,\qquad x\in[-1,1]
\]

is not assigned a false finite enclosure. The evaluator marks the domain as
uncertain across zero so the branch-and-bound engine can subdivide around the
singularity.


## Structural analysis before intervals

Not every transcendental-looking problem needs branch-and-bound. `symbopt` first tries exact structural routes for monotone compositions, rationally related trigonometric polynomials, and common univariate rational/log/exp expressions with analyzable domains, poles, limits, and convexity. Verified intervals are the rigorous fallback for supported bounded expressions when those analytic routes do not apply. See {doc}`../methods/structural-routes`.
