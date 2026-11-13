# Polynomial and rational minima and infima

Use these routes when you need to distinguish a proved attained minimum from an excluded boundary value or a limit at infinity. A candidate from numerical solving alone does not settle that distinction.

## Exact univariate optimization

For fixed rational or real algebraic coefficients, symbopt isolates all real roots of the derivative numerator, denominators, and constraint boundary polynomials. It checks each boundary point and open interval exactly. On each feasible interval, endpoint limits and critical values determine the infimum. A failed root isolation or comparison leaves this route unresolved.

```python
import sympy as sp
from symbopt import minimize_result, rational_graph, polynomial_sublevel_bounds

x, y, t = sp.symbols("x y t", real=True)
minimum = minimize_result((x * x - 1) ** 2, variables=x)
assert minimum.certified and minimum.attained is True
assert minimum.optimum_value == 0
assert {c.primal_values[x] for c in minimum.candidates} == {-1, 1}

limit = minimize_result(1 / (1 + x * x), variables=x)
assert limit.certified and limit.optimum_value == 0
assert limit.attained is False
```

Algebraic critical points can remain root objects; radical expressions are unnecessary. Rational constraints, isolated feasible points, repeated critical roots and open endpoints are included. Parameter families and inexact coefficients use other solver routes.

## Rational graphs and exact images

`rational_graph(p/q, constraints, variables, t)` returns a polynomial formula imposing the constraints, the nonzero-denominator conditions and `t*q == p`. Inequalities are converted without assuming a denominator sign. Input denominator exclusions survive internal cancellation. If an expression has already simplified away its denominator before the call, supply the excluded points explicitly.

```python
graph = rational_graph(1 / x, [], x, t)
assert graph.subs({x: 2, t: sp.Rational(1, 2)}) is sp.true
assert graph.subs({x: 0, t: 0}) is sp.false
```

For multivariate rational objectives, the optional semialg backend can eliminate the decision variables from this graph. Optimization of the resulting one-dimensional image determines the exact infimum and whether that value belongs to the image. Optimizer witnesses are recovered and checked when possible; a certified attained value can lack an explicit point if witness recovery fails.

Automatic graph projection has a degree/dimension cost budget. `semialg_certification="complete"` removes the default estimate limit, while `semialg_range_cost_limit` can supply a specific limit. This requests the exact route; it does not promise that the installed backend can eliminate every graph. `"candidate"` does not request this projection.

## Certified sublevel boxes

A feasible point with value U supplies an upper bound on the infimum. If a proof shows that f(x)>U outside a finite box, searching that box retains every point that could improve the upper bound.

```python
polynomial = x**4 + y**4 + x * y - 3 * x + 2 * y + 7
bounds = polynomial_sublevel_bounds(polynomial, (x, y), upper_bound=7)
left, right = bounds[x]
assert bounds[y] == (left, right)
assert left < 0 < right
```

The current sufficient test requires an even total degree and positive pure-power margins in the leading homogeneous form. Indefinite mixed leading terms are bounded by weighted arithmetic-geometric means; nonnegative even leading terms can be dropped. Lower-degree terms are bounded by their absolute coefficients. Exact powers of two provide box endpoints that the interval engine can represent without rounding inward.

The automatic dispatcher independently checks a feasible upper-bound point, attaches the exterior-bound proof to the result and enables interval search on the resulting box. The bound does not itself certify an optimum. `None` means that this sufficient test did not apply; it does not establish unboundedness. Coercive polynomials with unequal coordinate degrees or a leading form outside the supported dominance test can remain unresolved.

## Capability boundaries

| Input | Coverage |
| --- | --- |
| One-variable rational or polynomial data over fixed real algebraic coefficients | Complete critical/boundary enumeration when exact root isolation and comparisons succeed |
| Multivariate rational objective | Certified graph projection where semialg resolves the image within the selected budget |
| Polynomial with supported positive leading margins | Certified sublevel box; global proof still depends on the subsequent solver |
| General parameter-dependent coefficients | Restricted quadratic routes only; no general parameter stratification |
| General noncompact polynomial or rational problem | No complete analysis of all escaping curves or asymptotic critical values |

See {doc}`../guide/certification` for the distinction between exact global proof and rigorous bounds with a nonzero gap.
