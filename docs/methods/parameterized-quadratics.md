# Quadratics with symbolic parameters

Use explicit decision variables when other symbols represent parameters. This lets symbopt optimize a family of quadratics without assigning numerical values to its coefficients.

```python
import sympy as sp
from symbopt import minimize_result, argmin, maximize_result

x, y, a, b, c, lower = sp.symbols("x y a b c lower", real=True)
p = sp.Symbol("p", positive=True)
result = minimize_result(p * (x - a) ** 2 + c, variables=x)
assert result.certified and result.attained is True
assert result.optimum_value == c
assert result.best_candidate.primal_values == {x: a}
assert result.best_candidate.objective_value_numeric is None
assert argmin(p * (x - a) ** 2 + c, variables=x) == ({x: a},)
```

The same structural route supports coupled positive-semidefinite forms when it can prove a real stationary point valid throughout the parameters' assumptions. A numerical optimum is unnecessary: the quadratic expansion around that point proves the lower bound. Singular forms can supply a representative of a nonunique optimizer set. A negative diagonal curvature proves unboundedness for an unconstrained form.

```python
coupled = minimize_result((x - a) ** 2 + p * (y - x - b) ** 2 + c, variables=(x, y))
assert coupled.optimum_value == c
assert coupled.best_candidate.primal_values == {x: a, y: a + b}
maximum = maximize_result(-p * (x - a) ** 2 + c, variables=x)
assert maximum.optimum_value == c
```

## Scalar affine bounds

Affine constraints in one decision variable are normalized to endpoints. Their ordering and feasibility must be provable. For a convex quadratic, clipping its vertex to that interval gives its minimum.

```python
bounded = minimize_result(p * (x - a) ** 2 + c, x >= lower, variables=x)
vertex = sp.Max(a, lower)
assert bounded.best_candidate.primal_values == {x: vertex}
assert sp.simplify(bounded.optimum_value - (p * (vertex - a) ** 2 + c)) == 0
```

On a closed bounded interval, real quadratic coefficients can have unknown signs. The result can retain a Piecewise optimizer: positive curvature uses the clipped vertex; nonpositive curvature uses an endpoint. No division by the leading coefficient is used in its zero-curvature branch.

```python
family = minimize_result(a * x**2 + b * x + c, [x >= 0, x <= 1], variables=x)
assert family.certified and family.attained is True
assert family.optimum_value.subs({a: 1, b: -1, c: 0}) == -sp.Rational(1, 4)
assert family.optimum_value.subs({a: 0, b: -1, c: 0}) == -1
```

## Strict constraints and limits

For an open endpoint, the closure minimum can be an unattained infimum. If attainment depends on unresolved parameter inequalities, `attained` is None and the result supplies no uniformly feasible optimizer point.

```python
strict = minimize_result(p * (x - a) ** 2 + c, x > lower, variables=x)
assert strict.certified and strict.attained is None
assert strict.best_candidate is None
```

This is a restricted quadratic route, not general parameter-region optimization. It declines unknown curvature on an unbounded domain, unresolved feasibility such as independent lower and upper parameters with no known ordering, and general multivariate parameterized constraints. It also declines stationary formulas whose denominators may vanish for allowed parameter values. Parameters must make the coefficients real. Numerical candidate values remain None when the mathematical optimum still depends on parameters; exact values and certificate bounds retain their symbolic expressions.
