# Quickstart

Start with a value and coordinate mapping. Use an explicit result API when you need attainment, bounds or proof diagnostics.

```python
import sympy as sp
from symbopt import minimize, minimize_result, minval, argmin, maxval, argmax

x, y = sp.symbols("x y", real=True)
assert minimize((x - 3) ** 2, x >= 0, variables=x) == [0, {x: 3}]
result = minimize_result((x - 3) ** 2, x >= 0, variables=x)
assert result.certified and result.attained is True
assert result.optimum_value == 0
assert result.best_candidate.original_variable_values == {x: 3}
```

## Values, points and limits

```python
assert minval((x - 3) ** 2, x >= 0, variables=x) == 0
assert argmin((x - 3) ** 2, x >= 0, variables=x) == ({x: 3},)
assert maxval(-((x - 1) ** 2), variables=x) == 0
assert argmax(-((x - 1) ** 2), variables=x) == ({x: 1},)
assert minval(x, x > 0, variables=x) == 0
assert argmin(x, x > 0, variables=x) == ()
```

The last problem has an infimum and no minimizing point. Compact results can report a limiting coordinate; the detailed `attained` field distinguishes it from a feasible optimizer.

## Constrained and parameterized models

```python
result = minimize_result(x * x + y * y, sp.Eq(x + y, 1), variables=(x, y))
assert result.certified and result.optimum_value == sp.Rational(1, 2)
assert result.best_candidate.original_variable_values == {
    x: sp.Rational(1, 2),
    y: sp.Rational(1, 2),
}
a = sp.Symbol("a", positive=True)
assert minval(a * (x - 2) ** 2, variables=x) == 0
```

Explicit decision variables keep `a` as a parameter. Restricted quadratic families have symbolic support; general parameter-region optimization can remain unresolved.

## Inspect the evidence

```python
explanation = result.explain()
assert isinstance(explanation, str) and result.method
assert (
    result.certificate.lower_bound
    == result.certificate.upper_bound
    == sp.Rational(1, 2)
)
```

A simpler exact route can finish before KKT or SOS work is needed, leaving those fields empty. Inspect the mathematical value and proof status before relying on route diagnostics. See {doc}`understanding-results`, {doc}`../guide/solver-selection` and {doc}`../examples/applications`.
