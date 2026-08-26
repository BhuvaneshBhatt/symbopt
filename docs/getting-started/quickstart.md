# Quickstart

The high-level API is built around `minimize()` and `maximize()`, with concise
value and optimizer helpers.

```python
import sympy as sp
from symbopt import argmin, argmax, maximize, maxval, minimize, minval

x = sp.symbols("x", real=True)

result = minimize((x - 3)**2, [x >= 0], variables=[x])
```

Start by inspecting the global-value semantics:

```python
result.optimum_value
result.attained
result.certified
result.status
```

Then inspect represented candidates and proof data if needed:

```python
result.best_candidate
result.certificate
result.diagnostics
```

For this problem the optimizer is \(x=3\), the minimum is 0, and the optimum is
attained.

## Value-only and optimizer-only helpers

```python
minval((x - 3)**2, [x >= 0], variables=[x])
argmin((x - 3)**2, [x >= 0], variables=[x])

maxval(-(x - 1)**2, variables=[x])
argmax(-(x - 1)**2, variables=[x])
```

`minval()`/`maxval()` return certified infimum/supremum values. `argmin()` and
`argmax()` return represented optimizer mappings.

Attainment is separate from the value:

```python
minval(x, [x > 0], variables=[x])
# 0

argmin(x, [x > 0], variables=[x])
# ()
```

The strict problem has infimum 0 but no minimizer.

## Explicit decision variables and parameters

When an expression contains parameters, specify the decision variables:

```python
a = sp.symbols("a", positive=True)
result = minimize(a * (x - 2)**2, variables=[x])
```

Here `a` remains a parameter rather than being optimized over.

## A constrained example

```python
x, y = sp.symbols("x y", real=True)

result = minimize(
    x**2 + y**2,
    [sp.Eq(x + y, 1)],
    variables=[x, y],
)

result.optimum_value
```

The exact solution is

\[
x=y=\frac12,\qquad f^\star=\frac12.
\]

Depending on structure and installed backends, the result may be obtained by a
convex route, symbolic KKT/elimination, or exact semialgebraic reasoning. Do not
rely on a particular internal route unless you explicitly call a lower-level
solver.

## A bounded transcendental example

```python
t = sp.symbols("t", real=True)
result = minimize(
    sp.sin(t),
    [t >= 0, t <= 2 * sp.pi],
    variables=[t],
)
```

Common exact trigonometric forms can use structural analysis; otherwise supported
bounded transcendental expressions can fall back to verified interval methods.

## Inspect routing and provenance

KKT construction is lazy:

```python
result.kkt_system
```

may be `None` when an earlier route solved the problem. Useful provenance fields
include:

```python
result.structural_result
result.semialg_result
result.interval_result
result.sos_result
result.precision_provenance
```

Use {doc}`../guide/solver-selection` for routing rules and
{doc}`understanding-results` for result semantics.
