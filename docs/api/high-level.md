# High-level optimization API

The high-level API consists of `minimize()`, `maximize()`, value-only helpers,
and optimizer-set helpers. These functions share the same structure-aware and
certification-aware pipeline.

## `minimize`

```{autofunction} symbopt.minimize
```

A typical call is:

```python
import sympy as sp
from symbopt import minimize

x = sp.symbols("x", real=True)
result = minimize((x - 2)**2, [x >= 0], variables=[x])

result.optimum_value
result.attained
result.certified
result.best_candidate
result.certificate
```

Prefer `result.optimum_value` for the global value. `best_candidate` is a
represented feasible optimizer candidate and can be `None` when a certified
infimum is not attained.

## `maximize`

```{autofunction} symbopt.maximize
```

`maximize()` returns the same `PipelineResult` model as `minimize()`. The result
records `optimization_sense="maximize"`. When KKT equations were needed,
`kkt_objective` records the normalized minimization objective that generated the
stored KKT system, so there is no ambiguity between the user's objective and the
internal sign convention.

## Value-only helpers

`minval()` returns a certified infimum and `maxval()` returns a certified
supremum. They do not require attainment. If the pipeline cannot certify a
global value, the helpers return `None` rather than presenting a local candidate
as global.

```{autofunction} symbopt.minval
```

```{autofunction} symbopt.maxval
```

For example:

```python
x = sp.symbols("x", real=True)
minval(x, [x > 0], variables=[x])
# 0
```

## Optimizer helpers

`argmin()` and `argmax()` return tuples of represented optimizer mappings. A
certified but unattained infimum/supremum yields an empty tuple.

```{autofunction} symbopt.argmin
```

```{autofunction} symbopt.argmax
```

Thus:

```python
argmin(x, [x > 0], variables=[x])
# ()
```

## Decision variables and parameters

Use `variables=[...]` when not every free symbol is a decision variable:

```python
a, x = sp.symbols("a x", positive=True)
result = minimize(a * (x - 2)**2, variables=[x])
```

Here `x` is optimized and `a` remains a symbolic parameter. When `variables` is
omitted, all free symbols are treated as decision variables.

## Precision

`working_precision=N` is a package-level precision request and provenance
contract. Exact routes remain exact. High-precision inexact input is not
silently sent through float64-only LP/QP/SOS fast paths; the dispatcher chooses
precision-preserving symbolic/numerical or verified-interval routes instead.

Inspect:

```python
result.working_precision
result.precision_provenance
```

for the requested precision, detected input precision, effective routing policy,
and backend limitations.

## Lazy KKT construction

`PipelineResult.kkt_system` is optional. If an LP/QP, structural route, or exact
backend solves the problem before KKT search, no KKT system is built. This avoids
unnecessary symbolic differentiation and is expected behavior, not missing
diagnostic information.

See {doc}`../guide/solver-selection`, {doc}`../getting-started/understanding-results`,
and {doc}`../guide/certification`.
