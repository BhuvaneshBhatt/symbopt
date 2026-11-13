# High-level optimization API

The high-level API consists of `minimize()`, `maximize()`, value-only helpers, and optimizer-set helpers. These functions share the same structure-aware and certification-aware pipeline.

## `minimize`

```{autofunction} symbopt.minimize
```

A typical call is:

```python
import sympy as sp
from symbopt import minimize, minimize_result, minval, argmin, maxval, argmax

x = sp.symbols("x", real=True)
result = minimize((x - 2) ** 2, x >= 0, variables=x)
# [0, {x: 2}]

full = minimize((x - 2) ** 2, x >= 0, variables=x, full_output=True)
full.optimum_value
full.attained
full.certified
full.method
full.explain()
```

Constraints may be a relation, `And`, `Or`, or an iterable. `Or` is optimized branch-by-branch. Use `bounds={x: (lower, upper)}` for compact bound syntax.

Use `region=` for a SymPy set, Boolean formula or optional semialg region, intersected with the other constraints. Coordinate-free inputs require `variables=`; symbolic regions retain their owned coordinate order. All global value and optimizer helpers forward this keyword. See {doc}`../guide/regions` for executable examples and conversion limits.

## `maximize`

```{autofunction} symbopt.maximize
```

`maximize()` uses the same compact result convention as `minimize()`. Request `full_output=True` for the `PipelineResult`, which records `optimization_sense="maximize"`. When KKT equations were needed, `kkt_objective` records the normalized minimization objective that generated the stored KKT system, so there is no ambiguity between the user's objective and the internal sign convention.

## Value-only helpers

`minval()` returns a certified infimum and `maxval()` returns a certified supremum. They do not require attainment. If the pipeline cannot certify a global value, the helpers return `None`. Numerical candidates remain available in the detailed result.

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

`argmin()` and `argmax()` return tuples of represented optimizer mappings. A certified but unattained infimum/supremum yields an empty tuple.

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
result = minimize_result(a * (x - 2) ** 2, variables=[x])
```

Here `x` is optimized and `a` remains a symbolic parameter. When `variables` is omitted, a symbolic region supplies its owned coordinates; without a region, all free symbols are treated as decision variables.

## Precision

`working_precision=N` is a package-level precision request and provenance contract. Exact routes remain exact. High-precision inexact input is not silently sent through float64-only LP/QP/SOS fast paths; the dispatcher chooses precision-preserving symbolic/numerical or verified-interval routes instead.

A rounded floating objective evaluation does not supply an exact global value. The pipeline retains it as candidate information unless an independent exact bound establishes that value. Constant floating literals can be represented exactly by their stored binary rational value. Use exact coefficients when an exact infimum or optimizer-locus formula is needed.

Inspect:

```python
result.working_precision
result.precision_provenance
```

for the requested precision, detected input precision, effective routing policy, and backend limitations.

## Lazy KKT construction

`PipelineResult.kkt_system` is optional. If an LP/QP, structural route, or exact backend solves the problem before KKT search, no KKT system is built. This avoids unnecessary symbolic differentiation and is expected behavior, not missing diagnostic information.

See {doc}`../guide/solver-selection`, {doc}`../getting-started/understanding-results`, and {doc}`../guide/certification`.

## Result ergonomics

A detailed `PipelineResult` provides boolean `infeasible` and `unbounded` properties, a stable `method` label, and `explain()` for a concise route and certification summary. Multiple represented isolated optimizers are preserved; free coordinates in a continuum are represented by omitted dictionary keys.

## Assumptions and backend options

Symbol sign assumptions are used only when `use_symbol_assumptions=True`. Backend-specific controls can be grouped under `options`, with sections `interval`, `sos`, `semialg`, and `kkt`. Flat keyword arguments take precedence when supplied explicitly, including explicit default values.
