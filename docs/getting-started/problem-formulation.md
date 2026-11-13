# Problem formulation

`symbopt` accepts SymPy expressions and SymPy relational constraints.

## Variables and assumptions

Use ordinary SymPy symbols, preferably with real assumptions when the problem is real-valued:

```python
import sympy as sp
from symbopt import minimize

x, y = sp.symbols("x y", real=True)
```

The package is currently designed for continuous real optimization.

## Objective

The objective is a SymPy expression:

```python
objective = (x - 1) ** 2 + (y + 2) ** 2
```

Use `minimize()` for minimization and `maximize()` for maximization. The latter uses an internal sign normalization but returns maximization semantics in `PipelineResult`.

## Constraints

Use SymPy relationals:

```python
constraints = [
    sp.Eq(x + y, 1),
    sp.Ge(x, 0),
    sp.Le(y, 3),
]
```

Internally, inequalities are normalized to a residual \(h(x)\ge 0\). For constrained minimization this convention implies the Lagrangian sign

\[
L=f+\lambda^Tg-\mu^Th,\qquad \mu\ge 0.
\]

Strict inequalities are preserved. Because a strict feasible set may not contain its infimum, a result may describe an infimum on the closure rather than an attained minimum.

## Geometric regions

Pass `region=` to optimize over a SymPy set, Boolean formula or optional `semialg` geometry. The adapter intersects membership with the other constraints and preserves open boundaries, holes and lower-dimensional components. Sets and coordinate-free geometry require explicit variables; symbolic regions supply their owned coordinate order. See {doc}`../guide/regions` for checked examples, certificate replay and supported conversion cases.

## Explicit boxes

The interval optimizer requires finite bounds for variables it must branch on. `minimize()` can infer two-sided scalar bounds from simple relational constraints. You can also pass a box explicitly:

```python
result = minimize(
    sp.sin(x),
    box_bounds={x: (0.0, 6.3)},
)
```

Explicit box bounds are numerical search bounds; exact relational constraints still define feasibility.

## Algebraic and piecewise expressions

The "algebraization" layer can lift selected non-polynomial expressions into auxilliary variables plus polynomial graph/domain constraints. Supported constructs include rational powers, square roots, inverses, `Abs`, `Piecewise`, and selected real `RootOf` cases. Unsupported functions are rejected.

## Transcendental expressions

Bounded supported transcendental expressions can be handled by verified interval reasoning. The Arb evaluator covers common real elementary/transcendental functions supported by the package. Domain singularities and poles cause boxes to be subdivided or rejected rather than treated as finite enclosures.

## Explicit decision variables and parameters

By default `symbopt` infers decision variables from all free symbols. Use an explicit list whenever an expression also contains parameters:

```python
import sympy as sp
from symbopt import minimize

x = sp.symbols("x", real=True)
a = sp.symbols("a", positive=True)
result = minimize(a * x**2, variables=[x])
```

Here `x` is a decision variable and `a` is retained as a parameter. The `LiftedProblem` records these separately as `original_variables` and `parameter_symbols`. General parameter-stratified optimisation is still delegated to specialised backend capabilities; explicit variables prevent accidental optimisation over parameters even when no parametric closed form is available.
