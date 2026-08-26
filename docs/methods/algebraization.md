# Algebraization

Algebraization converts selected nonpolynomial expressions into polynomial
systems by introducing auxiliary variables and graph/domain constraints.

## Square roots

For

\[
y=\sqrt{x+1},
\]

the polynomial graph relation is

\[
y^2=x+1,
\]

together with the real branch condition \(y\ge0\).

## Inverses

For

\[
y=\frac1{p(x)},
\]

the lift includes

\[
yp(x)=1,
\]

which automatically excludes \(p(x)=0\).

## Absolute value and piecewise branches

`Abs` and `Piecewise` are handled by branch decomposition so that each branch
has explicit sign/condition constraints.

## RootOf

Fixed real `RootOf`/`CRootOf` branches are lifted with their defining polynomial
and an exact rational isolating interval, preserving branch identity without
floating-point root matching. Repeated occurrences reuse one lift. Exact
`AlgebraicNumber` inputs are preserved/algebraized without conversion through
binary floating point. Parameter-dependent root branches remain a harder case.

## Transcendentals

Some bounded transcendental problems are better left to verified interval
optimization rather than forced into polynomial algebraization.

## Inspection

```python
import sympy as sp
from symbopt import algebraize

x = sp.symbols("x", real=True)
problem = algebraize(sp.sqrt(x + 1), [sp.Ge(x, -1)])
print(problem.lifted_objective)
print(problem.all_constraints())
```

The `LiftedProblem` retains lift records so the transformation remains
inspectable.

## Further reading

See Cox--Little--O'Shea in {doc}`../reference/references` for the mathematical background.

### Piecewise result aggregation

Branch results are combined using rigorous branch lower bounds and feasible
candidate upper bounds. An unattained branch infimum is not discarded merely
because another branch has an attained candidate. Exact global certification is
claimed only when every relevant branch infimum is itself proved.
