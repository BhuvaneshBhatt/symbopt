# Assumptions and domains

`symbopt` targets continuous real optimization.

## Real symbols

Use `real=True` when practical:

```python
x = sp.symbols("x", real=True)
```

This helps SymPy simplify signs and classify roots.

## Domain restrictions created by algebraization

A lift is not just an algebraic equality. For example,

\[
y=\sqrt{x+1}
\]

requires both

\[
y^2=x+1
\]

and the correct real branch/domain restrictions, including \(y\ge0\).

Similarly, inverses require a nonzero denominator.

## Verified transcendental domains

The interval evaluator checks domains and singularities. Examples include:

- division by an interval containing zero,
- logarithms crossing or lying below zero,
- tangent crossing a pole,
- inverse trigonometric functions outside their real domains.

A box whose entire domain is invalid is rejected. A box that straddles a
singularity is subdivided or reported as uncertain.
