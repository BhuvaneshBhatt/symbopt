# Symbolic elimination

Polynomial candidate generation attempts symbolic elimination before falling
back to numerical `nsolve`.

The main tools include:

- direct symbolic equation solving,
- polynomial system solving,
- Gröbner-basis elimination,
- recovery of eliminated multipliers/coordinates.

Elimination is particularly useful when KKT systems contain multipliers that can
be removed to leave a smaller polynomial system in primal variables.

## Example

```python
from symbopt import elimination_polynomials
```

The lower-level elimination functions are intended for advanced use; most users
should call `minimize()` and inspect the candidate source/proof trace.

## Limitations

Gröbner bases can grow exponentially in difficult systems. Numerical fallback
is therefore a candidate-generation mechanism, not a guarantee that every
stationary solution has been found.

## Further reading

See Cox--Little--O'Shea in {doc}`../reference/references` for the mathematical background.

