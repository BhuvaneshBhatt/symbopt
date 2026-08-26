# Critical points

For a smooth unconstrained objective \(f(x)\), stationary points satisfy

\[
\nabla f(x)=0.
\]

`symbopt.critical_points()` solves these equations symbolically when possible and
classifies isolated stationary points using the Hessian.

## Classification

At a stationary point \(x^\star\):

- positive-definite Hessian → local minimum,
- negative-definite Hessian → local maximum,
- mixed signs → saddle point,
- zero eigenvalues → degenerate/inconclusive unless higher-order directional
  analysis resolves the case.

The degeneracy helpers examine nullspace directions with directional Taylor
expansions.

## Example

```python
import sympy as sp
from symbopt import critical_points

x, y = sp.symbols("x y", real=True)
critical_points(x**2 - y**2, [x, y])
```

The origin is a saddle because the Hessian has one positive and one negative
eigenvalue.

## Limitations

Large nonlinear stationary systems can be difficult to solve exactly. Numerical
root finding can provide candidates, but local classification is not a global
certificate.

## Further reading

See Nocedal--Wright and Bertsekas in {doc}`../reference/references` for the mathematical background.

