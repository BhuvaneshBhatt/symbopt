# Degenerate stationary points

A zero Hessian eigenvalue makes the ordinary second-derivative test inconclusive. `symbopt` supplements it with directional analysis in the Hessian nullspace.

For a nullspace direction \(d\), the package examines a Taylor expansion of

\[
f(x^\star+td)-f(x^\star)
\]

and finds the first nonzero coefficient when possible.

This can distinguish some flat minima from degenerate saddle behavior.

## Example

For

\[
f(x)=x^4,
\]

the Hessian at \(x=0\) is zero, but the fourth-order term is positive. The degeneracy analysis can therefore identify the point as a flat minimum candidate rather than treating every zero-Hessian case identically.

Multivariate degeneracy can still remain inconclusive when independent nullspace directions do not capture all mixed higher-order behavior.

## Further reading

See Nocedal–Wright and Bertsekas in {doc}`../reference/references` for the mathematical background.
