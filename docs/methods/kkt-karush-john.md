# KKT and Karush--John conditions

For equalities \(g_i(x)=0\) and inequalities normalized as

\[
h_j(x)\ge0,
\]

`symbopt` uses the minimization Lagrangian

\[
L(x,\lambda,\mu)
=
f(x)
+\sum_i\lambda_i g_i(x)
-\sum_j\mu_j h_j(x),
\qquad \mu_j\ge0.
\]

The KKT conditions are:

\[
\nabla_x L=0,
\]

\[
g_i(x)=0,\qquad h_j(x)\ge0,
\]

\[
\mu_j\ge0,
\]

\[
\mu_jh_j(x)=0.
\]

## Karush--John

When a constraint qualification cannot be assumed, the objective multiplier
\(\rho\ge0\) is introduced. Multiplier nontriviality is enforced by a
normalization of the Karush--John multiplier vector.

## Second-order classification

For constrained points, the full Hessian is generally not the correct test.
`symbopt` forms a tangent-space basis for equality and strongly active
constraints and examines the reduced Hessian

\[
Z^T\nabla^2_{xx}LZ.
\]

Active inequalities with unresolved multiplier status can make the
classification inconclusive.

## Example

```python
x = sp.symbols("x", real=True)
critical_points((x, [sp.Ge(x, 1)]), [x])
```

The boundary point \(x=1\) satisfies KKT with a nonnegative multiplier under the
package's \(h\ge0\) convention.

## Further reading

See Nocedal--Wright, Bertsekas, and Boyd--Vandenberghe in {doc}`../reference/references` for the mathematical background.

