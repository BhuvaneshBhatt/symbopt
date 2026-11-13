# Sum-of-squares optimization

A polynomial \(p(x)\) is SOS if it can be represented as

\[
p(x)=z(x)^TQz(x),\qquad Q\succeq0.
\]

`symbopt` compiles this representation directly into an SDP using exponent-indexed coefficient assembly.

## Exact base cases

Before compiling an SDP, `sos_decompose()` recognizes trivial exact cases. The zero polynomial returns an immediate exact zero Gram certificate, and exact polynomial squares (including scaled affine squares) can be certified directly. These paths avoid numerical SDP convergence for identities that are already obvious algebraically.

## Unconstrained lower bounds

To prove \(f(x)\ge\gamma\), the package searches for an SOS representation of

\[
f(x)-\gamma.
\]

## Constrained certificates

For inequalities \(h_j(x)\ge0\) and equalities \(g_i(x)=0\), a relaxation can take the form

\[
f-\gamma
=
\sigma_0
+\sum_j\sigma_jh_j
+\sum_i\tau_i g_i,
\]

where each \(\sigma_j\) is SOS.

## Early sparse/block propagation

The SOS compiler analyzes the correlative variable graph before constructing the SDP. Disconnected components become separate Gram matrices immediately. Connected sparse graphs are chordally extended; when maximal cliques overlap, each clique receives its own Gram basis and coefficient matching reconstructs the polynomial from the sum of all clique SOS blocks.

For example,

\[
x^2+y^2
\]

can be represented using separate \(\{x\}\) and \(\{y\}\) Gram blocks, whereas

\[
(x+y)^2
\]

contains a true \(x\)-\(y\) interaction and remains in one connected block.

For example, \((x+y)^2+(y+z)^2\) uses overlapping \(\{x,y\}\) and \(\{y,z\}\) Gram blocks. Exact recovery still verifies the complete polynomial identity and every clique Gram matrix, so sparsity does not weaken certificate semantics.

## Exact recovery

A numerical SDP can guide the search, but rigorous SOS certification requires an exact recovered certificate. Recovery is performed on the **coupled affine certificate system**, not by rationalizing each Gram entry independently.

The recovery path is

\[
\text{numerical SDP point}
\rightarrow
\text{exact affine certificate space}
\rightarrow
\text{high-precision projection}
\rightarrow
\text{rational reconstruction}
\rightarrow
\text{exact verification}.
\]

The affine system includes every Gram block and every equality-multiplier coefficient. Underdetermined parameters are fitted to the numerical SDP point at increasing precision, followed by increasingly permissive rational reconstruction. A recovered candidate is accepted only when:

1. the polynomial identity is exactly zero, and
2. every recovered Gram matrix is exactly positive semidefinite.

Higher numerical precision can make exact recovery more likely, but precision alone never turns a numerical relaxation into a proof.

## Example

```python
import sympy as sp
from symbopt import sos_decompose

x = sp.symbols("x", real=True)
result = sos_decompose((x**2 - 1) ** 2, [x])
print(result.success)
print(result.certificate)
```

See {doc}`../guide/certification` for proof semantics and {doc}`sparse-structure` for structural decomposition.

## Further reading

See Lasserre, Parrilo, Laurent, and Blekherman–Parrilo–Thomas in {doc}`../reference/references` for the mathematical background.

## Relaxation status

For constrained relaxations, `success` means that the numerical SDP relaxation produced a usable iterate. `certified` is stronger and requires exact identity recovery plus exact PSD verification. Approximate relaxations use `status="numerically_feasible"` and report only `lower_bound_numeric`; the symbolic `lower_bound` field is reserved for certified lower bounds.

## Independent replay

`SOSCertificate.verify()` rebuilds the polynomial identity from the bound, normalization scale, basis, every Gram block, nonnegative domain multiplier and equality multiplier. It checks exact symmetric PSD matrices and the stored residual summaries. `verified` invokes this replay; changing only a cached flag cannot establish a certificate. Optional `polynomial`, `lower_bound`, `inequalities` and `equalities` arguments bind replay to a caller's intended theorem. The proof may use a subset of the supplied domain premises.

Bisection bounds retain the exact binary rational used in recovery. Decimal rational reconstruction cannot raise a lower bound after its identity has been checked. Numeric SDP status and residual tolerances remain proposal evidence. See {doc}`../examples/applications` for a replay example.
