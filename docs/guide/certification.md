# Certification semantics

The package uses the word **certified** narrowly. A solver returning a candidate
is not sufficient; the result must be backed by a proof-producing route.

## Bound-based view

For minimization, let \(f^\star\) denote the global infimum. A certified lower
bound \(L\) and feasible candidate value \(U\) satisfy

\[
L\le f^\star\le U.
\]

If the bounds coincide exactly, or their difference is enclosed rigorously at
zero under the relevant proof semantics, the global optimum/infimum is proved.

## Exact certificates

Examples include an SOS identity

\[
f(x)-\gamma
=
\sigma_0(x)
+\sum_j\sigma_j(x)h_j(x)
+\sum_i\tau_i(x)g_i(x)
\]

where the polynomial identity is exact and the Gram matrices for each
\(\sigma_j\) are exactly positive semidefinite.

A small floating-point SDP residual is not an exact certificate. Numerical
SDP output may be projected onto the exact affine certificate space at higher
precision and rationally reconstructed, but the result becomes rigorous only
after the polynomial identity and all PSD conditions verify exactly.

## Verified interval certificates

A verified interval result carries explicit provenance indicating whether the
lower and upper bounds are rigorous. Rigorous bounds can come from:

- exact Bernstein bounds for polynomials,
- Arb natural evaluation,
- rigorous derivative monotonicity,
- centered Taylor enclosures with rigorous derivative bounds.

Heuristic corner/midpoint sampling is not allowed to contribute a rigorous
lower bound.

## Direct LP/QP results

For a structurally recognized LP or convex QP, the candidate is checked against
the original constraints. The QP route additionally requires a
positive-semidefinite quadratic form. These checks establish exact global correctness only when exact active-set
reconstruction succeeds, original-constraint feasibility is verified exactly,
all inequality multipliers are proved nonnegative, and the recognized convex
class conditions are established. Numerical feasibility fallback may validate
a candidate but does not create a rigorous LP/QP certificate.

## Strict inequalities

For a feasible set such as \(x>0\), the infimum can lie at \(x=0\) without being
attained. Certification on the closure is therefore distinguished in result
status from an attained minimum.

## Proof artifacts

`OptimalityCertificate.proof_objects` records the evidence used by the pipeline:
candidate origin, interval provenance, SOS exactness information, and a solver
decision trace.


## CAD/QE certificates

For an exact feasible candidate \(x^\star\) with value \(v\), the optional
`semialg` backend can prove global minimality by deciding the exact
semialgebraic query

\[
\exists x:\; C(x) \land f(x)<v.
\]

If that formula is unsatisfiable, no feasible point has a smaller objective, so
\(v\) is an exact global minimum. The resulting certificate records the backend
method and the decision query. When the semialg backend provides direct optimization,
`semialgebraic_minimize` is attempted earlier for exact polynomial/semialgebraic
problems and can supply certified higher-dimensional minima or infima together
with exact attainment metadata.


### Semialgebraic sign proofs

When exact SOS recovery does not close a polynomial lower bound, the optional
`semialg` backend can prove nonnegativity directly. For an unconstrained
polynomial this uses `prove_nonnegative`; under semialgebraic constraints the
claim is converted to an exact implication. A successful proof is recorded as a
`semialg_sign` artifact and can establish the candidate value as an exact global
lower bound.

### Exact objective ranges

`function_range` can eliminate the decision variables and describe the exact
semialgebraic image of the objective. `symbopt` then computes the exact lower
and upper endpoints of that image in the one-dimensional value variable. The
range formula itself is preserved, so disconnected ranges do not need to be
collapsed into a single interval. An exact lower endpoint is a rigorous lower
bound; when it equals a feasible candidate value, global optimality follows.

## Exact optimality versus epsilon-optimality

A rigorous lower bound that is merely numerically close to a feasible candidate
does not by itself prove exact equality. If

\[
0 \le U-L \le \varepsilon,
\]

`symbopt` reports the rigorous gap but does not set exact global certification
solely because the gap is small. Exact certification requires an exact proof
route such as symbolic equality of an exact SOS bound, semialgebraic proof, or
another proof that establishes the optimum exactly.

Verified interval endpoints are stored without replacing their outward-rounded
binary value by a nearby simplified rational, so bound direction is preserved.
