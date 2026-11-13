# Certification semantics

The package uses the word **certified** narrowly. A solver returning a candidate is not sufficient; the result must be backed by a proof-producing route.

## Bound-based view

For minimization, let \(f^\star\) denote the global infimum. A certified lower bound \(L\) and feasible candidate value \(U\) satisfy

\[
L\le f^\star\le U.
\]

If the bounds coincide exactly, or their difference is enclosed rigorously at zero under the relevant proof semantics, the global optimum/infimum is proved.

## Exact certificates

Examples include an SOS identity

\[
f(x)-\gamma
=
\sigma_0(x)
+\sum_j\sigma_j(x)h_j(x)
+\sum_i\tau_i(x)g_i(x)
\]

where the polynomial identity is exact and the Gram matrices for each \(\sigma_j\) are exactly positive semidefinite.

A small floating-point SDP residual is not an exact certificate. Numerical SDP output may be projected onto the exact affine certificate space at higher precision and rationally reconstructed, but the result becomes rigorous only after the polynomial identity and all PSD conditions verify exactly.

## Verified interval certificates

A verified interval result carries explicit provenance indicating whether the lower and upper bounds are rigorous. Rigorous bounds can come from:

- exact Bernstein bounds for polynomials,
- Arb natural evaluation,
- rigorous derivative monotonicity,
- centered Taylor enclosures with rigorous derivative bounds.

Heuristic corner/midpoint sampling is not allowed to contribute a rigorous lower bound.

## Direct LP/QP results

For a structurally recognized LP or convex QP, the candidate is checked against the original constraints. The QP route additionally requires a positive-semidefinite quadratic form. These checks establish exact global correctness only when exact active-set reconstruction succeeds, original-constraint feasibility is verified exactly, all inequality multipliers are proved nonnegative, and the recognized convex class conditions are established. Numerical feasibility fallback may validate a candidate but does not create a rigorous LP/QP certificate.

## Strict inequalities

For a feasible set such as \(x>0\), the infimum can lie at \(x=0\) without being attained. Certification on the closure is therefore distinguished in result status from an attained minimum.

## Proof artifacts

`OptimalityCertificate.proof_objects` records the evidence used by the pipeline: candidate origin, interval provenance, SOS exactness information, and a solver decision trace.

## CAD/QE certificates

For an exact feasible candidate \(x^\star\) with value \(v\), the optional `semialg` backend can prove global minimality by deciding the exact semialgebraic query

\[
\exists x:\; C(x) \land f(x)<v.
\]

If that formula is unsatisfiable, no feasible point has a smaller objective, so \(v\) is an exact global minimum. The resulting certificate records the backend method and the decision query. When the semialg backend provides direct optimization, `semialgebraic_minimize` is attempted earlier for exact polynomial/semialgebraic problems and can supply certified higher-dimensional minima or infima together with exact attainment metadata.

### Semialgebraic sign proofs

When exact SOS recovery does not close a polynomial lower bound, the optional `semialg` backend can prove nonnegativity directly. For an unconstrained polynomial this uses `prove_nonnegative`; under semialgebraic constraints the claim is converted to an exact implication. A successful proof is recorded as a `semialg_sign` artifact and can establish the candidate value as an exact global lower bound.

### Exact objective ranges

`function_range` can eliminate the decision variables and describe the exact semialgebraic image of the objective. `symbopt` then computes the exact lower and upper endpoints of that image in the one-dimensional value variable. The range formula itself is preserved, so disconnected ranges do not need to be collapsed into a single interval. An exact lower endpoint is a rigorous lower bound; when it equals a feasible candidate value, global optimality follows.

## Exact optimality versus epsilon-optimality

A rigorous lower bound that is merely numerically close to a feasible candidate does not by itself prove exact equality. If

\[
0 \le U-L \le \varepsilon,
\]

`symbopt` reports the rigorous gap but does not set exact global certification solely because the gap is small. Exact certification requires an exact proof route such as symbolic equality of an exact SOS bound, semialgebraic proof, or another proof that establishes the optimum exactly.

Verified interval endpoints are stored without replacing their outward-rounded binary value by a nearby simplified rational, so bound direction is preserved.

## Linear-fractional recession bounds

A numerical LP status is insufficient to certify an unbounded exact problem: a solver can discard a small constraint coefficient. The linear-fractional route checks numerical unboundedness with the exact rational LP before accepting it. A bounded margin LP checks that the transformed strict domain, including positive Charnes–Cooper scale, is nonempty. A finite answer from the exact LP replaces a conflicting numerical status.

A failed denominator-sign branch is distinct from a proved empty branch. Both possible branches must be resolved before a finite global infimum is certified. A verified branch with infimum minus infinity already establishes the global infimum. Algebraic-coefficient cases outside the exact rational LP fallback can remain unresolved.

## Exact source feasibility

Numerical candidate screening can tolerate small constraint residuals and can reconstruct rational coordinates from approximate proposals. Before such a proposal establishes attainment or a rigorous primal upper bound, symbopt rechecks the original source constraints, defined real value and objective equality. A valid SOS lower bound that matches an infeasible proposal does not prove optimality. Branch assembly uses verified branch evidence for bounds and attainment, and source-valid candidates for optimizer mappings.

A user-supplied `box_bounds` is search guidance. Its local interval lower bound contributes to a global certificate only when constraints or an improving-sublevel proof establish coverage. Use `bounds` or explicit relations to restrict the mathematical feasible domain.

## Infeasibility source premises

`verify_infeasibility_certificate` replays the nested SOS identity for minus one and checks that each preordering generator is a square-free product of the stored original inequalities. It rejects arbitrary negative Gram multipliers even when their fabricated identity is exact. Optional `equalities` and `inequalities` arguments bind replay to the caller's polynomial premises; these must match the stored tuples. It also checks the stored basis and ideal-membership summaries. See {doc}`../examples/capability-tour` for a proof and a changed-premise rejection.

Interval results and combined optimality summaries are evidence containers. They do not expose independent replay from arbitrary user-edited numerical flags. Use the native quadratic, SOS, infeasibility or univariate verifier for proof families that carry replayable witnesses.

## Floating-point witnesses and box ownership

A rigorous interval upper bound needs two checks: the proposed point satisfies the constraints inside the original requested box, and the upper endpoint encloses the objective at that exact binary point. Constraint feasibility alone does not certify a rounded objective value. Rational point values are rounded upward exactly; other supported values use a verified point enclosure. Equality corrections can propose points elsewhere in the search domain, but cannot establish a witness outside its bounds.

Contractor division rounds direct endpoint quotients outward. This avoids accumulating an inward reciprocal error before multiplication. Splits use finite midpoints and skip fixed or adjacent floating-point coordinates. A box that cannot be split further retains its lower-bound evidence. Exhausting floating-point resolution does not establish that its upper and lower bounds coincide. Exact singleton bounds may contain no representable floating-point witness; the low-level interval search can retain a bound without finding a point.
