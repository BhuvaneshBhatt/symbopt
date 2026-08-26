# Limitations

The package is intentionally explicit about unsupported or difficult cases.

## Symbolic complexity

Exact stationary/KKT solving and Gröbner elimination can become extremely
expensive as degree, variable count, or algebraic complexity increases.
Numerical fallback may find candidates without proving completeness.

## KKT conditions

KKT conditions are necessary only under suitable regularity assumptions.
Karush--John handling reduces reliance on constraint qualifications, but solving
the resulting symbolic system can still be difficult.

Second-order inequality classification uses tangent/reduced-Hessian information
for equality and strongly active constraints. General critical-cone analysis for
all degenerate active-inequality cases can remain inconclusive.

## LP/simplex

The public general LP driver handles equality constraints, free/bounded
variables, and feasibility initialization through HiGHS. The package's
low-level tableau/revised-simplex functions are standard-form routines and do
not implement a separate artificial-variable feasibility algorithm.

## QP

`solve_convex_qp()` is only for convex quadratic objectives. A non-PSD
quadratic form is rejected.

## SDP numerics

The internal SDP solver is not intended to compete with mature industrial SDP
solvers on large dense problems. A numerical SDP result is not an exact
certificate.

Sparse SDP/SOS routing now supports overlapping chordal-clique decomposition,
but the internal numerical SDP backend remains intended for small research
problems and chordal conversion cannot remove worst-case fill-in or dense cliques.

## SOS scaling

SOS Gram bases grow rapidly with degree and variable count. Exact Gram recovery
may fail even when the numerical relaxation appears feasible; in that case the
result is not promoted to an exact SOS certificate.

## Moments

Flatness and atom extraction are sensitive to rank decisions in numerical moment
data. Exact verification is preferred when the extracted measure can be
reconstructed symbolically.

## Analytic structural routes

The dedicated linear-fractional route requires an affine numerator and denominator
and affine equality/inequality constraints. It first uses a fixed-sign
Charnes--Cooper reduction when possible; otherwise it splits the feasible closure
into positive- and negative-denominator branches, solves each transformed LP, and
checks attainment against the original strict constraints. Singular branches can
therefore certify unattained finite limits or unbounded values. A remaining edge
case is dedicated status reporting for contradictory strict affine systems: these
are kept conservative rather than being promoted by this route alone.

Exact periodic-trigonometric routing now handles one-variable `sin`/`cos`
polynomials with modest rationally related frequencies by reducing them through
a tangent-half-angle substitution to an exact rational stationary polynomial.
High harmonics, arbitrary phase constants, tangent/secant poles, and general
noncommensurate frequencies remain outside this exact route.

Unbounded-domain transcendental analysis explicitly intersects the requested
region with the real continuous domain, records singularities, checks one-sided
limits, and uses exact convexity plus exhaustive stationary points on supported
univariate rational/logarithmic/exponential components. It remains intentionally
conservative and is not a general transcendental theorem prover.

Additively separable problems are detected from the decision-variable interaction
graph and independently certified components are recombined. Coupled objective
terms or constraints correctly suppress this decomposition; parameter-dependent
logical coupling is left to the ordinary pipeline.

## Precision

`working_precision` is a package-level precision request and provenance contract,
not a promise that every backend is arbitrary precision. Exact symbolic routes
remain exact, numerical KKT work uses the requested precision, and verified
interval precision is raised accordingly. SciPy/NumPy LP, QP, SDP, and dense linear-algebra internals remain float64.
For high-precision inexact inputs, `symbopt` now bypasses the float64 LP/QP/SOS
fast paths and records that routing decision in `PipelineResult.precision_provenance`;
full arbitrary-precision replacements for those numerical backends are still not
implemented.

## Interval optimization

Verified interval methods require finite bounds. Natural interval/ball
evaluation can overestimate ranges due to dependency. Monotonicity, Taylor,
polynomial Bernstein bounds, and verified interval-Newton/Krawczyk contraction
of equality systems help but do not eliminate exponential branch-and-bound
growth in hard dimensions.

Only functions implemented by the verified evaluator are eligible for rigorous
Arb certification.

## Algebraization

Not every analytic expression has a practical finite polynomial graph
representation. Unsupported functions are rejected or left to other solver
routes.

## Out of scope

The package does not currently provide:

- integer or mixed-integer optimization,
- general parametric solution regions,
- a built-in CAD / quantifier-elimination engine. Exact semialgebraic
  optimization and CAD/QE are delegated to the optional `semialg` backend.

For the positive capability summary, see {doc}`capability-matrix`. For proof semantics, see {doc}`../guide/certification`.



## semialg backend

Exact CAD/QE support is available through the optional `semialg` dependency. Backend capabilities are discovered at runtime rather than inferred from a fixed top-level module layout. In `semialg_certification="auto"` mode, routing is cost-aware: cheap structural/LP/QP routes run first, and small polynomial systems may use symbolic KKT/elimination before an expensive CAD range computation. Suitable semialgebraic problems can still route directly to semialg, including higher-dimensional optimization, positive-dimensional loci, and exact attainment/non-attainment when supported by the installed backend. Use `"candidate"` or `"complete"` to request those backend policies explicitly.


## Semialgebraic range and sign fallbacks

The `function_range` fallback is exact for expressions supported by semialg's
semialgebraic graph machinery, including polynomial/rational expressions and
selected constructs such as `Abs`, square roots, `Min`, `Max`, and simple
`Piecewise`. It is not a transcendental range engine. General constrained sign
proofs are formulated as exact implications when the dedicated global sign API
does not accept assumptions directly.
