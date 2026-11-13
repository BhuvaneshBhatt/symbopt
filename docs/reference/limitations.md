# Limitations

The package is explicit about unsupported or difficult cases.

## Polynomial and rational scope

Exact univariate critical/boundary enumeration supports fixed real algebraic coefficients when root isolation and comparisons complete. General rational graph projection depends on the optional semialg backend and its budget. Automatic sublevel boxes use a sufficient leading-form dominance test; refusal does not prove noncoercivity. Complete analysis of all escaping curves, arbitrary asymptotic critical values and general parameter regions remains outside these routes. See {doc}`../methods/polynomial-rational`.

## Transcendental image and optimizer coverage

funcprops can represent exact implicit critical values and complete some images that symbopt does not yet turn into an exact optimizer. Its automatic `range` and explicit `analyze_range` interfaces have different dispatch coverage. Use the [funcprops range tutorial](https://github.com/BhuvaneshBhatt/funcprops/blob/main/docs/ranges.md) for the supported entry point. Source holes, one-sided endpoint ownership and unresolved value equalities remain premises of any optimization conclusion.

For incommensurate periodic components, a dense closure can establish an infimum or supremum while attainment remains unknown. A closed extremal enclosure is not automatically an exact image. General analytic parameter collisions, escaping critical points and arbitrary multivariate transcendental ranges remain outside a complete solver. See {doc}`../guide/ranges-and-extrema`.

## Symbolic complexity

Exact stationary/KKT solving and Gröbner elimination can become extremely expensive as degree, variable count, or algebraic complexity increases. Numerical fallback may find candidates without proving completeness.

## KKT conditions

KKT conditions are necessary only under suitable regularity assumptions. Karush–John handling reduces reliance on constraint qualifications, but solving the resulting symbolic system can still be difficult.

Second-order inequality classification builds the normalized critical cone for degenerate active inequalities and tests the Lagrangian quadratic form with exact semialgebraic reasoning when `semialg` is available. Pure tangent-subspace cases are handled by exact reduced-Hessian linear algebra. Large or unsupported critical cones can still remain inconclusive.

## LP/simplex

The public general LP driver handles equality constraints, free/bounded variables, and feasibility initialization through HiGHS. The package's low-level tableau/revised-simplex functions are standard-form routines and do not implement a separate artificial-variable feasibility algorithm.

## Parameterized quadratics

Real symbolic coefficients are supported for unconstrained quadratics with proved curvature and for one-variable problems with affine bounds. Closed bounded intervals also support conditional curvature. These routes return a parameter-dependent value, and return an optimizer only when it is valid throughout the stated assumptions. The ordinary dispatch portfolio does not automatically produce closed-form answers across arbitrary parameter families. The explicit `optimization_family` interface provides finite-value, feasibility, attainment, unboundedness and optimizer relations through exact real QE, together with supported geometric fiber analysis; see {doc}`../guide/regions`. General higher-dimensional topological trivialization remains unsupported. See {doc}`../methods/parameterized-quadratics` for examples and precise conditions.

## QP

`solve_convex_qp()` is only for convex quadratic objectives. A non-PSD quadratic form is rejected.

Global quadratic methods are available through `minimize_quadratic()` and `solve_global_qp()`. Native methods require rational continuous coefficients and closed constraints; spatial search needs a proved finite domain reduction. Finite budgets may leave a positive gap or no feasible incumbent. Optional SCIP supports fixed real continuous and integer quadratic models with numerical bounds only. See {doc}`../methods/global-quadratics`.

## SDP numerics

The internal SDP solver is not intended to compete with mature industrial SDP solvers on large dense problems. An optional CVXPY adapter can route numerical proposal solves to installed conic solvers such as Clarabel, SCS, CVXOPT, or MOSEK-compatible backends. A numerical SDP result from any backend is still not an exact certificate.

Sparse SDP/SOS routing supports overlapping chordal-clique decomposition, but the internal numerical SDP backend remains intended for small research problems and chordal conversion cannot remove worst-case fill-in or dense cliques.

## SOS scaling

SOS Gram bases grow rapidly with degree and variable count. Exact Gram recovery may fail even when the numerical relaxation appears feasible; in that case the result is not promoted to an exact SOS certificate.

## Moments

Flatness and atom extraction are sensitive to rank decisions in numerical moment data. Exact verification is preferred when the extracted measure can be reconstructed symbolically.

## Analytic structural routes

The dedicated linear-fractional route requires an affine numerator and denominator and affine equality/inequality constraints. It first uses a fixed-sign Charnes–Cooper reduction when possible; otherwise it splits the feasible closure into positive- and negative-denominator branches, solves each transformed LP, and checks attainment against the original strict constraints. Singular branches can therefore certify unattained finite limits or unbounded values. Strict-affine contradictions are proved by a shared positive-slack LP before this route.

Exact periodic-trigonometric routing handles one-variable rational expressions built from `sin`, `cos`, `tan`, `cot`, `sec`, and `csc` with modest rationally related frequencies. The tangent-half-angle reduction uses exact rational or algebraic coefficient fields, isolates transformed denominator roots, and certifies one-sided pole behavior. Constant phases are supported when their sine and cosine values are exact algebraic numbers. Very high harmonics, nonalgebraic phase data, and frequency relations that cannot be certified remain outside this route. A separate mixed-periodic route handles continuous components with exactly known periods when inverse-period Q-linear independence is certified and the variable ranges over an unbounded interval. More general noncommensurate systems, discontinuous periodic components, and unresolved attainment remain unresolved.

Unbounded-domain transcendental analysis explicitly intersects the requested region with the real continuous domain, records singularities, checks one-sided limits, and uses exact convexity plus exhaustive stationary points on supported univariate rational/logarithmic/exponential components. These are sufficient routes with bounded function and domain support.

Additively separable problems are detected from the decision-variable interaction graph and independently certified components are recombined. Coupled objective terms or constraints correctly suppress this decomposition; parameter-dependent logical coupling is left to the ordinary pipeline.

## Precision

`working_precision` is a package-level precision request and provenance contract, not a promise that every backend is arbitrary precision. Exact symbolic routes remain exact, numerical KKT work uses the requested precision, and verified interval precision is raised accordingly. SciPy/NumPy LP, QP, SDP, and dense linear-algebra internals remain float64. For high-precision inexact inputs, `symbopt` bypasses the float64 LP/QP/SOS fast paths and records that routing decision in `PipelineResult.precision_provenance`; full arbitrary-precision replacements for those numerical backends are still not implemented.

## Interval optimization

Verified interval methods require finite bounds. Natural interval/ball evaluation can overestimate ranges due to dependency. Monotonicity, Taylor, polynomial Bernstein bounds, and verified interval-Newton/Krawczyk contraction of equality systems help but do not eliminate exponential branch-and-bound growth in hard dimensions.

Only functions implemented by the verified evaluator are eligible for rigorous Arb certification.

## Algebraization

Not every analytic expression has a practical finite polynomial graph representation. Unsupported functions are rejected or left to other solver routes.

## Out of scope

The package does not currently provide:

- general discrete optimization outside the optional quadratic SCIP adapter,
- automatic closed-form solutions across arbitrary parameter families; explicit exact family relations remain subject to CAD/QE scope and cost,
- a built-in CAD / quantifier-elimination engine. Exact semialgebraic optimization and CAD/QE are delegated to the optional `semialg` backend.

For the positive capability summary, see {doc}`capability-matrix`. For proof semantics, see {doc}`../guide/certification`.

## semialg backend

Exact CAD/QE and semialgebraic optimization are delegated to the optional `semialg` dependency. The canonical policy for when the automatic dispatcher uses that backend is documented in {doc}`../guide/solver-selection`; this page only records the architectural limitation that symbopt does not implement a second CAD/QE engine.

## Decision domains

Continuous global and local optimization reject decision symbols whose assumptions imply integrality, including `integer=True`, `even=True` and `odd=True`. This check also applies to inferred variables and to unused variables explicitly listed by the caller. It runs even when `use_symbol_assumptions=False`; that option cannot turn an integer problem into a continuous relaxation. Integer parameters remain allowed when the continuous decision variables are explicit. Unsupported integer problems raise `NotImplementedError`.

The explicit quadratic SCIP backend accepts integer, even and odd decision symbols and retains parity constraints. It never falls through to a continuous solver. Its numerical status and bounds are not exact certificates.
