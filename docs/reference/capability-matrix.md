# Capability matrix

This table summarizes the current public capabilities. "Rigorous global certificate" means that the package has a proof-producing path for at least a useful subset of that problem class; it does not mean that every instance will be certifiable.

| Problem/feature | Supported | Candidate type | Rigorous global certificate | Main route / restrictions | Executable example |
|---|---:|---|---:|---|---|
| Region inputs | Sets and formulas; semialg geometry optional | Same as the underlying solver | Same as the underlying solver | exact membership conversion; explicit or region-owned coordinate order; structural outer boxes and covering parameterizations | [Example](../guide/regions.md#sets-and-formulas) |
| Early structural rules | Recognized forms | exact witnesses, values or limits | Yes when the rule succeeds | bounded shape/domain checks; see {doc}`../methods/structural-rules` | [Example](../examples/capability-tour.md#structural-polynomial-rules) |
| Exact one-variable polynomial/rational | Yes over fixed real algebraic coefficients | algebraic critical/boundary enumeration | Yes when enumeration completes | exact root isolation, poles, exclusions and one-sided/infinite limits | [Example](../examples/capability-tour.md#univariate-polynomial-and-rational-functions) |
| Rational graph/image | Optional | exact range and verified witnesses | Yes where backend projection succeeds | semialg graph QE with a degree/dimension budget | [Example](../examples/capability-tour.md#optional-semialgebraic-projection) |
| Polynomial sublevel boxes | Structured cases | exact finite search box | Exterior bound only | positive even leading-form margins and a verified feasible upper bound | [Example](../examples/capability-tour.md#sublevel-search-boxes) |
| Unconstrained smooth polynomial | Yes | symbolic/algebraic/numerical | Sometimes | stationary equations; SOS/interval certification | [Example](../examples/capability-tour.md#structural-polynomial-rules) |
| Equality-constrained smooth | Yes | KKT/Lagrange | Sometimes | reduced-Hessian local classification; global proof method-dependent | [Example](../examples/capability-tour.md#smooth-constrained-minima) |
| Inequality-constrained smooth | Yes | KKT/Karush–John | Sometimes | \(h\ge0\) convention; budgeted active-set generation | [Example](../examples/capability-tour.md#smooth-constrained-minima) |
| Linear programming | Yes | numerical + exact postprocessing | Exact only when reconstruction and exact feasibility checks succeed | general LP driver; exact active KKT reconstruction when possible | [Example](../examples/capability-tour.md#linear-programs) |
| Convex quadratic programming | Yes | numerical + exact postprocessing | Exact only when exact reconstruction, feasibility, multiplier signs, and convexity checks succeed | PSD quadratic form + linear constraints; exact KKT reconstruction when possible | [Example](../examples/capability-tour.md#convex-and-parameterized-quadratics) |
| Parameterized quadratics | Restricted | exact symbolic values and points | Yes in supported cases | unconstrained PSD forms with a uniform real stationary point; scalar affine intervals with proved feasibility; closed bounded intervals allow conditional curvature | [Example](../examples/capability-tour.md#convex-and-parameterized-quadratics) |
| General nonconvex QP/QCQP | Yes within stated backend scope | exact incumbent and bounds, or numerical SCIP | Native bounds, exact optimum when bounds meet | face enumeration, spatial McCormick relaxations, optional SCIP; see {doc}`../methods/global-quadratics` | [Example](../examples/capability-tour.md#nonconvex-quadratic-bounds) |
| Polynomial SOS decomposition | Yes | Gram representation | Yes when exact recovery succeeds | coupled affine recovery across all Gram blocks; exact identity + exact PSD checks | [Example](../examples/capability-tour.md#sos-identities-and-constrained-proofs) |
| Constrained polynomial SOS | Yes | hierarchy lower bound | Yes when exact certificate recovered | numerical SDP feasibility alone is not rigorous | [Example](../examples/capability-tour.md#sos-identities-and-constrained-proofs) |
| Sparse polynomial SOS | Yes | structural + Gram blocks | Same as SOS route | disconnected components plus overlapping chordal clique Gram blocks; exact identity/PSD recovery unchanged | [Example](../examples/capability-tour.md#sparse-polynomial-structure) |
| Moment matrices/localizing matrices | Yes | symbolic | N/A by themselves | polynomial moment machinery | [Example](../examples/capability-tour.md#moment-and-localizing-matrices) |
| Flatness/atom extraction | Yes | algebraic/numerical atoms | Verification-dependent | independent monomial selection and joint separation; complete original moments and positive weights must verify exactly | [Example](../examples/capability-tour.md#moment-and-localizing-matrices) |
| Sparse polynomial structure | Yes | structural | Same as underlying method | support pruning, correlative components, chordal extension, overlapping clique propagation | [Example](../examples/capability-tour.md#sparse-polynomial-structure) |
| SDP dense | Yes | numerical | Not exact by itself | internal factorized numerical solver | [Example](../examples/capability-tour.md#numerical-dense-and-sparse-sdp) |
| SDP sparse | Yes for one-block chordal sparsity | numerical | Not exact by itself | chordal extension, coefficient allocation, overlap-consistency equalities, reconstruction | [Example](../examples/capability-tour.md#numerical-dense-and-sparse-sdp) |
| Square roots/rational powers | Yes | lifted polynomial | Method-dependent | graph/domain constraints | [Example](../examples/capability-tour.md#algebraic-functions-and-constants) |
| Inverses | Yes | lifted polynomial | Method-dependent | nonzero denominator enforced | [Example](../examples/capability-tour.md#algebraic-functions-and-constants) |
| `Abs` | Yes | branch/lift | Method-dependent | sign branches | [Example](../examples/capability-tour.md#absolute-values-and-piecewise-functions) |
| `Piecewise` | Yes | branch decomposition | Method-dependent | each branch solved separately | [Example](../examples/capability-tour.md#absolute-values-and-piecewise-functions) |
| fixed real `RootOf` / algebraic constants | Yes for isolatable real branches | lifted algebraic | Method-dependent | exact rational isolating intervals; branch identity preserved; parameter-dependent roots remain limited | [Example](../examples/capability-tour.md#algebraic-functions-and-constants) |
| bounded transcendental | Yes for supported functions | numerical + verified bounds | Rigorous bounds/epsilon-optimality; exact certification requires an exact proof of equality | finite box + supported Arb mapping | [Example](../examples/capability-tour.md#bounded-transcendental-enclosures) |
| strict inequalities | Yes | candidate/infimum | Closure-aware | attained minimum may not exist | [Example](../examples/capability-tour.md#open-boundaries-and-attainment) |
| Exact semialgebraic feasibility/implication | Yes, optional | Boolean/witness | Yes | `semialg` CAD/QE/RUR backend | [Example](../examples/capability-tour.md#optional-semialgebraic-projection) |
| Exact semialgebraic optimization | Yes, optional for backend-supported polynomial problems | exact value/points/attainment | Yes | `semialg`; higher-dimensional support and certification policy are discovered at runtime | [Example](../examples/capability-tour.md#optional-semialgebraic-projection) |
| Exact objective range | Yes, optional | exact semialgebraic image/endpoints | Yes | `semialg.function_range`; used selectively when range information is more useful than candidate search | [Example](../examples/capability-tour.md#optional-semialgebraic-projection) |
| Polynomial sign proof | Yes, optional | exact sign/counterexample | Yes | semialg sign prover globally; exact implication under constraints | [Example](../examples/capability-tour.md#optional-semialgebraic-projection) |
| integer optimization | Quadratic models with explicit optional SCIP | numerical incumbent and bounds | No exact proof | `backend="scip"`; other continuous routes reject integer decision symbols | [Example](../examples/capability-tour.md#integer-models-and-parameter-scope) |
| Parameter-dependent regions | Exact definitions and backend-supported QE | value, attainment and optimizer relations | Exact where elimination completes | CAD fiber partitions; 1D Hardt topology; higher-dimensional fiber cells and dimension | [Example](../guide/regions.md#parameter-families) |
| CAD / quantifier elimination | Yes, optional | exact Boolean/formula results | Yes | provided through `semialg`; used selectively by `symbopt` | [Example](../examples/capability-tour.md#optional-semialgebraic-projection) |

See also {doc}`limitations`, {doc}`numerical-tolerances`, {doc}`../guide/solver-selection`, and {doc}`../methods/structural-routes`.

## Structural exact routes

| Input structure | Current status | Executable example |
|---|---|---|
| Affine linear-fractional program, including strict constraints and sign-changing denominators | Charnes–Cooper LP route with denominator-sign branch splitting | [Example](../examples/capability-tour.md#linear-programs) |
| `exp(g(x))`, `log(g(x))`, supported positive powers | monotone reduction when the real branch is globally monotone | [Example](../examples/capability-tour.md#monotone-and-periodic-reductions) |
| Same-frequency univariate sine/cosine combination | exact periodic range route | [Example](../examples/capability-tour.md#monotone-and-periodic-reductions) |
| Modest rationally related univariate `sin`/`cos`/`tan`/`cot`/`sec`/`csc` expressions with algebraic phase data | exact tangent-half-angle rational route with denominator-root and one-sided pole analysis | [Example](../examples/capability-tour.md#monotone-and-periodic-reductions) |
| Common univariate rational/log/exp/radical expression on an unbounded interval | exact domain/range asymptotic route | [Example](../examples/capability-tour.md#monotone-and-periodic-reductions) |
| Shared univariate function-property analysis | cached funcprops-first domain, period, singularity/discontinuity, monotonic partition, endpoint, stationary-point, extrema, and range facts with SymPy fallback | [Example](../examples/capability-tour.md#analysis-and-scope-boundaries) |
| Continuous univariate components with provably Q-independent inverse periods on an unbounded interval | certified mixed-periodic product-range reduction | [Example](../examples/capability-tour.md#monotone-and-periodic-reductions) |
| General noncommensurate periodic systems without a certified decomposition, or very high-frequency commensurate trig | not claimed | [Example](../examples/capability-tour.md#analysis-and-scope-boundaries) |
| General multivariate transcendental asymptotics | not claimed | [Example](../examples/capability-tour.md#analysis-and-scope-boundaries) |

| Feature | Supported scope | Executable example |
|---|---|---|
| Exact optimizer loci | proved feasible level sets, or universal comparison with optional QE | [Example](../guide/regions.md#exact-optimizer-loci) |
| Covering parameterizations | complete structural polynomial/rational charts and lifted witnesses | [Example](../guide/regions.md#covering-parameterizations) |
