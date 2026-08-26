# Capability matrix

This table summarizes the current public capabilities. "Rigorous global
certificate" means that the package has a proof-producing path for at least a
useful subset of that problem class; it does not mean that every instance will
be certifiable.

| Problem/feature | Supported | Candidate type | Rigorous global certificate | Main route / restrictions |
|---|---:|---|---:|---|
| Unconstrained smooth polynomial | Yes | symbolic/algebraic/numerical | Sometimes | stationary equations; SOS/interval certification |
| Equality-constrained smooth | Yes | KKT/Lagrange | Sometimes | reduced-Hessian local classification; global proof method-dependent |
| Inequality-constrained smooth | Yes | KKT/Karush--John | Sometimes | \(h\ge0\) convention; budgeted active-set generation |
| Linear programming | Yes | numerical + exact postprocessing | Exact only when reconstruction and exact feasibility checks succeed | general LP driver; exact active KKT reconstruction when possible |
| Convex quadratic programming | Yes | numerical + exact postprocessing | Exact only when exact reconstruction, feasibility, multiplier signs, and convexity checks succeed | PSD quadratic form + linear constraints; exact KKT reconstruction when possible |
| General nonconvex QP | Partial | symbolic/polynomial | Sometimes | not sent to convex-QP solver |
| Polynomial SOS decomposition | Yes | Gram representation | Yes when exact recovery succeeds | coupled affine recovery across all Gram blocks; exact identity + exact PSD checks |
| Constrained polynomial SOS | Yes | hierarchy lower bound | Yes when exact certificate recovered | numerical SDP feasibility alone is not rigorous |
| Sparse polynomial SOS | Yes | structural + Gram blocks | Same as SOS route | disconnected components plus overlapping chordal clique Gram blocks; exact identity/PSD recovery unchanged |
| Moment matrices/localizing matrices | Yes | symbolic | N/A by themselves | polynomial moment machinery |
| Flatness/atom extraction | Yes | algebraic/numerical atoms | Verification-dependent | requires suitable moment data/rank structure |
| Sparse polynomial structure | Yes | structural | Same as underlying method | support pruning, correlative components, chordal extension, overlapping clique propagation |
| SDP dense | Yes | numerical | Not exact by itself | internal factorized numerical solver |
| SDP sparse | Yes for one-block chordal sparsity | numerical | Not exact by itself | chordal extension, coefficient allocation, overlap-consistency equalities, reconstruction |
| Square roots/rational powers | Yes | lifted polynomial | Method-dependent | graph/domain constraints |
| Inverses | Yes | lifted polynomial | Method-dependent | nonzero denominator enforced |
| `Abs` | Yes | branch/lift | Method-dependent | sign branches |
| `Piecewise` | Yes | branch decomposition | Method-dependent | each branch solved separately |
| fixed real `RootOf` / algebraic constants | Yes for isolatable real branches | lifted algebraic | Method-dependent | exact rational isolating intervals; branch identity preserved; parameter-dependent roots remain limited |
| bounded transcendental | Yes for supported functions | numerical + verified bounds | Rigorous bounds/epsilon-optimality; exact certification requires an exact proof of equality | finite box + supported Arb mapping |
| strict inequalities | Yes | candidate/infimum | Closure-aware | attained minimum may not exist |
| Exact semialgebraic feasibility/implication | Yes, optional | Boolean/witness | Yes | `semialg` CAD/QE/RUR backend |
| Exact semialgebraic optimization | Yes, optional for backend-supported polynomial problems | exact value/points/attainment | Yes | `semialg`; higher-dimensional support and certification policy are discovered at runtime |
| Exact objective range | Yes, optional | exact semialgebraic image/endpoints | Yes | `semialg.function_range`; used selectively when range information is more useful than candidate search |
| Polynomial sign proof | Yes, optional | exact sign/counterexample | Yes | semialg sign prover globally; exact implication under constraints |
| integer optimization | No | — | — | outside current package scope |
| parametric optimization regions | No general support | — | — | outside current package scope |
| CAD / quantifier elimination | Yes, optional | exact Boolean/formula results | Yes | provided through `semialg`; used selectively by `symbopt` |

See also {doc}`limitations`, {doc}`numerical-tolerances`, {doc}`../guide/solver-selection`, and {doc}`../methods/structural-routes`.


## Structural exact routes

| Input structure | Current status |
|---|---|
| Affine linear-fractional program, including strict constraints and sign-changing denominators | Charnes--Cooper LP route with denominator-sign branch splitting |
| `exp(g(x))`, `log(g(x))`, supported positive powers | monotone reduction when the real branch is globally monotone |
| Same-frequency univariate sine/cosine combination | exact periodic range route |
| Modest rationally related univariate sine/cosine frequencies | exact tangent-half-angle stationary-polynomial route |
| Common univariate rational/log/exp/radical expression on an unbounded interval | exact domain/range asymptotic route |
| Arbitrary/noncommensurate or very high-frequency trigonometric global optimization | not claimed |
| General multivariate transcendental asymptotics | not claimed |
