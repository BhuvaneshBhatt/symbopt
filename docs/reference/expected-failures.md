# Expected failures and coverage targets

The executable reference corpus and focused unsupported-case tests track different mathematical scopes. An expected failure can mean an unsupported mathematical family, a finite work budget, a result that does not yet satisfy the reference assertions, or a reference oracle requiring review. It does not always mean that no backend can find a useful candidate.

## Supported specialized cases

Exact bounded rational nonconvex QPs have a complete stationary-face route. Rational QCQPs with a proved finite domain have native spatial bounds and certificate replay. Optional SCIP covers broader numerical global QP, QCQP and mixed-integer searches, including rational strict lattice rows and parity assumptions. Numerical success does not satisfy a reference that requires an exact symbolic certificate. See {doc}`../methods/global-quadratics`.

Additional exact structural rules cover centered quadratic ellipsoids, PSD quadratic corrections that vanish at a linear support point, bounded perturbations of a single affine tangent pole, and strict-affine infeasibility through a shared positive-slack LP. The ellipsoid support rule handles examples with exact coefficients such as `E` and `pi` before rational conversion.

Open-domain references distinguish finite limiting values from attained optimizers. Comparisons also account for equivalent exact values expressed with `Min`, `Max` or `Piecewise`.

## Remaining families

| Family | Example or issue | Next useful improvement | Effort |
|---|---|---|---|
| Parameter regions | Parameter-dependent feasibility, rank changes, algebraic root branches | Explicit parameter strata and branch conditions through semialg, with exact specialization tests | Substantial |
| Noncompact QCQPs | No proved finite global domain, escaping curved paths | Proved bounds from nonnegative combinations of quadratic constraints; more recession and asymptotic reductions | Moderate to substantial |
| Exact curved-boundary optima | A feasible rational proposal may miss an algebraic equality intersection | Algebraic recovery on active quadratic systems and stronger native relaxations | Moderate |
| General integer models | Default symbolic corpus rejects discrete decisions | Explicit numerical SCIP already covers quadratic models; broader polynomial/transcendental discrete models need separate formulations | Varies |
| Special functions | Gamma/zeta poles, LambertW domains, named functions without a SymPy implementation | Small proved domain/pole rules first; exact function-property support for larger families | Small for selected identities, substantial generally |
| Transcendental constraints | Nonalgebraic phase shifts, boxed mixed functions, several variables | Better verified function ranges and domain splitting, then stronger numerical proposal methods | Moderate to substantial |
| Work budgets | Isolation, CAD, KKT or symbolic comparison exceeds the reference budget | Reuse exact facts, improve dispatch and isolate costs with benchmarks | Case-dependent |
| Oracle and API contracts | A pole has no optimizer; a removed point cannot be an attained maximum | Replace optimizer checks by value/attainment checks; resolve formal root targets against defining equations | Small per case |

Non-running expected failures record an unsupported target; they do not measure executed coverage. Gamma/zeta poles and removable-singularity cases need executable value, function-domain and attainment checks. General parameter cases and high-dimensional elimination require broader algorithms.

Budget-dependent outcomes can vary with optional dependencies and machine load. Per-case validation reports retain the observed outcome, budget, descriptive case identity and reason for each case. The corpus tests remain executable, and supported cases retain mathematical value, feasibility and attainment assertions.
