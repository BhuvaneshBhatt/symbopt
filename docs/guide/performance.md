# Performance guidance

Performance depends strongly on problem structure. The most effective user-level choices are usually structural rather than numerical micro-tuning:

- provide finite, reasonably tight boxes for verified interval problems,
- preserve polynomial structure where possible,
- start SOS hierarchy search at low order,
- formulate separable/sparse polynomial problems transparently,
- avoid unnecessary symbolic expansion before calling `minimize()`.

For implementation details, scaling behavior, caches, SOS basis growth, interval reuse, and SDP workspace reuse, see {doc}`../reference/performance`.

For the role of solver tolerances and which quantities can affect rigorous certification, see {doc}`../reference/numerical-tolerances`.

Affine coefficient extraction reads the polynomial's coordinate coefficients and constant directly, sharing the polynomial cache with LP/QP preparation. Monomial enumeration filters a small Cartesian product for tiny requests and generates fixed-degree compositions when that product would be much larger than the output. Both paths preserve the same total-degree and lexicographic ordering. Within exact refinement, affine coefficient matrices and projection operators are prepared lazily and retained for repeated proposals. Every proposed solution still undergoes exact substitution checks.

## FLINT and structural preprocessing

`python-flint` is a declared dependency. `verified_intervals.py` calls its Arb real-ball arithmetic directly for natural interval evaluation, derivative signs, Taylor bounds, point enclosures and residual checks. These calls provide rigorous enclosures when the supported expression/domain conditions hold.

Exact symbolic work goes through SymPy. In the validated environment, SymPy selects FLINT ground integers and the `DUP_Flint` representation for eligible univariate polynomials over the integers and rationals. Rational domain matrices can use a FLINT-backed dense representation, but this does not imply that every symbolic matrix operation uses a native FLINT matrix kernel. Algebraic and symbolic coefficient domains have different backend paths. General LP/QP/SDP numerical routines still use SciPy/NumPy or their configured numerical solver.

The structural preprocessor avoids unnecessary expansion and general solving when a small exact proof is available. A proof obtained before planning leaves `solver_plan` absent. The structural recognizers can be disabled independently for comparison or backend-specific tests; see {doc}`../methods/structural-rules`.
