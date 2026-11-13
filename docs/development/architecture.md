# Architecture

`symbopt` is organized around structure detection, candidate generation, and proof-oriented certification.

```text
SymPy objective + constraints
            |
            v
       algebraize.py
            |
            v
    problem_models.py
            |
            v
    solver_pipeline.py
      /       |        \
     /        |         \
structural   LP/QP   symbolic/polynomial
 routing             candidate search
    |                    |
    |              candidates.py
    |                    |
    |            critical_points.py
    |                    |
    |                  kkt.py
    |                    |
    |              elimination.py
    |
    +-- quadratic_optimization.py
    +-- linear_fractional.py
    +-- trigonometric_optimization.py
    +-- mixed_periodic.py
    +-- transcendental_analysis.py
    +-- univariate_analysis.py
    +-- structural_routing.py
                         |
             +-----------+-----------+
             |                       |
           SOS                    intervals
        sos.py /               intervals.py /
      sos_program.py       verified_intervals.py
             |                       |
             v                       v
           SDP                  Arb/Bernstein
        sdp.py /
    semidefinite.py
             |
             +-----------+
                         |
                       moments
              moment.py / moments_dual.py
                         |
                         v
                  certification.py
                         |
                         v
                   PipelineResult
```

## High-level orchestration

`solver_pipeline.py` owns the public extremum pipeline and coordinates solver families through stage helpers. Problem preparation, early global routing, KKT candidate search, SOS search, interval search, and proof combination are kept as separate stages so each stage can reuse already-computed symbolic data.

Specialized global recognition is coordinated by `structural_routing.py`. Parameterized quadratic, linear-fractional, commensurate trigonometric, mixed-periodic, and transcendental solvers remain separate. `univariate_analysis.py` owns a cached `UnivariateAnalysis` for each normalized one-variable problem. It lazily obtains natural/feasible domains, periodicity, discontinuities and singularities, monotonic partitions, endpoint limits, stationary points, extremum envelopes, and exact ranges from `funcprops`, with SymPy fallbacks on supported real-operand domains. Structural solvers consume the same analysis instance for source domains, images and critical geometry.

## Algebraization boundary

`algebraize.py` converts selected nonpolynomial expressions into a `LiftedProblem`. Already-polynomial problems use a no-lift fast path. When lifts are needed, quotient-ring reduction compiles one Gröbner reducer and reuses it throughout the transformation. The lifted representation records auxiliary variables, defining equalities, branch inequalities, strict inequalities, graph relations, lift metadata, and cached residual/structural metadata.

Cached residual groups are also the canonical input to KKT and SOS preparation. This avoids repeatedly expanding the same relations after algebraization.

## Candidate generation

`candidates.py` solves polynomial/KKT systems through symbolic solving, elimination, active-set enumeration, and numerical fallback. Exact and numerical search consume the same active-set plan; numerical KKT is omitted only after a complete exact search. The KKT system itself is constructed lazily, only when a candidate-search route requires it. Candidates are filtered against lifted constraints before local classification and before entering the integrated result.

## Certification boundary

A candidate is not automatically a proof. Certification combines feasible upper bounds with rigorous lower-bound mechanisms such as exact SOS certificates or verified interval enclosures.

## Numerical backends

The LP and QP drivers use mature SciPy numerical backends for general continuous linear/convex-quadratic subproblems. The SDP solver is implemented inside the package and is more limited in scale.

## Public interface

`symbopt.api.__all__` is the explicit root-level public interface. Internal solver helpers and implementation modules are imported directly from their modules during package development and are not added to the root export list by implicit namespace discovery.

## Invariants worth preserving

When extending the codebase:

1. Normalized inequalities use \(h(x)\ge0\).
2. Numerical evidence must not be promoted to exact proof without verification.
3. Strict-constraint closure semantics must remain explicit.
4. Public result objects should preserve enough provenance to explain how a conclusion was obtained.
5. Solver-specific code should not leak unnecessary assumptions into the high-level problem representation.
6. Cached symbolic residuals should be reused rather than recomputed downstream.

## Exact semialgebraic backend

`semialgebraic.py` is the adapter boundary to the optional `semialg` package. It owns lazy imports, formula construction, exact feasibility/implication, RUR result conversion, CAD/QE candidate certification, and exact optimization result conversion. The rest of the pipeline depends on the adapter rather than on `semialg` implementation details.

`rational_optimization.py` owns rational domain preservation, exact univariate critical/boundary enumeration, polynomial graph construction and the optional semialg graph-image adapter. `sublevel_bounds.py` derives and replays sufficient exterior bounds for structured even-degree polynomials, exposing an exact sublevel box and attaching the feasible-level proof when the dispatcher uses that box. These modules keep optimization orchestration in symbopt and delegate real graph projection to semialg.

The early rule engine is split by mathematical role: `rule_utils.py` shares domain and witness checks, `elementary_rules.py` handles visible inequalities and rays, `box_rules.py` handles Cartesian domains, `spectral_rules.py` handles small matrix formulas, `convex_rules.py` checks curvature and sufficient KKT conditions, and `piecewise_linear_rules.py` and `shape_reductions.py` provide verified reductions. `fast_rules.py` orders these recognizers and enforces work budgets before the ordinary portfolio.

## Mathematical boundary checks

Compact queries return plain lists, and explicit result APIs expose candidates and proof metadata. Source-domain records keep real-valued outputs with complex intermediate operands; a real-operand continuous-domain calculation is not an exhaustive replacement. Exact endpoints enter numerical boxes through shared outward rounding in `precision.py`. SOS records bind the polynomial identity and its domain premises, and replay does not call proposal solvers.
