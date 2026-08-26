# Solver selection

`minimize()` is a dispatcher, not a single algorithm. It recognizes structure,
chooses a cheap justified route when possible, generates candidates when needed,
and seeks rigorous global evidence without confusing numerical convergence with
proof.

## Routing summary

A simplified order is:

1. normalize the problem and explicit decision variables;
2. detect precision-sensitive routing requirements;
3. exploit separability and exact structural reductions;
4. recognize LP and convex-QP problems;
5. selectively use the optional exact semialgebraic backend when its expected
   cost is appropriate or the user explicitly requests it;
6. construct KKT/candidate systems only when they are actually needed;
7. use SOS/moment and verified interval methods for global bounds/certification;
8. combine independent proof artifacts into the final certificate.

The exact path depends on problem structure and available optional backends.

| Problem structure | Typical primary route | Certification route |
|---|---|---|
| Linear objective + affine constraints | LP | exact/verified postprocessing |
| Convex quadratic + affine constraints | convex QP | convexity + exact/verified checks |
| Linear-fractional + affine constraints | structural Charnes--Cooper | exact branch comparison |
| Smooth symbolic | critical/KKT candidates | symbolic classification or global fallback |
| Small polynomial system | symbolic KKT/elimination may run first | exact comparison, SOS, intervals, semialg |
| Larger/exact semialgebraic | cost-aware semialg or other polynomial routes | CAD/QE/range, SOS, intervals |
| Rational-frequency trig polynomial | tangent-half-angle structural route | exact algebraic comparison |
| Common univariate rational/log/exp | domain/pole/convexity route | exact limits/stationary analysis |
| Bounded supported transcendental | interval branch-and-bound | Arb/verified enclosures |
| Additively separable | component decomposition | every component must certify |

## Direct LP/QP recognition

Affine constraints are routed to LP/QP before expensive semialgebraic redundancy
checks. LP and QP solvers already tolerate redundant affine rows, so sending
those rows through CAD/QE first would add cost without improving correctness.

If the objective is quadratic, the symmetric Hessian must satisfy the convex-QP
preconditions. Otherwise routing falls through to more general methods.

## Structural fast paths

Before general KKT/SOS/interval search, symbopt recognizes globally justified
special structures, including linear-fractional programs, monotone outer
compositions, rational-frequency trigonometric polynomials, common univariate
transcendental/rational forms, and independent-variable decomposition.

See {doc}`../methods/structural-routes` for mathematical details.

## Candidate generation and lazy KKT

For unresolved problems, symbopt can use exact polynomial candidate extraction,
KKT systems, active-set enumeration, elimination, and numerical `nsolve`
fallbacks.

KKT construction is **lazy**. If an LP/QP, structural route, or exact backend
already resolves the problem, `PipelineResult.kkt_system` remains `None`. This
avoids symbolic differentiation that cannot contribute to the answer.

Active-set planning itself is deterministic and does not depend on CAD/QE
pruning. Cheap box/residual activity information can prioritize or eliminate
patterns; exact semialgebraic reasoning remains available at solver and
certification stages.

Candidate generation does not imply global optimality.

## Semialgebraic backend policy

The optional `semialg` backend is capability-discovered at runtime. symbopt does
not assume that a particular function must be re-exported from the top-level
`semialg` module; it adapts to the APIs exposed by the installed backend.

`semialg_certification` accepts:

- `"auto"`: cost-aware routing;
- `"candidate"`: request the backend's candidate-level exact optimization mode;
- `"complete"`: request complete exact certification where supported.

In `"auto"` mode, an exact polynomial problem is **not automatically sent to CAD
first**. Small systems with inexpensive symbolic KKT/elimination may use that
route before a costly exact range computation. Conversely, suitable
higher-dimensional semialgebraic problems can route directly to semialg when
that is the better exact backend. `semialg_range_cost_limit` can override the
backend's automatic range-cost threshold.

A certified semialg result preserves exact value, attainment, method,
diagnostics, backend version, and capability provenance. An uncertified
candidate result may seed feasible upper-bound candidates but is not promoted to
an exact lower bound.

Standalone feasibility, implication, sign, range, and exact-point helpers remain
available where supported, but the high-level pipeline avoids redundantly
re-running them after an authoritative direct optimizer result.

## Global certification

For polynomial problems, SOS hierarchy search can produce lower bounds. Only an
exactly recovered polynomial identity with exact PSD verification counts as an
exact SOS proof.

For finite boxes, the interval engine supplies rigorous enclosures. Polynomial
boxes can use Bernstein/Taylor tightening, while supported transcendental
expressions use Arb real-ball evaluation. Equality systems can be contracted
with interval Newton or Krawczyk methods before subdivision.

## Precision-sensitive routing

`working_precision` and the precision of inexact inputs affect route selection.
If high-precision inexact input is present, float64-only LP/QP/SOS proposal
routes are bypassed rather than silently discarding precision. Exact input can
still use fast floating-point proposal stages when the final result is
independently exact-certified.

## Why a method may be skipped

A route can be skipped because its structural preconditions are not proved, the
problem would lose requested precision, the required backend capability is not
available, finite interval bounds cannot be justified, a relaxation is too
large/unsupported, or exact certificate recovery fails. `PipelineResult` keeps
typed diagnostics and provenance so these choices remain inspectable.
