# Solver selection

`minimize()` is a dispatcher, not a single algorithm. It recognizes structure, chooses a cheap justified route when possible, generates candidates when needed, and seeks rigorous global evidence without confusing numerical convergence with proof.

## Early structural proofs

Before algebraization and full complexity planning, the dispatcher tries bounded structural recognizers for leading terms, nonnegative forms, scalar images, product boxes, piecewise linear objectives, spectral formulas and classical inequalities. They return only with an exact bound/value proof and checked attainment information. See {doc}`../methods/structural-rules` for the capability matrix, budgets and `options={"structural":{"enabled":False}}` control. Exact unconstrained quadratic algebra and sufficient convex KKT certification also improve the ordinary direct/candidate paths.

## Routing summary

A simplified order is:

1. normalize the problem and explicit decision variables;
2. detect precision-sensitive routing requirements;
3. compute a cheap package-wide complexity plan (dimension, degree, constraint count, active-set estimate, elimination score, SOS Gram size, sparsity, semialg availability, and finite-box status);
4. exploit separability and exact structural reductions;
5. recognize LP and convex-QP problems;
6. selectively use the optional exact semialgebraic backend when its expected cost is appropriate or the user explicitly requests it;
7. construct KKT/candidate systems only when the planner keeps them within budget;
8. use SOS/moment and verified interval methods when their estimated sizes and domain requirements are appropriate;
9. combine independent proof artifacts into the final certificate.

The exact path depends on problem structure and available optional backends.

| Problem structure | Typical primary route | Certification route |
|---|---|---|
| Linear objective + affine constraints | LP | exact/verified postprocessing |
| Convex quadratic + affine constraints | convex QP | convexity + exact/verified checks |
| Linear-fractional + affine constraints | structural Charnes–Cooper | exact branch comparison |
| Smooth symbolic | critical/KKT candidates | symbolic classification or global fallback |
| Small polynomial system | symbolic KKT/elimination may run first | exact comparison, SOS, intervals, semialg |
| Larger/exact semialgebraic | cost-aware semialg or other polynomial routes | CAD/QE/range, SOS, intervals |
| Commensurate rational trigonometric expression | tangent-half-angle rational structural route | exact algebraic comparison plus certified pole limits |
| Common univariate rational/log/exp | domain/pole/convexity route | exact limits/stationary analysis |
| Bounded supported transcendental | interval branch-and-bound | Arb/verified enclosures |
| Additively separable | component decomposition | every component must certify |

## Direct LP/QP recognition

Affine constraints are routed to LP/QP before expensive semialgebraic redundancy checks. LP and QP solvers already tolerate redundant affine rows, so sending those rows through CAD/QE first would add cost without improving correctness.

If the objective is quadratic, the symmetric Hessian must satisfy the convex-QP preconditions. Otherwise routing falls through to more general methods.

## Structural fast paths

Before general KKT/SOS/interval search, symbopt recognizes globally justified special structures, including linear-fractional programs, monotone outer compositions, commensurate rational trigonometric expressions, certified mixed-periodic decompositions, common univariate transcendental/rational forms, and independent-variable decomposition.

See {doc}`../methods/structural-routes` for mathematical details.

## Exact polynomial and rational routes

The univariate rational route enumerates algebraic critical points, constraint cuts, excluded denominator zeros and one-sided limits. Direct LP/QP postprocessing keeps priority for closed affine quadratic problems; exact univariate enumeration can replace an unresolved direct result. The rational graph route projects `t*q=p` together with nonzero-denominator and feasibility conditions through semialg. Its default degree/dimension estimate limit is 12; `semialg_range_cost_limit` overrides that estimate and `semialg_certification="complete"` removes the default limit. Candidate mode does not request graph projection.

For an unresolved polynomial problem without a finite box, the dispatcher can prove that an improving sublevel set lies in a finite box. This requires an independently checked exact feasible value and a supported positive leading-form bound. The proof is retained in the result, the interval plan is refreshed, and subsequent interval or SOS evidence must still establish the claimed optimum or gap. See {doc}`../methods/polynomial-rational`.

## Candidate generation and lazy KKT

For unresolved problems, symbopt can use exact polynomial candidate extraction, KKT systems, active-set enumeration, elimination, and numerical `nsolve` fallbacks.

KKT construction is **lazy**. If an LP/QP, structural route, or exact backend already resolves the problem, `PipelineResult.kkt_system` remains `None`. This avoids symbolic differentiation that cannot contribute to the answer.

Active-set planning itself is deterministic and does not depend on CAD/QE pruning. Cheap box/residual activity information can prioritize or eliminate patterns; exact semialgebraic reasoning remains available at solver and certification stages.

Candidate generation does not imply global optimality.

## Semialgebraic backend policy

The optional `semialg` backend is capability-discovered at runtime. symbopt does not assume that a particular function must be re-exported from the top-level `semialg` module; it adapts to the APIs exposed by the installed backend.

`semialg_certification` accepts:

- `"auto"`: cost-aware routing;
- `"candidate"`: request the backend's candidate-level exact optimization mode;
- `"complete"`: request complete exact certification where supported.

In `"auto"` mode, an exact polynomial problem is **not automatically sent to CAD first**. Small systems with inexpensive symbolic KKT/elimination may use that route before a costly exact range computation. Conversely, suitable higher-dimensional semialgebraic problems can route directly to semialg when that is the better exact backend. `semialg_range_cost_limit` can override the backend's automatic range-cost threshold.

A certified semialg result preserves exact value, attainment, method, diagnostics, backend version, and capability provenance. An uncertified candidate result may seed feasible upper-bound candidates but is not promoted to an exact lower bound.

Standalone feasibility, implication, sign, range, and exact-point helpers remain available where supported, but the high-level pipeline avoids redundantly re-running them after an authoritative direct optimizer result.

## Global certification

For polynomial problems, SOS hierarchy search can produce lower bounds. Only an exactly recovered polynomial identity with exact PSD verification counts as an exact SOS proof.

For finite boxes, the interval engine supplies rigorous enclosures. Polynomial boxes can use Bernstein/Taylor tightening, while supported transcendental expressions use Arb real-ball evaluation. Equality systems can be contracted with interval Newton or Krawczyk methods before subdivision.

## Precision-sensitive routing

`working_precision` and the precision of inexact inputs affect route selection. If high-precision inexact input is present, float64-only LP/QP/SOS proposal routes are bypassed rather than silently discarding precision. Exact input can still use fast floating-point proposal stages when the final result is independently exact-certified.

## Why a method may be skipped

A route can be skipped because its structural preconditions are not proved, the problem would lose requested precision, the required backend capability is not available, finite interval bounds cannot be justified, a relaxation is too large/unsupported, or exact certificate recovery fails. `PipelineResult` keeps typed diagnostics and provenance so these choices remain inspectable.

## Automatic complexity planner

`plan_problem()` exposes the same cheap structural analysis used by `method="auto"`. The returned `SolverPlan` records route decisions and reasons, while full pipeline results retain the plan in `result.solver_plan` and include it in `result.explain()`. Planner deferral is a performance decision, not a proof of mathematical inapplicability; explicit backend policies can still request expensive routes.

## Canonical ownership of routing policy

This page is the canonical description of **when** the high-level pipeline chooses or defers a route. Method pages describe how individual algorithms work; {doc}`../reference/limitations` describes mathematical and scaling boundaries without duplicating this routing policy.

The planner currently considers variable and constraint counts, polynomial degree, monomial density, estimated active-set count, estimated KKT/Gröbner complexity, SOS Gram dimensions, semialg availability, and finite-box availability. Inspect the decision directly:

```python
import sympy as sp
from symbopt import plan_problem

x, y = sp.symbols("x y", real=True)
objective = (x - 1) ** 4 + (y + 1) ** 4
constraints = [x >= -2, x <= 2, y >= -2, y <= 2]
variables = (x, y)
box_bounds = {x: (-2, 2), y: (-2, 2)}
plan = plan_problem(objective, constraints, variables, box_bounds)
print(plan.complexity)
print(plan.reasons)
print(plan.explain())
```
