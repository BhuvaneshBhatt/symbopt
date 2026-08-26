# Understanding results

`minimize()` and `maximize()` return a `PipelineResult`. The object separates the
**global value claim**, **attainment**, **represented candidates**, and the
**evidence used for certification**.

## Start with the global value

The preferred high-level fields are:

```python
result.optimum_value
result.attained
result.certified
result.status
```

`optimum_value` can be known even when no optimizer exists. For example, the
infimum of \(x\) over \(x>0\) is 0 but is not attained. In that case the result
can have `optimum_value == 0`, `attained is False`, and `best_candidate is None`.

## Candidate solutions

`result.best_candidate` is a `CandidateSolution` when a represented feasible
candidate exists. Important fields include:

- `primal_values`: values in the lifted problem,
- `original_variable_values`: projection to the original variables,
- `objective_value`: symbolic/exactified objective value,
- `objective_value_numeric`: numerical ordering value,
- `source`: candidate-generation route,
- `local_classification`: local classification when available.

A candidate can exist without a rigorous global certificate, and an exact
unattained infimum can exist without a candidate.

## Optimization sense and KKT provenance

`result.optimization_sense` is either `"minimize"` or `"maximize"`.

KKT construction is lazy, so `result.kkt_system` may be `None` when an earlier
global route solved the problem. If maximization required KKT work,
`result.kkt_objective` records the normalized minimization objective used to
construct those equations. This keeps the public objective sense distinct from
the internal KKT sign convention.

## Certification

`result.certified` is `True` only when an exact proof establishes the relevant
global optimum/infimum/supremum claim. A merely small rigorous numerical gap is
reported as a bound, not promoted to exact certification.

`result.certificate` can contain:

- a primal solution/value when attained,
- lower and upper bounds,
- a certification gap,
- a proof status,
- proof artifacts identifying the contributing methods.

For minimization the core inequality is

\[
L\le f^\star\le U.
\]

A feasible candidate supplies an upper bound. Exact SOS proofs, exact
semialgebraic reasoning, and rigorous interval enclosures can supply lower-bound
or global-value evidence depending on the route.

## Exact, rigorous numerical, and heuristic evidence

**Exact symbolic evidence**
: algebraic substitution, exact polynomial identities, exact recovered Gram
  matrices, exact semialgebraic proofs, or equivalent symbolic statements.

**Rigorous numerical evidence**
: verified interval/ball enclosures with guaranteed outward bounds.

**Heuristic numerical evidence**
: floating-point convergence, approximate SDP residuals, sampling, or local
  solver information.

Heuristic evidence may guide search but does not by itself set `certified=True`.

## Precision provenance

`result.working_precision` records the requested decimal precision, while
`result.precision_provenance` records input precision and backend-routing
semantics. In particular, high-precision inexact inputs can disable float64-only
LP/QP/SOS proposal paths rather than being silently downcast.

## Solver-specific provenance

Depending on routing, a result may also contain:

- `structural_result`,
- `semialg_result` and `semialg_range`,
- `sos_result` and SOS lower-bound data,
- `interval_result`,
- `critical_point_results`,
- typed `diagnostics`.

These fields explain how the pipeline arrived at its global claim or why a route
was skipped or failed.

## Strict constraints and non-attainment

Strict inequalities are preserved semantically. Closure optimization may be used
to determine a boundary infimum, but `attained=False` prevents that boundary
point from being reported as an optimizer of the original strict problem.

At certificate level, `proven_optimal` denotes an attained exact optimum, while
`proven_infimum` denotes an exact infimum that need not be attained.

See {doc}`../guide/certification` for complete proof semantics.
