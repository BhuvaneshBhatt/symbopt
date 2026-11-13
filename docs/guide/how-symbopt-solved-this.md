# How symbopt solved this

A useful way to understand `symbopt` is to follow the decision chain rather than memorize a list of solvers. The high-level pipeline normalizes the problem, measures its structure, chooses inexpensive routes first, generates candidates, and only then decides whether the available evidence is strong enough to certify a global answer.

## Example 1: a convex quadratic

```python
import sympy as sp
from symbopt import minimize_result

x, y = sp.symbols("x y", real=True)
r = minimize_result(
    (x - 1) ** 2 + (y - 2) ** 2,
    [x >= 0, y >= 0, x + y <= 4],
    variables=[x, y],
)

print(r.optimum_value)
print(r.method)
print(r.explain())
```

The decision chain is:

```text
relations + variables
→ normalized continuous problem
→ polynomial degree 2 / convex quadratic structure
→ direct QP route before symbolic KKT or SOS
→ exact postprocessing of the numerical candidate
→ certified optimum and optimizer
```

The planner still exists for this problem, but its job is mostly to prevent expensive fallback work if the direct route fails:

```python
print(r.solver_plan.complexity)
print(r.solver_plan.reasons)
```

Planner decisions are performance decisions, not mathematical impossibility proofs. Explicit backend requests may still ask for a route that `auto` would defer.

## Example 2: one polynomial, several proof routes

Consider

\[
p(x,y)=(x+y)^2+(x-y)^2=2x^2+2y^2.
\]

The mathematical answer is obvious, but it is a useful cross-check because several subsystems can support it.

```python
from symbopt import SDPSolverConfig, sos_decompose

p = (x + y) ** 2 + (x - y) ** 2
r = minimize_result(p, variables=[x, y])
assert r.optimum_value == 0
```

The ordinary pipeline can certify the minimum structurally or through exact critical-point reasoning. An SOS route instead asks for a Gram identity

\[
p=z^TQz,\qquad Q\succeq0.
\]

```python
sos = sos_decompose(p, [x, y])
assert sos.certificate is not None
assert sos.certificate.verified
```

If CVXPY plus Clarabel or SCS is installed, the numerical SDP solve may be outsourced:

```python
from symbopt import available_external_sdp_solvers

solvers = available_external_sdp_solvers()
if "CLARABEL" in solvers:
    sos = sos_decompose(
        p,
        [x, y],
        solver_config=SDPSolverConfig(
            backend="cvxpy",
            external_solver="CLARABEL",
            tol=1e-8,
        ),
    )
```

The important point is that Clarabel or SCS does **not** create the final proof. The numerical candidate is followed by exact affine reconstruction, exact polynomial-identity verification, and exact PSD verification. Only the last stage makes `certificate.verified` true. See {doc}`../methods/semidefinite-programming` for the canonical SDP backend and post-verification description.

When `semialg` is installed, exact semialgebraic feasibility/range/optimization is another possible proof source. symbopt owns the optimization strategy and result semantics; semialg owns the underlying CAD/QE and real-algebraic reasoning.

## Example 3: bounded transcendental optimization

```python
from symbopt import enclose_expression

bound = enclose_expression(sp.sin(x) + x**2, (x,), ((-1.0, 1.0),))
print(bound)
```

For a finite box and a supported function library, Arb-backed interval evaluation gives rigorous enclosures. The high-level optimizer can combine those enclosures with branch-and-bound, monotonicity, Taylor/Bernstein tightening, or interval contraction. If the box is not finite, the automatic planner will not pretend interval branch-and-bound is applicable.

## Reading a detailed result

For debugging or reproducibility, the useful inspection order is:

```python
r.method
r.status
r.certified
r.attained
r.infeasible
r.unbounded
r.solver_plan
r.diagnostics
r.certificate
r.explain()
```

`method` is a stable user-facing route label. `status` may be more specific and is not intended as the primary programmatic interface.

## What happens when no proof route closes?

symbopt may still have a useful local or numerical candidate, but it does not promote that evidence to a certified global answer. This distinction is central to the package: candidate generation and proof are separate stages.

For the canonical routing policy, see {doc}`solver-selection`. For proof semantics, see {doc}`certification`. For unsupported families and scaling limits, see {doc}`../reference/limitations`.
