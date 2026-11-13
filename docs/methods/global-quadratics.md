# Global quadratic optimization

Use global quadratic search when a quadratic objective has negative curvature or quadratic constraints make the feasible set nonconvex. Local stationarity and a successful local numerical solve do not establish a global minimum. The global API retains a feasible incumbent and a lower bound so you can see what has been proved and what remains unresolved.

`minimize_quadratic` and `verify_quadratic_certificate` accept `region=` in addition to constraints and bounds. Membership must reduce to one conjunction of quadratic relations within the selected backend's scope. Use the same region during proof replay. See {doc}`../guide/regions` for an executable example and coordinate rules.

## Exact quadratic programs

```python
import sympy as sp
from symbopt import minimize_quadratic, solve_global_qp, verify_quadratic_certificate

x, y = sp.symbols("x y", real=True)
constraints = [x >= 0, y >= 0, x + y <= 1]
result = minimize_quadratic(
    x**2 - y**2, constraints, (x, y), backend="faces", certified=True
)
assert result.objective_value == -1
assert result.point == (0, 1)
assert result.exact and result.certified
assert verify_quadratic_certificate(
    x**2 - y**2, result.certificate, constraints, (x, y)
)

# Matrix convention: x.T*Q*x/2 + c.T*x. Coordinates are free by default.
matrix_result = solve_global_qp([[2, 0], [0, -2]], [0, 0], bounds=[(-1, 1), (-1, 1)])
assert matrix_result.objective_value == -1 and matrix_result.exact
assert not matrix_result.certified  # independent proof replay is opt-in
```

The face solver enumerates independent active constraint sets on a bounded rational polytope. On each face it solves the affine stationary system and checks feasibility, including singular stationary families. Every global minimizer is stationary on its smallest face, so complete enumeration proves the minimum without assuming positive curvature or nondegenerate KKT multipliers. Rational LP dual witnesses prove the coordinate bounds; Farkas witnesses justify infeasible faces. Equality-only problems and explicit feasible descending rays have separate exact linear-algebra proofs.

A face limit stops enumeration and returns a valid interval lower bound and feasible upper bound. Its incumbent is not presented as a complete face-search optimum. Enumeration is combinatorial; this method is useful for small exact problems.

## Quadratic constraints and spatial search

```python
nonconvex = [x * x + y * y >= 1]
result = minimize_quadratic(
    x * x + y * y, nonconvex, (x, y), backend="spatial", certified=True
)
assert result.lower_bound == result.upper_bound == 1
assert result.exact and result.certified

# A finite budget still returns bounds enclosing the true optimum -1/2.
ball = [x * x + y * y <= 1]
limited = minimize_quadratic(
    -x * y, ball, (x, y), backend="spatial", max_nodes=8, certified=True
)
assert limited.lower_bound <= -sp.Rational(1, 2) <= limited.upper_bound < 0
assert limited.certified and not limited.exact
assert verify_quadratic_certificate(-x * y, limited.certificate, ball, (x, y))
```

Spatial search partitions a rational box and uses shared product variables with McCormick envelopes and square tangents to construct linear relaxations. A numerical LP basis can propose a solution; only reconstructed rational primal and dual evidence supplies a bound. Exact interval arithmetic supplies a weaker bound if reconstruction fails. Numerical local searches likewise supply proposals whose original feasibility must verify exactly before they improve the upper bound.

The search box must follow from affine constraints, a positive-definite ellipsoid constraint, or a positive-definite objective sublevel containing an exactly checked feasible witness. The last reduction covers every improving point and can handle an unbounded feasible set. Arbitrary search hints never truncate the mathematical domain. Missing finite domain reductions return `unsupported` with any available incumbent.

## Choosing a backend

| Backend | Coefficients and domains | Global evidence | Main limit |
|---|---|---|---|
| `faces` | rational continuous QP with closed affine constraints | exact full-face proof, equality reduction or descending ray | bounded polytope required for enumeration; exponential face count |
| `spatial` | rational continuous closed QCQP | exact bounds and partition replay; exact optimum when bounds meet | proved finite domain reduction; node and gap budgets |
| `scip` | fixed finite real QP/QCQP, including integer decision symbols | numerical primal and dual bounds | optional native solver, tolerances, no exact certification |
| `auto` | fixed finite real data | exact methods first; spatial for rational continuous data; optional SCIP otherwise | explicit `scip` required for integer decisions |

Install the optional solver with `python -m pip install "symbopt[scip]"`. Use `scip_available()` to check whether its native library imports. For integer quadratic optimization, create symbols with `integer=True` and explicitly select `backend="scip"`. The native backends require closed constraints. SCIP converts strict rational polynomial rows involving only integer variables by clearing denominators and enforcing a unit lattice margin. Continuous strict inequalities and excluded boundaries remain unsupported by the adapter. Specialized symbolic routes elsewhere in symbopt still handle supported open-domain infima.

## Result and proof semantics

`GlobalQuadraticResult.objective_value` is the feasible incumbent value, or `-oo` for a proved escaping ray. `lower_bound`, `upper_bound`, `gap`, `status` and `exact` describe the search outcome. `certified=False` is the default; `certified=True` replays native evidence. `bounds_verified` distinguishes native rational bounds from SCIP's numerical bounds. Positive-gap native results can have verified bounds and `certified=True` while `exact=False`; this certifies the enclosure, not equality with the incumbent. A numerical SCIP optimum always has `certified=False`, `bounds_verified=False` and `exact=False`.

`max_faces`, `max_nodes`, `time_limit`, `absolute_gap` and `relative_gap` control work. Time limits are checked between subproblems; an individual exact LP, symbolic solve or incumbent proposal can overrun the requested wall time. A positive requested gap does not imply finite termination or a closed-form optimum on every quadratic model. Algebraic boundary feasibility, weak relaxations, unbounded domains without a proved reduction and larger problems can remain unresolved.

The ordinary symbolic portfolio automatically tries a small exact face search after cheaper rules. Explicit selection uses `options={"quadratic": {"backend": "spatial", "max_nodes": 128, "certified": True}}` with `minimize_result` or `maximize_result`. The detailed pipeline exposes the quadratic search result in `structural_result.metadata["quadratic_result"]`. Partial and numerical searches retain candidate values and bounds without populating the pipeline's exact `optimum_value`.
