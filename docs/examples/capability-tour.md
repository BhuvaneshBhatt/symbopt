# Executable capability tour

Use this page to choose a route for a small problem and see what its answer establishes. Every row of the capability matrix links to a section here. Each section checks a mathematical value, witness, enclosure or supported-scope refusal. Run the page with `python tools/check_docs.py` after installing the optional analysis packages for the backend examples.

The page is one Python session. `semialg` and SCIP examples run when those optional backends are installed; the core examples run without them. The installed-wheel dependency jobs execute the same examples outside the source tree.

```python
import sympy as sp
import numpy as np
import symbopt as so

x, y, z = sp.symbols("x y z", real=True)
```

## Structural polynomial rules

A visible sum of nonnegative terms can give an exact value before building a general polynomial system.

```python
visible = so.minimize_result(5 + (x + y - 1) ** 2 + (x - y) ** 4, variables=(x, y))
assert visible.certified and visible.optimum_value == 5
assert visible.best_candidate.primal_values == {
    x: sp.Rational(1, 2),
    y: sp.Rational(1, 2),
}
```

## Univariate polynomial and rational functions

Critical points, poles and endpoint limits determine these infima. Rational expressions retain their denominator exclusions.

```python
polynomial = so.minimize_result(x**4 - 2 * x**2, variables=x)
assert polynomial.certified and polynomial.optimum_value == -1
assert so.argmin(x**4 - 2 * x**2, variables=x) == ({x: -1}, {x: 1})
rational = so.minimize_result(x**2 + 1 / x**2, variables=x)
assert rational.certified and rational.optimum_value == 2
assert set(p[x] for p in so.argmin(x**2 + 1 / x**2, variables=x)) == {-1, 1}
```

## Sublevel search boxes

A proved exterior bound makes an unbounded search finite. A feasible point at the supplied level is also needed before using this reduction in an optimization proof.

```python
objective = x**4 + y**4 - x - y
bounds = so.polynomial_sublevel_bounds(objective, (x, y), 0)
assert objective.subs({x: 0, y: 0}) == 0
assert set(bounds) == {x, y}
assert all(lo < 0 < hi for lo, hi in bounds.values())
radius = bounds[x][1]
# For r=max(abs(x),abs(y))>=1, f >= r**4-2*r.
assert radius >= 1 and radius**4 - 2 * radius > 0
```

## Smooth constrained minima

Equality and inequality models use the original constraints when validating a witness.

```python
equality = so.minimize_result(x * x + y * y, sp.Eq(x + y, 2), variables=(x, y))
assert equality.certified and equality.optimum_value == 2
assert equality.best_candidate.primal_values == {x: 1, y: 1}
inequality = so.minimize_result(x * x + y * y, [x >= 1, y >= 2], variables=(x, y))
assert inequality.certified and inequality.optimum_value == 5
assert inequality.best_candidate.primal_values == {x: 1, y: 2}
```

## Linear programs

The low-level driver returns a numerical candidate. Check its constraints and objective; exact proof information is available through the symbolic pipeline.

```python
lp = so.solve_lp([-2, -1], [[1, 1], [1, 0]], [3, 2])
assert lp.status == "optimal"
assert np.allclose(lp.x, [2, 1], atol=1e-9)
assert abs(lp.objective + 5) < 1e-9
exact_lp = so.minimize_result(
    -2 * x - y, [x >= 0, y >= 0, x + y <= 3, x <= 2], variables=(x, y)
)
assert exact_lp.certified and exact_lp.optimum_value == -5
fractional = so.minimize_result((2 * x + 1) / (x + 1), x >= 0, variables=x)
assert fractional.certified and fractional.optimum_value == 1
assert fractional.best_candidate.primal_values[x] == 0
```

## Convex and parameterized quadratics

The matrix interface minimizes `x.T*Q*x/2+c.T*x`. Parameters need sufficient sign assumptions for a uniform symbolic formula.

```python
qp = so.solve_convex_qp([[2, 0], [0, 4]], [-2, -8])
assert qp.status == "optimal" and np.allclose(qp.x, [1, 2], atol=1e-7)
assert abs(qp.objective + 9) < 1e-8
a = sp.Symbol("a", positive=True)
parameterized = so.minimize_result(a * (x - 2) ** 2, variables=x)
assert parameterized.certified and parameterized.optimum_value == 0
assert parameterized.best_candidate.primal_values[x] == 2
```

## Nonconvex quadratic bounds

Complete face enumeration proves the value over a compact rational polytope. A limited spatial search can instead return a valid interval containing the optimum.

```python
box = [x >= -1, x <= 1, y >= -1, y <= 1]
nonconvex = so.minimize_quadratic(
    x * x - y * y, box, (x, y), backend="faces", certified=True
)
assert nonconvex.objective_value == -1
assert nonconvex.lower_bound == nonconvex.upper_bound == -1
assert so.verify_quadratic_certificate(
    x * x - y * y, nonconvex.certificate, box, (x, y)
)
triangle = [x >= 0, y >= 0, x + y <= 1]
limited = so.minimize_quadratic(
    -x * y, triangle, (x, y), backend="spatial", max_nodes=4, certified=True
)
assert limited.lower_bound <= -sp.Rational(1, 4) <= limited.upper_bound
assert so.verify_quadratic_certificate(-x * y, limited.certificate, triangle, (x, y))
```

## SOS identities and constrained proofs

Replay checks the Gram matrices and polynomial identity. The inequality premises belong to the proof: changing them changes the theorem.

```python
sos = so.sos_decompose((x * x - 1) ** 2, [x])
assert sos.certificate.verify(
    polynomial=(x * x - 1) ** 2, inequalities=(), equalities=()
)
constrained = so.sos_poly_relaxation(
    x * x,
    [x],
    inequalities=[1 - x * x],
    search_lo=-1,
    search_hi=1,
    max_iter=1,
    ineq_mult_degree=0,
)
assert constrained.certified and constrained.lower_bound == 0
infeasible = so.search_infeasibility_certificate(
    [x], [-x - 1], [x], force=True, multiplier_degree=0
)
assert infeasible.certified
assert so.verify_infeasibility_certificate(
    infeasible.certificate, [x], equalities=[x], inequalities=[-x - 1]
)
assert not so.verify_infeasibility_certificate(
    infeasible.certificate, [x], equalities=[x], inequalities=[1 - x]
)
```

## Sparse polynomial structure

Disconnected or overlapping supports reduce the Gram blocks needed for SOS recovery. Structural sparsity alone is not a sign proof.

```python
graph = so.build_sparsity_graph([x * x + y * y, z * z], (x, y, z))
assert {frozenset(component) for component in graph.connected_components()} == {
    frozenset([x]),
    frozenset([y]),
    frozenset([z]),
}
sparse = so.sos_decompose((x + y) ** 2 + (y + z) ** 2, [x, y, z])
assert sparse.certificate.verify(polynomial=(x + y) ** 2 + (y + z) ** 2)
assert len(sparse.certificate.gram_matrices) == 2
```

## Moment and localizing matrices

A populated moment matrix records an actual measure. Atom extraction must reproduce its moments and positive weights.

```python
moments = {(0,): sp.S.One, (1,): sp.Integer(2), (2,): sp.Integer(4)}
moment = so.build_moment_matrix([x], 1, moment_values=moments)
assert moment.matrix == sp.Matrix([[1, 2], [2, 4]])
localizing = so.build_localizing_matrix(x, [x], 1, moment_values=moments)
assert localizing.matrix == sp.Matrix([[2]])
atoms = so.extract_moment_atoms(moment.matrix, moment.basis, [x])
assert atoms.verified and atoms.atoms == [{x: 2}] and atoms.weights == [1]
```

## Numerical dense and sparse SDP

These solvers supply approximate PSD matrices. Their numerical status does not by itself establish an exact SOS theorem.

```python
problem = so.SDPProblem.single_block(
    np.eye(2), [np.diag([1, 0]), np.diag([0, 1])], [1, 2]
)
for solver in (so.solve_sdp_dense, so.solve_sdp_sparse_chordal):
    solution = solver(problem, max_iter=500, tol=1e-8)
    matrix = solution.X_blocks[0]
    assert np.linalg.eigvalsh(matrix).min() >= -1e-6
    assert np.allclose(np.diag(matrix), [1, 2], atol=1e-4)
    assert abs(solution.primal_objective - 3) < 1e-4
```

## Algebraic functions and constants

Lifted graphs keep real-branch restrictions. Fixed real algebraic constants can also appear as exact coefficients.

```python
radical = so.minimize_result((sp.sqrt(x) - 2) ** 2, x >= 0, variables=x)
assert radical.certified and radical.optimum_value == 0
assert radical.best_candidate.primal_values[x] == 4
inverse = so.minimize_result((1 / x - 2) ** 2, variables=x)
assert inverse.certified and inverse.optimum_value == 0
assert inverse.best_candidate.primal_values[x] == sp.Rational(1, 2)
t = sp.Symbol("t")
root = sp.CRootOf(t**3 - t - 1, 0)
algebraic = so.minimize_result((x - root) ** 2, variables=x)
assert algebraic.certified and algebraic.optimum_value == 0
assert algebraic.best_candidate.primal_values[x] == root
```

## Absolute values and piecewise functions

Each branch retains its own domain. A point proposed for an excluded boundary cannot establish attainment.

```python
absolute = so.minimize_result(sp.Abs(x - 1) + sp.Abs(x - 3), variables=x)
assert absolute.certified and absolute.optimum_value == 2
piecewise = so.minimize_result(
    sp.Piecewise((x * x, x < 0), ((x - 2) ** 2 + 1, True)), variables=x
)
assert piecewise.certified and piecewise.optimum_value == 0
assert piecewise.attained is False
assert (
    so.argmin(sp.Piecewise((x * x, x < 0), ((x - 2) ** 2 + 1, True)), variables=x) == ()
)
```

## Bounded transcendental enclosures

Arb arithmetic encloses supported expressions on explicit finite boxes. An enclosure is a bound on that box, and does not prove that the box covers an unconstrained global search.

```python
enclosure = so.enclose_expression(sp.sin(x), (x,), ((0, 1),))
assert enclosure.rigorous
assert sp.Rational(enclosure.lower) <= 0
assert sp.Rational(enclosure.upper) >= sp.sin(1)
```

## Open boundaries and attainment

The infimum need not be a value at a feasible point. Use the detailed result to distinguish a limiting point from an attained minimizer.

```python
open_result = so.minimize_result(x, x > 0, variables=x)
assert open_result.certified and open_result.optimum_value == 0
assert open_result.attained is False
assert so.argmin(x, x > 0, variables=x) == ()
```

## Optional semialgebraic projection

Install `symbopt[semialg]` for these exact backend examples. The same backend handles feasibility, implication, optimization, polynomial signs, graph images and range projection. General QE belongs to semialg; symbopt exposes the operations it needs for optimization.

```python
if so.semialg_available():
    from symbopt.semialgebraic import exact_feasible, minimize_exact

    assert exact_feasible([x * x < 0], [x]) is False
    sign = so.prove_nonnegative_exact(x, [x >= 0], [x])
    assert sign.proven is True
    optimum = minimize_exact(x * x, [x >= 1], [x])
    assert optimum.certified and optimum.value == 1 and optimum.attained
    image = so.function_range_exact(x * x, [x >= -2, x <= 2], [x])
    assert (image.infimum, image.supremum) == (0, 4)
    assert image.infimum_attained and image.supremum_attained
    graph_image = so.function_range_exact(1 / (1 + x * x), [], [x])
    assert (graph_image.infimum, graph_image.supremum) == (0, 1)
    assert (
        graph_image.infimum_attained is False and graph_image.supremum_attained is True
    )
```

## Integer models and parameter scope

Native continuous quadratic methods reject integer variables. Explicit SCIP uses a numerical integer solver. A symbolic parameter with unresolved sign also falls outside the fixed-coefficient global-QP interface; the restricted symbolic rules above are a separate capability.

```python
n = sp.Symbol("n", integer=True)
try:
    so.minimize_quadratic((n - sp.Rational(1, 3)) ** 2, variables=n, backend="faces")
except NotImplementedError:
    pass
else:
    raise AssertionError("continuous quadratic route accepted an integer model")
if so.scip_available():
    integer = so.minimize_quadratic(
        (n - sp.Rational(1, 3)) ** 2, variables=n, backend="scip"
    )
    assert integer.status == "optimal"
    assert abs(float(integer.point[0])) < 1e-8
    assert abs(float(integer.objective_value) - 1/9) < 1e-8
    assert integer.certified is False
    assert abs(float(integer.objective_value) - 1 / 9) < 1e-7
    assert not integer.certified
parameter = sp.Symbol("parameter", real=True)
try:
    so.minimize_quadratic(parameter * x * x, variables=x)
except NotImplementedError:
    pass
else:
    raise AssertionError("fixed-coefficient interface accepted a parameter")
```

## Monotone and periodic reductions

A monotone outer function preserves a proved scalar minimum. Periodic ranges and certified dense product ranges give infima without automatically producing an attaining point.

```python
monotone = so.minimize_result(sp.exp((x - 2) ** 2), variables=x)
assert monotone.certified and monotone.optimum_value == 1
periodic = so.minimize_result(3 * sp.sin(x) + 4 * sp.cos(x), variables=x)
assert periodic.certified and periodic.optimum_value == -5
rational_trig = so.minimize_result(
    sp.sin(x) ** 2 + sp.cos(x) ** 2 + sp.sin(2 * x), variables=x
)
assert rational_trig.certified and rational_trig.optimum_value == 0
mixed = so.minimize_result(sp.sin(x) + sp.cos(sp.sqrt(2) * x), variables=x)
assert mixed.certified and mixed.optimum_value == -2
assert mixed.attained is None
asymptotic = so.minimize_result(sp.exp(x), variables=x)
assert asymptotic.certified and asymptotic.optimum_value == 0
assert asymptotic.attained is False
```

## Analysis and scope boundaries

The univariate analysis interface caches source-domain facts. Its optional provider is funcprops. Dependent inverse periods do not justify a product-range argument, and the univariate asymptotic route does not claim a multivariate theorem.

```python
from symbopt.univariate_analysis import analyze_univariate
from symbopt.mixed_periodic import decompose_mixed_periodic
from symbopt.transcendental_analysis import try_transcendental

analysis = analyze_univariate(x * x, [], x)
assert analysis.feasible_domain == sp.S.Reals
dependent = sp.sin(x) + sp.sin(sp.sqrt(2) * x) + sp.sin((1 + sp.sqrt(2)) * x)
assert decompose_mixed_periodic(dependent, x) is None
assert try_transcendental(sp.exp(x * y), [], (x, y)) is None
```
