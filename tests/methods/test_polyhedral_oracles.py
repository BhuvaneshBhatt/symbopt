"""Independent rational oracles for numerical LP/QP and native quadratic bounds."""

from itertools import combinations
from random import Random

import numpy as np
import pytest
import sympy as sp

from symbopt import (
    minimize_quadratic,
    revised_simplex,
    simplex,
    solve_convex_qp,
    solve_lp,
    verify_quadratic_certificate,
)


def stationary_values(matrix, linear, rows, rhs, equalities=(), targets=()):
    """Enumerate exact feasible stationary points on independent faces.

    All test domains are compact. A quadratic minimum lies on a face where its
    tangent derivative vanishes. Singular stationary families have a constant
    objective and reach a lower-dimensional face, so nonsingular systems plus
    vertices suffice here. No production optimizer or floating solver is used.
    """
    size = len(linear)
    q, c = sp.Matrix(matrix), sp.Matrix(linear)
    result = []
    for count in range(size - len(equalities) + 1):
        for active in combinations(range(len(rows)), count):
            selected = [*equalities, *(rows[i] for i in active)]
            values = [*targets, *(rhs[i] for i in active)]
            a = sp.Matrix(selected) if selected else sp.zeros(0, size)
            if a.rank() != a.rows:
                continue
            kkt = q.row_join(a.T).col_join(a.row_join(sp.zeros(a.rows)))
            if kkt.det() == 0:
                continue
            solution = kkt.inv() * sp.Matrix([*-c, *values])
            point = solution[:size]
            if any(
                sum(v * x for v, x in zip(row, point)) > b for row, b in zip(rows, rhs)
            ):
                continue
            vector = sp.Matrix(point)
            value = (vector.T * q * vector)[0] / 2 + (c.T * vector)[0]
            result.append((value, point))
    if not result:
        raise ValueError("test oracle requires a nonempty compact polytope")
    return min(value for value, _ in result)


def transformed(matrix, linear, rows, rhs, bounds, kind, equalities=(), targets=()):
    """Apply an equivalent row/coordinate transformation, or positive cost scaling."""
    q, c, a, b = map(np.asarray, (matrix, linear, rows, rhs))
    e = np.asarray(equalities).reshape((-1, len(c)))
    h = np.asarray(targets)
    scale = 1
    if kind == "rows":
        factors = np.arange(1, len(a) + 1)
        a, b = (a * factors[:, None])[::-1], (b * factors)[::-1]
        if len(e):
            e, h = -3 * e, -3 * h
    elif kind == "coordinates":
        q, c, a, e = q[::-1, ::-1], c[::-1], a[:, ::-1], e[:, ::-1]
        bounds = bounds[::-1]
    elif kind == "objective":
        scale = 3
        q, c = scale * q, scale * c
    return q, c, a, b, bounds, e, h, scale


@pytest.fixture(params=range(8), ids=lambda seed: f"polytope-{seed}")
def linear_case(request):
    rng = Random(37 + request.param)
    size = 2 + request.param % 2
    rows = [[rng.randint(-4, 4) for _ in range(size)] for _ in range(3)]
    rhs = [rng.randint(0, 7) for _ in rows]
    # The origin is feasible, and explicit positive upper bounds make the
    # polytope compact while keeping a valid initial simplex slack basis.
    rows += sp.eye(size).tolist()
    rhs += [rng.randint(1, 6) for _ in range(size)]
    linear = [rng.randint(-5, 5) for _ in range(size)]
    oracle_rows = [*rows, *(-sp.eye(size)).tolist()]
    expected = stationary_values(
        sp.zeros(size), linear, oracle_rows, [*rhs, *([0] * size)]
    )
    return linear, rows, rhs, expected


@pytest.mark.parametrize("solver", [simplex, revised_simplex, solve_lp])
@pytest.mark.parametrize("kind", ["original", "rows", "coordinates", "objective"])
def test_lp_differential(linear_case, solver, kind):
    linear, rows, rhs, expected = linear_case
    size = len(linear)
    _, c, a, b, _, _, _, scale = transformed(
        sp.zeros(size).tolist(), linear, rows, rhs, [(0, None)] * size, kind
    )
    result = solver(c, a, b)
    assert result.status == "optimal"
    assert np.min(result.x) >= -1e-8
    assert np.all(a @ result.x <= b + 1e-8)
    assert c @ result.x == pytest.approx(float(scale * expected), abs=1e-8)
    assert result.objective == pytest.approx(float(scale * expected), abs=1e-8)


@pytest.mark.parametrize("kind", ["original", "rows", "coordinates", "objective"])
def test_lp_affine_slice(kind):
    q = [[0, 0], [0, 0]]
    c, a, b = [-2, 1], [[1, 0], [0, -1], [1, 0]], [3, 2, 3]
    e, h = [[1, 1]], [1]
    expected = stationary_values(q, c, a, b, e, h)
    _, c, a, b, bounds, e, h, scale = transformed(
        q, c, a, b, [(None, None)] * 2, kind, e, h
    )
    result = solve_lp(c, a, b, e, h, bounds=bounds)
    assert result.status == "optimal"
    assert result.objective == pytest.approx(float(scale * expected), abs=1e-8)
    assert np.all(a @ result.x <= b + 1e-8)
    assert e @ result.x == pytest.approx(h, abs=1e-8)


QUADRATICS = [
    ([[4, 1], [1, 2]], [-5, 2], (), ()),
    ([[2, 2], [2, 2]], [-2, -2], (), ()),
    ([[0, 0], [0, 0]], [-2, 1], (), ()),
    ([[4, 1], [1, 2]], [-5, 2], ((1, 1),), (1,)),
    ([[2, 0], [0, 0]], [0, 1], ((1, -1),), (0,)),
]


@pytest.mark.parametrize("q,c,e,h", QUADRATICS)
@pytest.mark.parametrize("kind", ["original", "rows", "coordinates", "objective"])
def test_convex_qp_differential(q, c, e, h, kind):
    rows = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [2, 2]]
    rhs = [2, 2, 3, 1, 2, 4]
    expected = stationary_values(q, c, rows, rhs, e, h)
    q, c, a, b, bounds, e, h, scale = transformed(
        q, c, rows, rhs, [(None, None)] * 2, kind, e, h
    )
    result = solve_convex_qp(
        q, c, a, b, e if len(e) else None, h if len(e) else None, bounds=bounds
    )
    assert result.status == "optimal"
    assert np.all(a @ result.x <= b + 1e-7)
    if len(e):
        assert e @ result.x == pytest.approx(h, abs=1e-7)
    assert result.objective == pytest.approx(float(scale * expected), abs=1e-7)
    assert 0.5 * result.x @ q @ result.x + c @ result.x == pytest.approx(
        result.objective, abs=1e-9
    )


@pytest.mark.parametrize(
    "q,c",
    [
        ([[2, 1], [1, -2]], [-1, 2]),
        ([[0, -1], [-1, 0]], [0, 0]),
        ([[-2, 0], [0, -4]], [1, -1]),
    ],
)
@pytest.mark.parametrize("backend", ["faces", "spatial"])
def test_nonconvex_qp_bounds(q, c, backend):
    x, y = sp.symbols("x y", real=True)
    vector = sp.Matrix([x, y])
    objective = (vector.T * sp.Matrix(q) * vector)[0] / 2 + (sp.Matrix(c).T * vector)[0]
    rows, rhs = [[1, 0], [-1, 0], [0, 1], [0, -1]], [2, 1, 1, 2]
    expected = stationary_values(q, c, rows, rhs)
    constraints = [x <= 2, x >= -1, y <= 1, y >= -2]
    result = minimize_quadratic(
        objective, constraints, (x, y), backend=backend, max_nodes=8, certified=True
    )
    assert verify_quadratic_certificate(
        objective, result.certificate, constraints, (x, y)
    )
    assert result.lower_bound <= expected <= result.upper_bound
    assert objective.subs(dict(zip((x, y), result.point))) == result.upper_bound
    if backend == "faces":
        assert result.objective_value == expected
        assert result.lower_bound == result.upper_bound
