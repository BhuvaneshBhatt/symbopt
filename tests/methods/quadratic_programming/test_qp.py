import numpy as np
import pytest

from symbopt import solve_convex_qp


def test_qp_unconstrained_bowl():
    Q = np.array([[2.0, 0.0], [0.0, 2.0]])
    c = np.array([-2.0, -4.0])
    r = solve_convex_qp(Q, c)
    assert r.status == "optimal"
    assert np.linalg.norm(r.x - np.array([1.0, 2.0])) < 1e-4


def test_qp_linear_constraint():
    Q = np.eye(2) * 2
    c = np.array([-2.0, -4.0])
    r = solve_convex_qp(Q, c, A_ub=np.array([[1.0, 1.0]]), b_ub=np.array([2.0]))
    assert r.status == "optimal"
    assert r.x.sum() <= 2.0001


def test_qp_equality_constraint():
    Q = np.eye(2) * 2
    c = np.zeros(2)
    r = solve_convex_qp(Q, c, A_eq=np.array([[1.0, 1.0]]), b_eq=np.array([1.0]))
    assert r.status == "optimal"
    assert abs(r.x.sum() - 1) < 1e-5


def test_qp_free_variable_negative_minimum():
    Q = np.array([[2.0]])
    c = np.array([4.0])
    r = solve_convex_qp(Q, c)
    assert r.status == "optimal"
    assert abs(r.x[0] + 2.0) < 1e-6
    assert abs(r.objective + 4.0) < 1e-6


def test_qp_rejects_nonconvex_matrix():
    r = solve_convex_qp(np.array([[-2.0]]), np.array([0.0]))
    assert r.status == "nonconvex"


def test_qp_rejects_bad_x0():
    with pytest.raises(ValueError):
        solve_convex_qp(
            np.eye(2),
            np.zeros(2),
            x0=np.zeros(3),
        )


@pytest.mark.parametrize(
    "matrix,rhs",
    [
        (np.array([[1.0, 2.0, 3.0]]), np.array([1.0])),
        (np.array([[1.0, 2.0]]), np.array([1.0, 2.0])),
    ],
)
def test_qp_bad_ineq_shapes(matrix, rhs):
    with pytest.raises(ValueError):
        solve_convex_qp(
            np.eye(2),
            np.zeros(2),
            A_ub=matrix,
            b_ub=rhs,
        )


def test_qp_requires_rhs_pair():
    with pytest.raises(ValueError):
        solve_convex_qp(
            np.eye(2),
            np.zeros(2),
            A_eq=np.array([[1.0, 1.0]]),
        )


@pytest.mark.parametrize(
    "Q,c",
    [
        (np.array([[np.nan]]), np.array([0.0])),
        (np.array([[1.0]]), np.array([np.inf])),
    ],
)
def test_qp_rejects_nonfinite(Q, c):
    with pytest.raises(ValueError):
        solve_convex_qp(Q, c)


def test_qp_rejects_bad_tol():
    with pytest.raises(ValueError):
        solve_convex_qp(
            np.eye(1),
            np.zeros(1),
            convexity_tol=-1.0,
        )


def test_solve_qp_accepts_single_constraint_row_and_scalar_rhs():
    r = solve_convex_qp(
        [[2.0, 0.0], [0.0, 2.0]],
        [-2.0, -4.0],
        A_ub=[1.0, 1.0],
        b_ub=4.0,
    )
    assert r.status == "optimal"
    assert r.x.sum() <= 4.0 + 1e-8


def test_solve_qp_accepts_single_variable_bounds_pair():
    r = solve_convex_qp([[2.0]], [-4.0], bounds=(0.0, 1.0))
    assert r.status == "optimal"
    assert abs(r.x[0] - 1.0) < 1e-7
