import numpy as np
import pytest

from symbopt import revised_simplex, simplex, solve_lp


def test_primal_simplex_optimal():
    r = simplex(np.array([1.0, 1.0]), A_ub=np.array([[1.0, 1.0]]), b_ub=np.array([1.0]))
    assert r.status == "optimal"
    assert abs(r.objective) < 1e-6


def test_revised_simplex_optimal():
    r = revised_simplex(
        np.array([-2.0, -1.0]),
        A_ub=np.array([[1.0, 1.0], [1.0, 0.0]]),
        b_ub=np.array([3.0, 2.0]),
    )
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-5)
    assert r.x == pytest.approx([2, 1])


@pytest.mark.parametrize("solver", [simplex, revised_simplex])
def test_slack_reentry(solver):
    # The first binding row becomes inactive at the optimum. Its slack must
    # re-enter the basis after the other original variable has entered.
    result = solver([-4, 0], [[2, -4], [3, -1], [0, 1]], [2, 5, 6])
    assert result.status == "optimal"
    assert result.x == pytest.approx([11 / 3, 6])
    assert result.objective == pytest.approx(-44 / 3)


@pytest.mark.parametrize("solver", [simplex, revised_simplex])
def test_unbounded_value(solver):
    result = solver([-1], [[-1]], [0])
    assert result.status == "unbounded"
    assert result.objective == -np.inf


def test_solve_lp_maximize():
    r = solve_lp(
        np.array([1.0, 1.0]),
        A_ub=np.array([[1.0, 1.0]]),
        b_ub=np.array([2.0]),
        maximize=True,
    )
    assert r.status == "optimal"
    assert abs(r.objective - 2) < 1e-6


def test_lp_infeasible_or_failure_status():
    r = solve_lp(
        np.array([1.0]), A_ub=np.array([[1.0], [-1.0]]), b_ub=np.array([0.0, -1.0])
    )
    assert r.status != "optimal"


def test_solve_lp_equality_constraint():
    r = solve_lp(
        np.array([1.0]),
        A_eq=np.array([[1.0]]),
        b_eq=np.array([2.0]),
        bounds=[(None, None)],
    )
    assert r.status == "optimal"
    assert abs(r.x[0] - 2.0) < 1e-8


def test_solve_lp_free_variable():
    r = solve_lp(
        np.array([1.0]),
        A_ub=np.array([[-1.0]]),
        b_ub=np.array([-2.0]),
        bounds=[(None, None)],
    )
    assert r.status == "optimal"
    assert abs(r.x[0] - 2.0) < 1e-8


def test_simplex_rejects_infeasible_slack_basis():
    r = simplex(
        np.array([1.0]),
        A_ub=np.array([[-1.0]]),
        b_ub=np.array([-2.0]),
    )
    assert r.status == "needs_phase_one"


def test_revised_simplex_minimizes_positive_cost():
    r = revised_simplex(
        np.array([1.0, 1.0]),
        A_ub=np.array([[1.0, 1.0]]),
        b_ub=np.array([1.0]),
    )
    assert r.status == "optimal"
    assert abs(r.objective) < 1e-8


@pytest.mark.parametrize(
    "matrix,rhs",
    [
        (np.array([[1.0, 2.0]]), np.array([1.0])),
        (np.array([[1.0]]), np.array([1.0, 2.0])),
    ],
)
def test_lp_bad_ineq_shapes(matrix, rhs):
    with pytest.raises(ValueError):
        solve_lp(np.array([1.0]), A_ub=matrix, b_ub=rhs)


def test_lp_requires_rhs_pair():
    with pytest.raises(ValueError):
        solve_lp(np.array([1.0]), A_eq=np.array([[1.0]]))


@pytest.mark.parametrize(
    "c,A_ub,b_ub",
    [
        (np.array([np.nan]), None, None),
        (np.array([1.0]), np.array([[np.inf]]), np.array([1.0])),
        (np.array([1.0]), np.array([[1.0]]), np.array([np.nan])),
    ],
)
def test_lp_rejects_nonfinite(c, A_ub, b_ub):
    with pytest.raises(ValueError):
        solve_lp(c, A_ub=A_ub, b_ub=b_ub)


def test_solve_lp_accepts_single_constraint_row_and_scalar_rhs():
    r = solve_lp([1.0, 1.0], A_ub=[-1.0, -1.0], b_ub=-1.0)
    assert r.status == "optimal"
    assert abs(r.objective - 1.0) < 1e-9


def test_solve_lp_accepts_single_variable_bounds_pair():
    r = solve_lp([1.0], bounds=(2.0, 3.0))
    assert r.status == "optimal"
    assert abs(r.x[0] - 2.0) < 1e-9
