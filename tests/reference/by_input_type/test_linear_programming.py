import numpy as np
import pytest
import sympy as sp

from symbopt import minimize_result, solve_lp

pytestmark = pytest.mark.reference

x, y, z, t, v = sp.symbols("x y z t v", real=True)


def _candidate(result):
    assert result.best_candidate is not None
    return result.best_candidate


def _assert_value(result, expected):
    candidate = _candidate(result)
    assert sp.simplify(candidate.objective_value - expected) == 0


def _assert_point(result, **expected):
    candidate = _candidate(result)
    for name, value in expected.items():
        symbol = next(
            var for var in candidate.original_variable_values if str(var) == name
        )
        assert sp.simplify(candidate.original_variable_values[symbol] - value) == 0


def _maximize(objective, constraints=None, **kwargs):
    result = minimize_result(-objective, constraints or [], **kwargs)
    candidate = _candidate(result)
    return -candidate.objective_value, candidate.original_variable_values


def test_four_variable_linear_program_one():
    x1, x2, x3, x4 = sp.symbols("x1 x2 x3 x4", real=True)
    result = minimize_result(
        x1 + x2 + x3 + x4,
        [
            sp.Eq(x1 - 5 * x2 + x3 + 3 * x4, 19),
            sp.Le(x1 - 4 * x2 + 2 * x4, 5),
            sp.Le(-5 * x2 + 15 * x4, 10),
            sp.Ge(x1, 0),
            sp.Ge(x2, 0),
            sp.Ge(x3, 0),
            sp.Ge(x4, 0),
        ],
        use_sos=False,
    )
    _assert_value(result, sp.Rational(53, 3))
    point = _candidate(result).original_variable_values
    assert sp.simplify(point[x1] - 5 * point[x2] + point[x3] + 3 * point[x4] - 19) == 0
    assert sp.simplify(point[x1] - 4 * point[x2] + 2 * point[x4] - 5) <= 0
    assert sp.simplify(-5 * point[x2] + 15 * point[x4] - 10) <= 0
    assert all(sp.simplify(point[var]) >= 0 for var in (x1, x2, x3, x4))


def test_four_variable_linear_program_two():
    x1, x2, x3, x4 = sp.symbols("x1 x2 x3 x4", real=True)
    result = minimize_result(
        2 * x1 + x2 + x3 - x4,
        [
            sp.Le(6 * x1 - x2 + x3, 10),
            sp.Le(x1 + 5 * x2, 4),
            sp.Eq(x1 + 5 * x2 + x4, 5),
            sp.Ge(x1, 0),
            sp.Ge(x2, 0),
            sp.Ge(x3, 0),
            sp.Ge(x4, 0),
        ],
        use_sos=False,
    )
    _assert_value(result, -5)
    _assert_point(result, x1=0, x2=0, x3=0, x4=5)


def test_twelve_variable_linear_program():
    c = (
        np.array(
            [46, 44, 44, 42, 40, 40, 42, 44, 44, 46, 48, 48],
            dtype=float,
        )
        / 5.0
    )
    rows = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
        ],
        dtype=float,
    )
    rhs = np.array([3, 3, 8, 8, 10, 10, 8, 8, 14, 14, 5, 5], dtype=float)
    result = solve_lp(
        c,
        A_ub=-rows,
        b_ub=-rhs,
        bounds=[(None, None)] * 12,
    )
    assert result.status == "optimal"
    assert abs(result.objective - 1244 / 5) < 1e-8


def test_reference_linear_program_seven_variables():
    # The same mathematical program is also checked through value and argument
    # wrappers and alternate numerical-precision settings.
    c = np.array([0, 0, 0, 0, 0, 0, 1], dtype=float)
    A_eq = np.array(
        [
            [1, 0, 0, 0, -0.25, -0.5, 0],
            [0, 0, 0, 1, -9, -3, 0],
            [0, 0, 1, 0, 1, 0.5, -1],
            [0, 1, 0, 0, 8, 12, 0],
        ],
        dtype=float,
    )
    b_eq = np.array([-0.75, 6, -0.5, 20], dtype=float)
    result = solve_lp(
        c,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(0, 100)] * 7,
    )
    assert result.status == "optimal"
    assert abs(result.objective - 1.25) < 1e-10
    expected = np.array([0, 2, 0, 10.5, 0, 1.5, 1.25])
    assert np.linalg.norm(result.x - expected, ord=np.inf) < 1e-9
