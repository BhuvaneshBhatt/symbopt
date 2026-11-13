import pytest
import sympy as sp

from symbopt import minimize_result

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


def test_quartic_ball_exact_minimum():
    result = minimize_result(
        x**2 - y**3,
        [sp.Le(x**4 + y**4, 3)],
        use_sos=False,
    )
    _assert_value(result, -(3 ** sp.Rational(3, 4)))
    _assert_point(result, x=0, y=3 ** sp.Rational(1, 4))


def test_quartic_ball_corrected_constraint_regression():
    result = minimize_result(
        x**2 - y**3,
        [sp.Le(x**4 + y**4, 3)],
        use_sos=False,
    )
    _assert_value(result, -(3 ** sp.Rational(3, 4)))
    _assert_point(result, x=0, y=3 ** sp.Rational(1, 4))


def test_unconstrained_two_variable_quadratic():
    result = minimize_result(x**2 - 3 * x + y**2 + 3 * y + 3, use_sos=False)
    _assert_value(result, -sp.Rational(3, 2))
    _assert_point(result, x=sp.Rational(3, 2), y=-sp.Rational(3, 2))


def test_unused_variable_mathematical_core():
    # symbopt infers variables; unused z,t are not part of the optimization model.
    result = minimize_result(x**2 - 3 * x + y**2 + 3 * y + 3, use_sos=False)
    _assert_value(result, -sp.Rational(3, 2))
    _assert_point(result, x=sp.Rational(3, 2), y=-sp.Rational(3, 2))


def test_concave_quadratic_maximum_via_negation():
    value, point = _maximize((24 + x) * (600 - 12 * x), use_sos=False)
    assert sp.simplify(value - 16428) == 0
    assert sp.simplify(point[x] - 13) == 0


def test_quartic_double_well_minimum():
    result = minimize_result(x**4 - x**2, use_sos=False)
    _assert_value(result, -sp.Rational(1, 4))
    point = _candidate(result).original_variable_values[x]
    assert sp.simplify(point**2 - sp.Rational(1, 2)) == 0


def test_equality_plus_strict_ball_quadratic():
    result = minimize_result(
        x**2 + 3 * y**2,
        [sp.Eq(x + y, 1), sp.Lt(x**2 + y**2, 3)],
        use_sos=False,
    )
    _assert_value(result, sp.Rational(3, 4))
    _assert_point(result, x=sp.Rational(3, 4), y=sp.Rational(1, 4))


def test_linear_objective_quadratic_constraint():
    result = minimize_result(x, [sp.Le(x**2, 3)], use_sos=False)
    _assert_value(result, -sp.sqrt(3))
    _assert_point(result, x=-sp.sqrt(3))
