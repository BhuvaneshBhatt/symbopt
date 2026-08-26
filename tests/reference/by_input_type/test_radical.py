import pytest
import sympy as sp

from symbopt import minimize

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
    result = minimize(-objective, constraints or [], **kwargs)
    candidate = _candidate(result)
    return -candidate.objective_value, candidate.original_variable_values


def test_sqrt_plus_linear_minimum():
    result = minimize(
        sp.Rational(1, 3) * sp.sqrt(4 + x**2) + sp.Rational(1, 5) * (6 - x),
        use_sos=False,
    )
    _assert_value(result, sp.Rational(26, 15))
    _assert_point(result, x=sp.Rational(3, 2))


def test_sqrt_quadratic_norm_minimum():
    result = minimize(
        sp.sqrt(x**2 + (1 - 2 * x) ** 2),
        use_sos=False,
    )
    _assert_value(result, 1 / sp.sqrt(5))
    _assert_point(result, x=sp.Rational(2, 5))


def test_sqrt_global_minimum():
    result = minimize(sp.sqrt(x), use_sos=False)
    _assert_value(result, 0)
    _assert_point(result, x=0)
