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


def test_absolute_value_disk_minimum():
    result = minimize_result(
        x**2 + 3 * y**2 + sp.Abs(x),
        [sp.Le(x**2 + y**2, 1)],
        use_sos=False,
    )
    _assert_value(result, 0)
    _assert_point(result, x=0, y=0)
