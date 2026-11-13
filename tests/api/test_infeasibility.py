import pytest
import sympy as sp

from symbopt import maximize_result, minimize_result


@pytest.mark.parametrize(
    "optimize, value", [(minimize_result, sp.oo), (maximize_result, -sp.oo)]
)
def test_impossible_real_equality(optimize, value):
    x, y = sp.symbols("x y", real=True)
    result = optimize(
        x - y**3, sp.Eq(x * x + y * y, -1, evaluate=False), variables=(x, y)
    )
    assert result.certified and result.infeasible and not result.unbounded
    assert result.optimum_value == value
    assert result.attained is False
    assert result.best_candidate is None


def test_real_equality_can_be_feasible():
    x, y = sp.symbols("x y", real=True)
    result = minimize_result(x * x + y * y, sp.Eq(x * x + y * y, 1), variables=(x, y))
    assert result.certified and result.attained is True
    assert result.optimum_value == 1
    point = result.best_candidate.primal_values
    assert sp.simplify(point[x] ** 2 + point[y] ** 2) == 1
