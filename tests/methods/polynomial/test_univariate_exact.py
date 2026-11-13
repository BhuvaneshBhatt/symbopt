import pytest
import sympy as sp

from symbopt import minimize_result
from symbopt.rational_optimization import try_univariate_rational

x = sp.Symbol("x", real=True)


@pytest.mark.parametrize(
    "objective,constraints,value,points",
    [
        ((x * x - 1) ** 2, [], 0, {-1, 1}),
        (1 / (1 + x * x), [], 0, set()),
        (-(x**3), [], -sp.oo, set()),
        (x**3, [], -sp.oo, set()),
        (x * x, [x > 0], 0, set()),
        (1 / (x - 1), [x > 1], 0, set()),
        (1 / (x - 1), [x < 1], -sp.oo, set()),
        (x * x, [sp.Eq(x * x, 2)], 2, {-sp.sqrt(2), sp.sqrt(2)}),
        (sp.S(3), [x > 0, x < 1], 3, {sp.Rational(1, 2)}),
        (x * x, [x > 0, x <= 0], sp.oo, set()),
        ((x - sp.sqrt(2)) ** 4 + (x - sp.sqrt(2)) ** 2, [], 0, {sp.sqrt(2)}),
        (x * x, [1 / x < 0], 0, set()),
    ],
)
def test_exact_cases(objective, constraints, value, points):
    result = try_univariate_rational(objective, constraints, (x,))
    assert result.certified and result.value == value
    assert result.attained == bool(points)
    assert {p[x] for p in result.points} == points


def test_nonradical_critical_point():
    objective = x**6 + x * x + x
    result = try_univariate_rational(objective, [], (x,))
    assert result.certified and result.attained
    root = result.points[0][x]
    assert root == sp.real_roots(sp.diff(objective, x))[0]
    assert result.value == objective.subs(x, root)


def test_missing_root_support_is_not_a_proof():
    parameter = sp.Symbol("a", real=True)
    assert try_univariate_rational((x - parameter) ** 4, [], (x,)) is None
    assert try_univariate_rational(x * x + sp.pi * x, [], (x,)) is None
    assert try_univariate_rational(x * x + sp.Float("0.1"), [], (x,)) is None


def test_pipeline_minima():
    result = minimize_result((x * x - 1) ** 2, variables=x, use_sos=False)
    assert result.certified and result.attained and result.optimum_value == 0
    assert {c.primal_values[x] for c in result.candidates} == {-1, 1}


def test_tiny_exact_coordinate():
    shift = sp.Rational(1, 10**50)
    result = try_univariate_rational((x - shift) ** 4, [], (x,))
    assert result.value == 0 and result.points == ({x: shift},)


def test_root_object_coefficient():
    z = sp.Symbol("z")
    coefficient = sp.CRootOf(z**3 - z - 1, 0)
    result = try_univariate_rational((x - coefficient) ** 4, [], (x,))
    assert result.certified and result.attained and result.value == 0
    assert result.points == ({x: coefficient},)
