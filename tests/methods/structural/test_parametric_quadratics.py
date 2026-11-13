import pytest
import sympy as sp

from symbopt import argmin, maximize_result, minimize, minimize_result
from symbopt.quadratic_optimization import try_parametric_quadratic

x, y, a, b, c, lower = sp.symbols("x y a b c lower", real=True)
p = sp.Symbol("p", positive=True)
n = sp.Symbol("n", nonnegative=True)


def test_translated_quadratic():
    assert minimize((x - a) ** 2, variables=x) == [0, {x: a}]


@pytest.mark.parametrize("weight", [p, n])
def test_symbolic_value(weight):
    result = minimize_result(weight * (x - a) ** 2 + c, variables=x)
    assert result.certified and result.attained is True
    assert result.optimum_value == c
    assert result.best_candidate.primal_values == {x: a}
    assert result.best_candidate.objective_value_numeric is None
    assert result.certificate.lower_bound == result.certificate.upper_bound == c


def test_coupled_quadratic():
    result = minimize_result((x - a) ** 2 + p * (y - x - b) ** 2 + c, variables=(x, y))
    assert result.optimum_value == c and result.attained is True
    point = result.best_candidate.primal_values
    assert point == {x: a, y: a + b}
    assert (
        sp.simplify(
            result.lifted_problem.original_objective.subs(point, simultaneous=True) - c
        )
        == 0
    )


def test_singular_quadratic():
    result = minimize_result(n * (x + y - a) ** 2 + c, variables=(x, y))
    assert result.certified and result.optimum_value == c
    assert (
        sp.simplify(
            result.best_candidate.primal_values[x]
            + result.best_candidate.primal_values[y]
            - a
        )
        == 0
    )


def test_maximum():
    result = maximize_result(-p * (x - a) ** 2 + c, variables=x)
    assert result.certified and result.optimum_value == c
    assert result.best_candidate.primal_values == {x: a}
    assert result.best_candidate.objective_value_numeric is None


def test_affine_parameter_bound():
    result = minimize_result(p * (x - a) ** 2 + c, x >= lower, variables=x)
    point = sp.Max(a, lower)
    assert result.certified and result.attained is True
    assert sp.simplify(result.optimum_value - (p * (point - a) ** 2 + c)) == 0
    assert argmin(p * (x - a) ** 2 + c, x >= lower, variables=x) == ({x: point},)


@pytest.mark.parametrize("shift,attained,value", [(1, True, c), (-1, False, p + c)])
def test_strict_bound(shift, attained, value):
    result = minimize_result(p * (x - lower - shift) ** 2 + c, x > lower, variables=x)
    assert result.certified and result.attained is attained
    assert result.optimum_value == value


def test_conditional_attainment():
    result = minimize_result(p * (x - a) ** 2 + c, x > lower, variables=x)
    assert result.certified and result.attained is None
    assert result.best_candidate is None
    assert (
        sp.simplify(result.optimum_value - (p * (sp.Max(a, lower) - a) ** 2 + c)) == 0
    )


@pytest.mark.parametrize(
    "curvature,slope,expected",
    [(1, -1, sp.Rational(-1, 4)), (0, -1, -1), (-1, 0, -1), (2, 1, 0)],
)
def test_conditional_curvature(curvature, slope, expected):
    result = minimize_result(a * x * x + b * x + c, [x >= 0, x <= 1], variables=x)
    assert result.certified and result.attained is True
    specialization = {a: curvature, b: slope, c: 0}
    assert sp.simplify(result.optimum_value.subs(specialization)) == expected
    point = result.best_candidate.primal_values[x].subs(specialization)
    assert point >= 0 and point <= 1
    assert sp.simplify(curvature * point**2 + slope * point - expected) == 0


def test_singleton_and_empty():
    singleton = minimize_result(a * x * x + b * x + c, sp.Eq(x, lower), variables=x)
    assert singleton.certified and singleton.attained is True
    assert singleton.optimum_value == a * lower**2 + b * lower + c
    empty = minimize_result((x - a) ** 2, [x >= lower, x <= lower - 1], variables=x)
    assert empty.certified and empty.infeasible and not empty.unbounded


def test_unbounded():
    result = minimize_result(-p * x * x + c, variables=x)
    assert result.certified and result.unbounded and not result.infeasible


def test_undecidable_parameters():
    assert try_parametric_quadratic(a * x * x + b * x + c, [], (x,)) is None
    assert try_parametric_quadratic((x - a) ** 2, [x >= a, x <= b], (x,)) is None
    z = sp.Symbol("z")
    assert try_parametric_quadratic((x - z) ** 2, [], (x,)) is None


@pytest.mark.parametrize(
    "curvature,slope,expected", [(-1, 1, sp.Rational(1, 4)), (0, 1, 1), (1, 0, 1)]
)
def test_conditional_maximum(curvature, slope, expected):
    result = maximize_result(a * x * x + b * x + c, [x >= 0, x <= 1], variables=x)
    assert result.certified and result.attained is True
    specialization = {a: curvature, b: slope, c: 0}
    assert sp.simplify(result.optimum_value.subs(specialization)) == expected
    point = result.best_candidate.primal_values[x].subs(specialization)
    assert point >= 0 and point <= 1
    assert sp.simplify(curvature * point**2 + slope * point - expected) == 0


def test_parameterized_monotone_bound():
    objective = sp.exp(p * (x - a) ** 2 + c)
    result = minimize_result(objective, x >= lower, variables=x)
    assert result.certified and result.attained is True
    assert result.best_candidate.primal_values == {x: sp.Max(a, lower)}
    assert argmin(objective, x >= lower, variables=x) == ({x: sp.Max(a, lower)},)


def test_constant_parameter():
    assert minimize(c, variables=x) == [c, {}]


def test_unknown_curvature_is_unresolved():
    result = minimize_result(a * x * x, variables=x, use_sos=False)
    assert result.certified is False and result.optimum_value is None


def test_flat_direction_and_indefinite():
    flat = minimize_result((x - a) ** 2 + p * y + c, variables=(x, y))
    assert flat.certified and flat.unbounded
    indefinite = minimize_result((x - a) * (y - b) + c, variables=(x, y))
    assert indefinite.certified and indefinite.unbounded


def test_collapsing_open_interval():
    assert try_parametric_quadratic((x - a) ** 2, [x > 0, x <= n], (x,)) is None
