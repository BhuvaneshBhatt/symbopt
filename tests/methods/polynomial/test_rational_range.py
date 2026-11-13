import pytest
import sympy as sp

from symbopt import minimize_result, rational_graph
from symbopt.rational_optimization import try_rational_range

x, y, t = sp.symbols("x y t", real=True)


def test_denominator_signs():
    graph = rational_graph(1 / x, [1 / y >= 1], (x, y), t)
    assert graph.subs({x: -2, y: sp.Rational(1, 2), t: sp.Rational(-1, 2)}) is sp.true
    assert graph.subs({x: 0, y: sp.Rational(1, 2), t: 0}) is sp.false
    assert graph.subs({x: 1, y: -1, t: 1}) is sp.false


def test_cancellation_hole():
    objective = sp.Mul(x * x - 1, sp.Pow(x - 1, -1, evaluate=False), evaluate=False)
    graph = rational_graph(objective, [], x, t)
    assert graph.subs({x: 1, t: 2}) is sp.false
    assert graph.subs({x: 2, t: 3}) is sp.true
    result = minimize_result(objective, [x >= 1], variables=x, use_sos=False)
    assert result.certified and result.optimum_value == 2 and result.attained is False


def test_graph_symbol_capture():
    with pytest.raises(ValueError, match="value symbol"):
        rational_graph(t / x, [], x, t)


@pytest.mark.parametrize(
    "objective,value,attained",
    [
        (1 / (1 + x * x + y * y), 0, False),
        ((x * x + y * y) / (1 + x * x + y * y), 0, True),
        (1 / (x * y), -sp.oo, False),
    ],
)
def test_exact_graph_range(objective, value, attained):
    pytest.importorskip("semialg")
    result = try_rational_range(objective, [], (x, y), cost_limit=None)
    assert result.certified and result.value == value and result.attained is attained
    image = result.metadata["range_formula"]
    symbol = result.metadata["value_symbol"]
    if value != -sp.oo:
        assert sp.simplify(image.subs(symbol, value)) is (
            sp.true if attained else sp.false
        )
    else:
        assert sp.simplify(image.subs(symbol, -100)) is sp.true


def test_range_budget():
    assert try_rational_range(1 / (1 + x * x + y * y), [], (x, y), cost_limit=0) is None


def test_disk_optimizer():
    pytest.importorskip("semialg")
    objective = (3 * x + y * y) / (1 + x * x)
    result = minimize_result(
        objective, x * x + y * y <= 3, variables=(x, y), use_sos=False
    )
    assert (
        result.certified
        and result.attained
        and result.optimum_value == -sp.Rational(3, 2)
    )
    assert result.best_candidate.primal_values == {x: -1, y: 0}


def test_empty_graph_image():
    pytest.importorskip("semialg")
    result = try_rational_range(
        1 / (1 + x * x + y * y),
        [sp.Lt(x * x, -1, evaluate=False)],
        (x, y),
        cost_limit=None,
    )
    assert result.certified and result.value == sp.oo and result.attained is False
    assert result.points == ()


def test_parameter_exclusion_is_not_discarded():
    from symbopt.linear_fractional import try_linear_fractional

    parameter = sp.Symbol("a", real=True)
    assert try_linear_fractional(1 / x, [sp.Ne(parameter * x, 0)], (x,)) is None


def test_zero_coefficient_exclusion_is_not_discarded():
    from symbopt.linear_fractional import _implied_denominator_exclusion

    zero = sp.sin(1) ** 2 + sp.cos(1) ** 2 - 1
    condition = sp.Ne(zero * x, 0, evaluate=False)
    assert _implied_denominator_exclusion(condition, x, (x,)) is False


@pytest.mark.parametrize("excluded", [sp.Symbol("a", real=True), sp.Float("0.5")])
def test_domain_survives_unsupported_coefficients(excluded):
    from symbopt.rational_optimization import preserve_rational_domain

    objective = sp.Mul(
        x - excluded, sp.Pow(x - excluded, -1, evaluate=False), evaluate=False
    )
    conditions = preserve_rational_domain(objective, [], (x,))
    assert sp.And(*conditions).subs(x, excluded) is sp.false


def test_empty_parameterized_hole():
    parameter = sp.Symbol("a", real=True)
    objective = sp.Mul(
        x - parameter, sp.Pow(x - parameter, -1, evaluate=False), evaluate=False
    )
    result = minimize_result(objective, sp.Eq(x, parameter), variables=x, use_sos=False)
    assert result.infeasible or not result.certified
