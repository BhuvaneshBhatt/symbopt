"""Domain and attainment contracts at the function-property boundary."""

import pytest
import sympy as sp
from sympy.logic.boolalg import BooleanFunction

from symbopt.univariate_analysis import _condition_to_set, _range_result, _set_attains
from tests.reference.corpus.checks import decode_case, load_cases

x, y = sp.symbols("x y", real=True)


def test_unknown_attainment():
    assert _set_attains(sp.Interval(0, 1), y) is None
    assert _set_attains(sp.Interval(0, 1), 0) is True
    assert _set_attains(sp.Interval.open(0, 1), 0) is False
    assert _set_attains(sp.S.Reals, -sp.oo) is False


def test_semantic_domain():
    class IntegerCondition(BooleanFunction):
        nargs = 1

    assert _condition_to_set(sp.Not(IntegerCondition(x)), x) is None
    assert _condition_to_set(y > 0, x) is None
    assert _condition_to_set(None, x) is None


def test_endpoint_witness():
    pytest.importorskip("funcprops")
    result = _range_result("range", -sp.asinh(x), [x >= -5, x <= 5], x)
    assert result.value == -sp.asinh(5)
    assert result.attained is True
    assert result.points == ({x: sp.Integer(5)},)


def test_indexed_attainment():
    pytest.importorskip("funcprops")
    result = _range_result("range", -sp.airybi(x), [x >= -5, x <= 5], x)
    # An unresolved indexed envelope cannot disprove compact-domain attainment.
    assert result.attained is not False


def test_corrected_quadratic_oracle():
    objective = sp.Rational(21, 4) * x**2 - sp.Rational(3, 2) * x - sp.Rational(3, 4)
    cases = [
        case
        for case in load_cases()
        if case["operation"] == "maximize_result"
        and case["objective"] == sp.srepr(objective)
        and case["working_precision"] is None
    ]
    assert len(cases) == 1
    case = cases[0]
    f, constraints, (v,) = decode_case(case)
    assert len(constraints) == 1
    assert constraints[0].subs(v, -5) is sp.true
    assert constraints[0].subs(v, 5) is sp.true
    assert constraints[0].subs(v, 6) is sp.false
    assert sp.sympify(case["expected_values"][0]) == f.subs(v, -5) == 138
    assert f.subs(v, 5) == 123


def test_decimal_literals():
    decimal_objective = "Add(Float('1.0'), Pow(Symbol('x', real=True), Integer(4)))"
    cases = [
        case
        for case in load_cases()
        if case["operation"] == "minimize_result"
        and case["objective"] == decimal_objective
        and case["working_precision"] in (None, 20)
    ]
    assert len(cases) == 2
    assert {case["working_precision"] for case in cases} == {None, 20}
    for case in cases:
        f, constraints, (v,) = decode_case(case)
        assert len(constraints) == 1
        assert constraints[0].subs(v, 3) is sp.true
        assert constraints[0].subs(v, 4) is sp.true
        assert constraints[0].subs(v, 2) is sp.false
        assert f.free_symbols == {v}
        assert f.subs(v, 3) == sp.Float(82)


def test_analysis_before_algebraization():
    pytest.importorskip("funcprops")
    from symbopt import minimize_result

    result = minimize_result(sp.asinh(x), [x >= -1, x <= 1], variables=x)
    assert sp.simplify(result.optimum_value + sp.log(1 + sp.sqrt(2))) == 0
    assert result.attained is True
    assert result.best_candidate.original_variable_values == {x: -1}


def test_complex_intermediate_domain():
    pytest.importorskip("funcprops")
    from symbopt import minimize_result
    from symbopt.univariate_analysis import analyze_univariate

    objective = sp.cos(sp.I * sp.log(x), evaluate=False)
    constraints = (x >= -2, x <= -sp.Rational(1, 2))
    analysis = analyze_univariate(objective, constraints, x)
    assert analysis.feasible_domain == sp.Interval(-2, -sp.Rational(1, 2))
    result = minimize_result(objective, constraints, variables=x)
    assert result.certified and result.optimum_value == -sp.Rational(5, 4)
    assert result.attained is True
    assert (
        sp.simplify(
            objective.subs(result.best_candidate.original_variable_values)
            - result.optimum_value
        )
        == 0
    )
