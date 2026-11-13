import sympy as sp

from symbopt import minimize_result, minval


def test_convex_exponential_stationary_point_is_global():
    x = sp.symbols("x", real=True)
    result = minimize_result(sp.exp(x) - 2 * x, variables=[x])
    assert result.certified is True
    assert result.structural_result.kind == "transcendental_domain"
    assert sp.simplify(result.optimum_value - (2 - 2 * sp.log(2))) == 0
    assert result.best_candidate.original_variable_values[x] == sp.log(2)


def test_log_domain_and_convexity_are_used_exactly():
    x = sp.symbols("x", real=True)
    result = minimize_result(x - sp.log(x), [x > 0], variables=[x])
    assert result.certified is True
    assert result.optimum_value == 1
    assert result.attained is True
    assert result.best_candidate.original_variable_values == {x: sp.S.One}
    assert result.lifted_problem.original_constraints == [x > 0]
    assert result.structural_result.metadata["sufficient_conditions"] is True


def test_one_sided_log_limit_certifies_unbounded_below():
    x = sp.symbols("x", real=True)
    result = minimize_result(sp.log(x) + x, [x > 0], variables=[x])
    assert result.certified is True
    assert result.optimum_value is -sp.oo
    assert result.attained is False
    assert minval(sp.log(x) + x, [x > 0], variables=[x]) is -sp.oo


def test_rational_pole_can_certify_negative_infinity():
    x = sp.symbols("x", real=True)
    result = minimize_result(-1 / (x - 1) ** 2, variables=[x])
    assert result.certified is True
    assert result.optimum_value is -sp.oo
    assert result.attained is False
    assert result.structural_result.metadata["singularities"] == (sp.S.One,)


def test_log_escape_needs_real_path():
    from symbopt.univariate_analysis import _logarithmic_escape

    x = sp.Symbol("x", real=True)
    source = sp.Interval.open(0, 1)
    for expression in [sp.log(x) + sp.I * x, sp.log(x) + (-1) ** x]:
        assert sp.limit(expression, x, 0, dir="+") == -sp.oo
        assert _logarithmic_escape(expression, x, source, "univariate_range") is None
    result = _logarithmic_escape(sp.log(x), x, source, "univariate_range")
    assert result.certified and result.value == -sp.oo and result.attained is False
