import sympy as sp

from symbopt import minimize, minval


def test_convex_exponential_stationary_point_is_global():
    x = sp.symbols("x", real=True)
    result = minimize(sp.exp(x) - 2 * x, variables=[x])
    assert result.certified is True
    assert result.structural_result.kind == "transcendental_domain"
    assert sp.simplify(result.optimum_value - (2 - 2 * sp.log(2))) == 0
    assert result.best_candidate.original_variable_values[x] == sp.log(2)


def test_log_domain_and_convexity_are_used_exactly():
    x = sp.symbols("x", real=True)
    result = minimize(x - sp.log(x), [x > 0], variables=[x])
    assert result.certified is True
    assert result.optimum_value == 1
    assert result.attained is True
    assert result.structural_result.metadata["singularities"] == "{0}"


def test_one_sided_log_limit_certifies_unbounded_below():
    x = sp.symbols("x", real=True)
    result = minimize(sp.log(x) + x, [x > 0], variables=[x])
    assert result.certified is True
    assert result.optimum_value is -sp.oo
    assert result.attained is False
    assert minval(sp.log(x) + x, [x > 0], variables=[x]) is -sp.oo


def test_rational_pole_can_certify_negative_infinity():
    x = sp.symbols("x", real=True)
    result = minimize(-1 / (x - 1) ** 2, variables=[x])
    assert result.certified is True
    assert result.optimum_value is -sp.oo
    assert result.attained is False
    assert "1" in result.structural_result.metadata["singularities"]
