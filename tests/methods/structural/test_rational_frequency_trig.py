import sympy as sp

from symbopt import minimize_result


def test_mixed_integer_frequency_trig_is_optimized_exactly():
    x = sp.symbols("x", real=True)
    objective = sp.sin(x) + sp.cos(3 * x)
    result = minimize_result(objective, variables=[x])
    assert result.certified is True
    assert result.attained is True
    assert result.structural_result.kind == "rational_frequency_trig"
    assert result.structural_result.metadata["transformation"] == "tangent-half-angle"
    assert "CRootOf" in sp.sstr(result.optimum_value)
    assert abs(float(sp.N(result.optimum_value, 30)) + 1.87870685012) < 1e-10


def test_rational_common_frequency_trig_polynomial():
    x = sp.symbols("x", real=True)
    objective = sp.sin(x / 2) + sp.cos(3 * x / 2)
    result = minimize_result(objective, variables=[x])
    assert result.certified is True
    assert result.structural_result.kind == "rational_frequency_trig"
    assert result.structural_result.metadata["common_frequency"] == "1/2"


def test_rational_frequency_trig_value_and_argument_wrappers():
    from symbopt import argmax, maxval, minval

    x = sp.symbols("x", real=True)
    objective = sp.sin(x) + sp.cos(2 * x)
    assert minval(objective, variables=[x]) == -2
    assert maxval(objective, variables=[x]) == sp.Rational(9, 8)
    points = argmax(objective, variables=[x])
    assert len(points) == 2
    assert all(
        sp.simplify(objective.subs(point) - sp.Rational(9, 8)) == 0 for point in points
    )
