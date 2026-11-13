import sympy as sp

from symbopt import maximize_result, minimize_result


def test_tangent_poles_certify_unbounded_below_and_above():
    x = sp.symbols("x", real=True)
    minimum = minimize_result(sp.tan(x), variables=x)
    maximum = maximize_result(sp.tan(x), variables=x)
    assert minimum.certified is True
    assert minimum.optimum_value == -sp.oo
    assert minimum.attained is False
    assert minimum.method == "rational_frequency_trig"
    assert maximum.certified is True
    assert maximum.optimum_value == sp.oo
    assert maximum.attained is False
    assert minimum.structural_result.metadata["real_denominator_roots"]
    assert minimum.structural_result.metadata["pole_limits"]


def test_even_order_poles_do_not_hide_finite_minimum():
    x = sp.symbols("x", real=True)
    result = minimize_result(sp.sec(x) ** 2, variables=x)
    assert result.certified is True
    assert result.optimum_value == 1
    assert result.attained is True
    assert result.method == "rational_frequency_trig"


def test_negative_even_order_poles_certify_negative_infinity():
    x = sp.symbols("x", real=True)
    result = minimize_result(-(sp.csc(x) ** 2), variables=x)
    assert result.certified is True
    assert result.optimum_value == -sp.oo
    assert result.attained is False


def test_algebraic_phase_shift_is_supported_exactly():
    x = sp.symbols("x", real=True)
    result = minimize_result(sp.cos(x + sp.pi / 4), variables=x)
    assert result.certified is True
    assert result.optimum_value == -1
    assert result.attained is True
    assert result.method == "rational_frequency_trig"
    assert "pi/4" in result.structural_result.metadata["phases"]


def test_algebraic_trig_phase():
    x = sp.symbols("x", real=True)
    objective = 1 / (2 + sp.cos(x + sp.pi / 4))
    result = minimize_result(objective, variables=x)
    assert result.certified is True
    assert result.optimum_value == sp.Rational(1, 3)
    assert result.attained is True
    assert result.method == "rational_frequency_trig"


def test_bounded_tangent_minimum():
    x = sp.symbols("x", real=True)
    result = minimize_result(
        sp.tan(x),
        constraints=[x >= 0, x <= sp.pi / 4],
        variables=x,
    )
    assert result.certified is True
    assert result.optimum_value == 0
    assert result.attained is True


def test_open_boundary_pole_uses_one_sided_limit():
    x = sp.symbols("x", real=True)
    result = minimize_result(
        sp.tan(x),
        constraints=[x > -sp.pi / 2, x < sp.pi / 2],
        variables=x,
    )
    assert result.certified is True
    assert result.optimum_value == -sp.oo
    assert result.attained is False


def test_half_angle_infinity_phase_pole_is_classified_exactly():
    x = sp.symbols("x", real=True)
    result = minimize_result(1 / (1 + sp.cos(x)), variables=x)
    assert result.certified is True
    assert result.optimum_value == sp.Rational(1, 2)
    assert result.attained is True
    assert result.structural_result.metadata["infinity_limits"] == ("oo", "oo")


def test_half_angle_pole_direction():
    x = sp.symbols("x", real=True)
    result = minimize_result(-1 / (1 + sp.cos(x)), variables=x)
    assert result.certified is True
    assert result.optimum_value == -sp.oo
    assert result.attained is False


def test_cotangent_reduction():
    x = sp.symbols("x", real=True)
    result = minimize_result(sp.cot(x), variables=x)
    assert result.certified is True
    assert result.optimum_value == -sp.oo
    assert result.attained is False
    assert result.method == "rational_frequency_trig"
