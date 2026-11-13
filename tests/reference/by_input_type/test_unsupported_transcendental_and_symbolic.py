"""Transcendental and parameter reference cases with explicit supported outcomes."""

import math

import pytest
import sympy as sp

from symbopt import minimize_result


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
    candidate = result.best_candidate
    return (
        -result.optimum_value,
        candidate.original_variable_values if candidate else {},
    )


x, y, z, t, v = sp.symbols("x y z t v", real=True)

pytestmark = [pytest.mark.reference, pytest.mark.timeout(10)]


@pytest.mark.unsupported
@pytest.mark.xfail(
    strict=True,
    reason="general exact exponential stationary solving is not yet implemented",
)
def test_exponential_base_two_productlog_minimum():
    result = minimize_result(x**2 + 2**x, use_sos=False)
    w = sp.LambertW(sp.log(2) ** 2 / 2)
    expected_x = -w / sp.log(2)
    expected = sp.exp(-w) + w**2 / sp.log(2) ** 2
    _assert_value(result, expected)
    _assert_point(result, x=expected_x)


@pytest.mark.unsupported
@pytest.mark.xfail(
    strict=True,
    reason="global trig optimization over periodic branch constraints remains incomplete",
)
def test_tangent_squared_with_sine_constraint():
    result = minimize_result(
        sp.tan(2 * x - sp.pi / 2) ** 2,
        [
            sp.Ge(sp.sin(x), -sp.Rational(1, 2)),
            sp.Le(sp.sin(x), sp.Rational(1, 2)),
        ],
        use_sos=False,
    )
    _assert_value(result, sp.Rational(1, 3))
    _assert_point(result, x=sp.pi / 6)


@pytest.mark.unsupported
@pytest.mark.xfail(
    strict=True,
    reason="mixed transcendental equality optimization is not yet solved exactly",
)
def test_trigonometric_equality_two_variable_minimum():
    result = minimize_result(
        y + sp.sin(10 * x),
        [
            sp.Eq(y**3, sp.cos(5 * x)),
            sp.Ge(x, -5),
            sp.Le(x, 5),
            sp.Ge(y, -5),
            sp.Le(y, 5),
        ],
        use_sos=False,
    )
    candidate = _candidate(result)
    assert abs(candidate.objective_value_numeric + 1.90075) < 2e-5
    assert abs(float(candidate.original_variable_values[x]) + 0.772093) < 2e-5
    assert abs(float(candidate.original_variable_values[y]) + 0.909588) < 2e-5


@pytest.mark.parametrize(
    "variant",
    [
        "symbolic-working-precision",
        "working-precision-20",
        "working-precision-0-a",
        "working-precision-0-b",
        "working-precision-1",
        "working-precision-30",
        "inexact-constraint-precision",
        "inexact-result-realness",
        "inexact-objective",
    ],
)
def test_precision_controls(variant):
    if variant == "symbolic-working-precision":
        with pytest.raises(TypeError):
            minimize_result((x - 2) ** 4, variables=x, working_precision=sp.Symbol("p"))
        return
    if variant.startswith("working-precision-0"):
        with pytest.raises(ValueError):
            objective = (x - 2) ** 2 if variant.endswith("a") else sp.exp(x) + x * x
            minimize_result(objective, variables=x, working_precision=0)
        return
    precision = (
        20
        if variant == "working-precision-20"
        else 1
        if variant == "working-precision-1"
        else 30
    )
    inexact = variant.startswith("inexact-")
    center = (
        sp.Float("2.0", 40) if variant == "inexact-result-realness" else sp.Integer(2)
    )
    objective = (x - center) ** 2
    constraints = []
    if variant == "inexact-constraint-precision":
        constraints = [x >= sp.Float("1.0", 40)]
    elif variant == "inexact-objective":
        objective *= sp.Float("2.0", 40)
    result = minimize_result(
        objective, constraints, variables=x, working_precision=precision, use_sos=False
    )
    assert result.working_precision == precision
    assert result.best_candidate.objective_value.is_zero is True
    point = result.best_candidate.original_variable_values[x]
    assert point.is_real is True and sp.simplify(point - center) == 0
    assert result.precision_provenance["float64_backends_allowed"] is (not inexact)


@pytest.mark.unsupported
@pytest.mark.xfail(
    strict=True,
    reason="general parametric optimization is outside the current supported set",
)
def test_parametric_quartic_ball_minimum():
    a = sp.symbols("a", real=True)
    result = minimize_result(
        a * x**2 - y**3,
        [sp.Le(x**4 + y**4, 3)],
        variables=(x, y),
        use_sos=False,
    )
    assert (
        result.certified
        and sp.simplify(result.optimum_value.subs(a, 0) + 3 ** sp.Rational(3, 4)) == 0
    )
    assert sp.simplify(result.optimum_value.subs(a, 1) + 3 ** sp.Rational(3, 4)) == 0


@pytest.mark.unsupported
@pytest.mark.xfail(
    strict=True,
    reason="exact LambertW recovery for this unconstrained exponential problem is incomplete",
)
def test_exponential_quadratic_lambertw_minimum():
    result = minimize_result(sp.exp(x) + 5 * (x - 4) ** 2, use_sos=False)
    expected_x = 4 - sp.LambertW(sp.E**4 / 10)
    _assert_point(result, x=expected_x)


@pytest.mark.unsupported
@pytest.mark.xfail(
    strict=True,
    reason="boxed transcendental algebraization currently needs stronger auxiliary handling",
)
def test_bounded_exponential_quadratic_lambertw_minimum():
    result = minimize_result(
        sp.exp(x) + 5 * (x - 4) ** 2,
        [sp.Ge(x, 1), sp.Le(x, 3)],
        use_sos=False,
    )
    expected_x = 4 - sp.LambertW(sp.E**4 / 10)
    _assert_point(result, x=expected_x)


@pytest.mark.unsupported
@pytest.mark.xfail(
    strict=True,
    reason="high-frequency exact trigonometric global root isolation remains unsupported",
)
def test_high_frequency_trigonometric_minimum():
    expr = -sp.Rational(4070375628, 10**11) * sp.cos(
        sp.Rational(5118786816, 10**9) * t
    ) + sp.Rational(4070375628, 10**11) * sp.cos(sp.Rational(1668728700, 10**8) * t)
    result = minimize_result(expr, use_sos=False)
    assert _candidate(result).objective_value_numeric < -0.077


@pytest.mark.unsupported
@pytest.mark.xfail(
    strict=True,
    reason="high-frequency exact trigonometric maximization remains unsupported",
)
def test_high_frequency_trigonometric_maximum():
    expr = -sp.Rational(4070375628, 10**11) * sp.cos(
        sp.Rational(5118786816, 10**9) * t
    ) + sp.Rational(4070375628, 10**11) * sp.cos(sp.Rational(1668728700, 10**8) * t)
    value, _ = _maximize(expr, use_sos=False)
    assert float(value) > 0.0805


@pytest.mark.unsupported
@pytest.mark.xfail(
    strict=True,
    reason="floating rational-objective precision behavior is not a symbopt API contract",
)
def test_float_rational_objective_precision_behavior():
    result = minimize_result(5280 * v / (12 + sp.Float("0.025") * v**2), use_sos=False)
    assert abs(_candidate(result).objective_value_numeric + 880 * math.sqrt(30)) < 1e-8


@pytest.mark.unsupported
@pytest.mark.xfail(
    strict=True,
    reason="bounded exponential-over-rational exact LambertW recovery remains incomplete",
)
def test_exponential_over_x_exact_minimum():
    result = minimize_result(
        sp.exp(x) + sp.Rational(39, 5) / x,
        [sp.Ge(x, 1), sp.Le(x, 2)],
        use_sos=False,
    )
    expected_x = 2 * sp.LambertW(sp.sqrt(sp.Rational(39, 5)) / 2)
    _assert_point(result, x=expected_x)


@pytest.mark.unsupported
@pytest.mark.xfail(
    strict=True,
    reason="agreement between exact and floating exponential-over-x formulations remains incomplete",
)
def test_exponential_over_x_exact_float_agreement():
    exact = minimize_result(
        sp.exp(x) + sp.Rational(39, 5) / x,
        [sp.Ge(x, 1), sp.Le(x, 2)],
        use_sos=False,
    )
    numeric = minimize_result(
        sp.exp(x) + sp.Float("7.8") / x,
        [sp.Ge(x, 1), sp.Le(x, 2)],
        use_sos=False,
    )
    assert (
        abs(
            _candidate(exact).objective_value_numeric
            - _candidate(numeric).objective_value_numeric
        )
        < 1e-10
    )


def test_quadratic_plus_cosine_closed_interval():
    result = minimize_result(
        x**2 + sp.cos(x),
        [sp.Ge(x, 0), sp.Le(x, 3)],
        use_sos=False,
    )
    _assert_value(result, 1)
    _assert_point(result, x=0)


def test_negative_log_unbounded_maximum():
    value, _ = _maximize(-sp.log(x), use_sos=False)
    assert value is sp.oo


def test_sqrt_unbounded_maximum():
    value, _ = _maximize(sp.sqrt(x), use_sos=False)
    assert value is sp.oo


@pytest.mark.unsupported
@pytest.mark.xfail(
    strict=True,
    reason="high-degree algebraic equality system with precision-controlled Root output remains unsupported",
)
def test_high_degree_three_variable_equalities():
    result = minimize_result(
        x + y + z,
        [
            sp.Eq(x**4 + y**4 + z**4, 1),
            sp.Eq(x**2 - 2 * y**2, sp.Rational(1, 2)),
        ],
        use_sos=False,
    )
    assert abs(_candidate(result).objective_value_numeric + 2.04718) < 1e-4


def test_exponential_quartic_difference_box():
    result = minimize_result(
        sp.exp(x**4 - y**4),
        [sp.Le(sp.Abs(x), 1), sp.Le(sp.Abs(y), 1)],
        use_sos=False,
    )
    _assert_value(result, sp.E**-1)


@pytest.mark.unsupported
@pytest.mark.xfail(
    strict=True,
    reason="mixed exponential-polynomial disk optimization remains unsupported exactly",
)
def test_linear_times_gaussian_disk():
    result = minimize_result(
        (x + 2 * y) * sp.exp(-(x**2) - y**2),
        [sp.Le(x**2 + y**2, 1)],
        use_sos=False,
    )
    expected = (-2 * sp.sqrt(sp.Rational(2, 5)) - 1 / sp.sqrt(10)) / sp.sqrt(sp.E)
    _assert_value(result, expected)


@pytest.mark.unsupported
@pytest.mark.xfail(
    strict=True,
    reason="general parametric optimization is outside the current supported set",
)
def test_parametric_elliptic_gaussian():
    a, b = sp.symbols("a b", nonzero=True, real=True)
    result = minimize_result(
        (x + y) * sp.exp(-(x**2) / a**2 - y**2 / b**2),
        [sp.Le(x**2 / a**2 + y**2 / b**2, 1)],
        variables=(x, y),
        use_sos=False,
    )
    assert result.certified
    assert (
        sp.simplify(result.optimum_value + sp.sqrt(a * a + b * b) / sp.sqrt(2 * sp.E))
        == 0
    )


@pytest.mark.unsupported
@pytest.mark.xfail(
    strict=True,
    reason="logarithmic disk optimization is not yet solved exactly",
)
def test_logarithmic_disk_product():
    result = minimize_result(
        x * y**2 * 0 + x * y * sp.log(1 + x**2 + y**2),
        [sp.Le(x**2 + y**2, 1)],
        use_sos=False,
    )
    _assert_value(result, -sp.log(2) / 2)


@pytest.mark.parametrize(
    "objective,constraints",
    [
        pytest.param(
            sp.exp(x + y) * (1 - x**2 - y**2),
            [sp.Le(x**2 + y**2, 1)],
            id="exponential_times_disk_factor",
            marks=[
                pytest.mark.unsupported,
                pytest.mark.xfail(
                    strict=True,
                    reason="The optimizer does not yet recover a boundary witness for this exponential disk objective",
                ),
            ],
        ),
        pytest.param(
            (x + y) * sp.exp(-(x**2) - y**2),
            [sp.Ge(x, 0), sp.Ge(y, 0), sp.Le(x + y, 2)],
            id="gaussian_weighted_affine_triangle",
        ),
    ],
)
def test_exponential_objectives_on_bounded_regions(objective, constraints):
    result = minimize_result(objective, constraints, variables=(x, y), use_sos=False)
    assert result.optimum_value == 0 and result.certified and result.attained
    point = result.best_candidate.original_variable_values
    assert all(r.subs(point) is sp.true for r in constraints)


@pytest.mark.unsupported
@pytest.mark.xfail(
    strict=True,
    reason="general parametric optimization is outside the current supported set",
)
def test_parametric_bounded_gaussian():
    b = sp.symbols("b", real=True)
    result = minimize_result(
        x * y**2 * sp.exp(-(x**2) - 2 * y**2),
        [sp.Ge(x, 0), sp.Le(x, 2), sp.Ge(y, 0), sp.Le(y, b), sp.Gt(b, 1)],
        variables=(x, y),
        use_sos=False,
    )
    assert result.certified
    assert result.optimum_value == sp.Piecewise((0, b > 1), (sp.oo, True))


@pytest.mark.unsupported
@pytest.mark.xfail(
    strict=True,
    reason="global Gamma minima across poles are outside current verified function-domain semantics",
)
def test_gamma_global_minimum():
    result = minimize_result(sp.gamma(x), use_sos=False)
    assert result.optimum_value is -sp.oo and result.unbounded
    assert result.attained is False


@pytest.mark.unsupported
@pytest.mark.xfail(
    strict=True,
    reason="global zeta behavior across its pole is outside current optimization semantics",
)
def test_zeta_global_minimum():
    result = minimize_result(sp.zeta(x), use_sos=False)
    assert result.optimum_value is -sp.oo and result.unbounded
    assert result.attained is False


@pytest.mark.unsupported
@pytest.mark.xfail(
    strict=True,
    reason="unbounded-domain LambertW global-domain reasoning is not yet implemented",
)
def test_lambertw_global_minimum():
    result = minimize_result(sp.LambertW(x), use_sos=False)
    _assert_value(result, -1)
    _assert_point(result, x=-(sp.E**-1))


@pytest.mark.parametrize(
    "objective,constraints",
    [
        (1 + y / (1 + x + y), [sp.Gt(x, 1), sp.Lt(x, 0)]),
        (1 - y / (1 + x + y), [sp.Gt(x, 1), sp.Lt(x, 0)]),
        (1 + y / (1 + x + y), [sp.Lt(x + y, -2), sp.Gt(x + y, 0)]),
        (1 + y / (1 + x + y), [sp.Gt(x + y, 1), sp.Lt(x + y, 0)]),
    ],
)
def test_linear_fractional_strict_infeasible_status(objective, constraints):
    from symbopt import minimize_result

    result = minimize_result(objective, constraints, use_sos=False)
    assert result.infeasible and result.certified
    assert result.optimum_value is sp.oo and result.attained is False


@pytest.mark.unsupported
@pytest.mark.xfail(
    strict=True,
    reason="removable-singularity limit maximization is not yet represented as an attained optimizer",
)
@pytest.mark.parametrize("real_domain", [False, True])
def test_tanh_over_x_supremum(real_domain):
    result = minimize_result(-sp.tanh(x) / x, variables=x, use_sos=False)
    assert result.optimum_value == -1 and result.attained is False
    assert not result.best_candidate
