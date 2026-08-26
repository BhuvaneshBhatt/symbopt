"""Documented optimization cases that are outside the current capability set."""

import math

import pytest
import sympy as sp

from symbopt import minimize


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
    result = minimize(-objective, constraints or [], **kwargs)
    candidate = _candidate(result)
    return -candidate.objective_value, candidate.original_variable_values


x, y, z, t, v = sp.symbols("x y z t v", real=True)

pytestmark = [pytest.mark.reference, pytest.mark.future]


@pytest.mark.xfail(
    run=False,
    strict=True,
    reason="general exact exponential stationary solving is not yet implemented",
)
def test_exponential_base_two_productlog_minimum():
    result = minimize(x**2 + 2**x, use_sos=False)
    w = sp.LambertW(sp.log(2) ** 2 / 2)
    expected_x = -w / sp.log(2)
    expected = sp.exp(-w) + w**2 / sp.log(2) ** 2
    _assert_value(result, expected)
    _assert_point(result, x=expected_x)


@pytest.mark.xfail(
    run=False,
    strict=True,
    reason="global trig optimization over periodic branch constraints remains incomplete",
)
def test_tangent_squared_with_sine_constraint():
    result = minimize(
        sp.tan(2 * x - sp.pi / 2) ** 2,
        [
            sp.Ge(sp.sin(x), -sp.Rational(1, 2)),
            sp.Le(sp.sin(x), sp.Rational(1, 2)),
        ],
        use_sos=False,
    )
    _assert_value(result, sp.Rational(1, 3))
    _assert_point(result, x=sp.pi / 6)


@pytest.mark.xfail(
    run=False,
    strict=True,
    reason="mixed transcendental equality optimization is not yet solved exactly",
)
def test_trigonometric_equality_two_variable_minimum():
    result = minimize(
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


@pytest.mark.xfail(
    run=False,
    strict=True,
    reason="full source-specific WorkingPrecision/result-precision compatibility is not yet guaranteed",
)
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
def test_quartic_ball_precision_variants(variant):
    # The exact mathematical core is exercised above. These retained cases track
    # These precision-provenance semantics are not part of the symbopt API.
    raise NotImplementedError(variant)


@pytest.mark.xfail(
    run=False,
    strict=True,
    reason="general parametric optimization is intentionally deferred",
)
def test_parametric_quartic_ball_minimum():
    a = sp.symbols("a", real=True)
    minimize(
        a * x**2 - y**3,
        [sp.Le(x**4 + y**4, 3)],
        use_sos=False,
    )


@pytest.mark.xfail(
    run=False,
    strict=True,
    reason="exact LambertW recovery for this unconstrained exponential problem is incomplete",
)
def test_exponential_quadratic_lambertw_minimum():
    result = minimize(sp.exp(x) + 5 * (x - 4) ** 2, use_sos=False)
    expected_x = 4 - sp.LambertW(sp.E**4 / 10)
    _assert_point(result, x=expected_x)


@pytest.mark.xfail(
    run=False,
    strict=True,
    reason="boxed transcendental algebraization currently needs stronger auxiliary handling",
)
def test_bounded_exponential_quadratic_lambertw_minimum():
    result = minimize(
        sp.exp(x) + 5 * (x - 4) ** 2,
        [sp.Ge(x, 1), sp.Le(x, 3)],
        use_sos=False,
    )
    expected_x = 4 - sp.LambertW(sp.E**4 / 10)
    _assert_point(result, x=expected_x)


@pytest.mark.xfail(
    run=False,
    strict=True,
    reason="high-frequency exact trigonometric global root isolation remains unsupported",
)
def test_high_frequency_trigonometric_minimum():
    expr = -sp.Rational(4070375628, 10**11) * sp.cos(
        sp.Rational(5118786816, 10**9) * t
    ) + sp.Rational(4070375628, 10**11) * sp.cos(sp.Rational(1668728700, 10**8) * t)
    result = minimize(expr, use_sos=False)
    assert _candidate(result).objective_value_numeric < -0.077


@pytest.mark.xfail(
    run=False,
    strict=True,
    reason="high-frequency exact trigonometric maximization remains unsupported",
)
def test_high_frequency_trigonometric_maximum():
    expr = -sp.Rational(4070375628, 10**11) * sp.cos(
        sp.Rational(5118786816, 10**9) * t
    ) + sp.Rational(4070375628, 10**11) * sp.cos(sp.Rational(1668728700, 10**8) * t)
    value, _ = _maximize(expr, use_sos=False)
    assert float(value) > 0.0805


@pytest.mark.xfail(
    run=False,
    strict=True,
    reason="floating rational-objective precision compatibility is not a symbopt API contract",
)
def test_float_rational_objective_precision_behavior():
    result = minimize(5280 * v / (12 + sp.Float("0.025") * v**2), use_sos=False)
    assert abs(_candidate(result).objective_value_numeric + 880 * math.sqrt(30)) < 1e-8


@pytest.mark.xfail(
    run=False,
    strict=True,
    reason="bounded exponential-over-rational exact LambertW recovery remains incomplete",
)
def test_exponential_over_x_exact_minimum():
    result = minimize(
        sp.exp(x) + sp.Rational(39, 5) / x,
        [sp.Ge(x, 1), sp.Le(x, 2)],
        use_sos=False,
    )
    expected_x = 2 * sp.LambertW(sp.sqrt(sp.Rational(39, 5)) / 2)
    _assert_point(result, x=expected_x)


@pytest.mark.xfail(
    run=False,
    strict=True,
    reason="exact and floating exponential-over-x compatibility remains incomplete",
)
def test_exponential_over_x_exact_float_agreement():
    exact = minimize(
        sp.exp(x) + sp.Rational(39, 5) / x,
        [sp.Ge(x, 1), sp.Le(x, 2)],
        use_sos=False,
    )
    numeric = minimize(
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


@pytest.mark.xfail(
    run=False,
    strict=True,
    reason="boxed cosine objective currently needs stronger transcendental auxiliary handling",
)
def test_quadratic_plus_cosine_closed_interval():
    result = minimize(
        x**2 + sp.cos(x),
        [sp.Ge(x, 0), sp.Le(x, 3)],
        use_sos=False,
    )
    _assert_value(result, 1)
    _assert_point(result, x=0)


@pytest.mark.xfail(
    run=False,
    strict=True,
    reason="unbounded transcendental maximization with unattained endpoint remains unsupported",
)
def test_negative_log_unbounded_maximum():
    value, _ = _maximize(-sp.log(x), use_sos=False)
    assert value is sp.oo


@pytest.mark.xfail(
    run=False,
    strict=True,
    reason="unbounded square-root maximization remains unsupported",
)
def test_sqrt_unbounded_maximum():
    value, _ = _maximize(sp.sqrt(x), use_sos=False)
    assert value is sp.oo


@pytest.mark.xfail(
    run=False,
    strict=True,
    reason="high-degree algebraic equality system with precision-controlled Root output remains unsupported",
)
def test_high_degree_three_variable_equalities():
    result = minimize(
        x + y + z,
        [
            sp.Eq(x**4 + y**4 + z**4, 1),
            sp.Eq(x**2 - 2 * y**2, sp.Rational(1, 2)),
        ],
        use_sos=False,
    )
    assert abs(_candidate(result).objective_value_numeric + 2.04718) < 1e-4


@pytest.mark.xfail(
    run=False,
    strict=True,
    reason="exact exponential objective over a box is not yet globally solved symbolically",
)
def test_exponential_quartic_difference_box():
    result = minimize(
        sp.exp(x**4 - y**4),
        [sp.Le(sp.Abs(x), 1), sp.Le(sp.Abs(y), 1)],
        use_sos=False,
    )
    _assert_value(result, sp.E**-1)


@pytest.mark.xfail(
    run=False,
    strict=True,
    reason="mixed exponential-polynomial disk optimization remains unsupported exactly",
)
def test_linear_times_gaussian_disk():
    result = minimize(
        (x + 2 * y) * sp.exp(-(x**2) - y**2),
        [sp.Le(x**2 + y**2, 1)],
        use_sos=False,
    )
    expected = (-2 * sp.sqrt(sp.Rational(2, 5)) - 1 / sp.sqrt(10)) / sp.sqrt(sp.E)
    _assert_value(result, expected)


@pytest.mark.xfail(
    run=False,
    strict=True,
    reason="general parametric optimization is intentionally deferred",
)
def test_parametric_elliptic_gaussian():
    a, b = sp.symbols("a b", nonzero=True, real=True)
    minimize(
        (x + y) * sp.exp(-(x**2) / a**2 - y**2 / b**2),
        [sp.Le(x**2 / a**2 + y**2 / b**2, 1)],
        use_sos=False,
    )


@pytest.mark.xfail(
    run=False,
    strict=True,
    reason="logarithmic disk optimization is not yet solved exactly",
)
def test_logarithmic_disk_product():
    result = minimize(
        x * y**2 * 0 + x * y * sp.log(1 + x**2 + y**2),
        [sp.Le(x**2 + y**2, 1)],
        use_sos=False,
    )
    _assert_value(result, -sp.log(2) / 2)


@pytest.mark.xfail(
    run=False,
    strict=True,
    reason="unevaluated symbolic-head behavior has no direct symbopt API analogue",
)
@pytest.mark.parametrize(
    "objective,constraints",
    [
        (
            sp.exp(x + y) * (1 - x**2 - y**2),
            [sp.Le(x**2 + y**2, 1)],
        ),
        (
            (x + y) * sp.exp(-(x**2) - y**2),
            [sp.Ge(x, 0), sp.Ge(y, 0), sp.Le(x + y, 2)],
        ),
    ],
)
def test_symbolic_head_only_transcendental_cases(objective, constraints):
    minimize(objective, constraints, use_sos=False)
    raise AssertionError("Python has no symbolic unevaluated Minimize head result")


@pytest.mark.xfail(
    run=False,
    strict=True,
    reason="general parametric optimization is intentionally deferred",
)
def test_parametric_bounded_gaussian():
    b = sp.symbols("b", real=True)
    minimize(
        x * y**2 * sp.exp(-(x**2) - 2 * y**2),
        [sp.Ge(x, 0), sp.Le(x, 2), sp.Ge(y, 0), sp.Le(y, b), sp.Gt(b, 1)],
        use_sos=False,
    )


@pytest.mark.xfail(
    run=False,
    strict=True,
    reason="global Gamma minima across poles are outside current verified function-domain semantics",
)
def test_gamma_global_minimum():
    result = minimize(sp.gamma(x), use_sos=False)
    assert _candidate(result).objective_value is -sp.oo


@pytest.mark.xfail(
    run=False,
    strict=True,
    reason="global zeta behavior across its pole is outside current optimization semantics",
)
def test_zeta_global_minimum():
    result = minimize(sp.zeta(x), use_sos=False)
    assert _candidate(result).objective_value is -sp.oo


@pytest.mark.xfail(
    run=False,
    strict=True,
    reason="unbounded-domain LambertW global-domain reasoning is not yet implemented",
)
def test_lambertw_global_minimum():
    result = minimize(sp.LambertW(x), use_sos=False)
    _assert_value(result, -1)
    _assert_point(result, x=-(sp.E**-1))


@pytest.mark.xfail(
    run=False,
    strict=True,
    reason="strict affine infeasibility is not yet promoted to a dedicated linear-fractional infeasible status",
)
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
    result = minimize(objective, constraints, use_sos=False)
    assert "infeas" in result.status.lower()


@pytest.mark.xfail(
    run=False,
    strict=True,
    reason="removable-singularity limit maximization is not yet represented as an attained optimizer",
)
@pytest.mark.parametrize("real_domain", [False, True])
def test_tanh_over_x_supremum(real_domain):
    value, point = _maximize(sp.tanh(x) / x, use_sos=False)
    assert sp.simplify(value - 1) == 0
    assert sp.simplify(point[x]) == 0
