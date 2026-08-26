import sympy as sp

from symbopt.certification import build_certificate, exact_float_scalar
from symbopt.problem_models import IntervalResult


def test_exact_float_scalar_preserves_binary_value():
    value = 0.3
    exact = exact_float_scalar(value)
    assert exact == sp.Rational(*value.as_integer_ratio())
    assert exact != sp.Rational(3, 10)


def test_rigorous_interval_bound_is_not_nsimplified_upward():
    result = IntervalResult(
        best_lower_bound=0.3,
        best_upper_bound=0.4,
        best_point={},
        boxes_processed=1,
        remaining_boxes=0,
        feasible_boxes=1,
        status="success",
        lower_bound_rigorous=True,
        upper_bound_rigorous=True,
    )
    certificate = build_certificate(interval_result=result)
    assert certificate.lower_bound == sp.Rational(*(0.3).as_integer_ratio())
    assert certificate.upper_bound == sp.Rational(*(0.4).as_integer_ratio())


def test_interval_epsilon_gap_is_not_exact_certification():
    from symbopt import minimize

    x = sp.symbols("x", real=True)
    result = minimize(
        (x - sp.Rational(1, 3)) ** 2,
        [sp.Ge(x, 0), sp.Le(x, 1)],
        use_sos=False,
        interval_tol=1e-2,
    )
    if result.status == "rigorously_bounded_by_interval":
        assert result.certified is False
        assert result.certification_gap is not None
        assert result.certification_gap >= 0
