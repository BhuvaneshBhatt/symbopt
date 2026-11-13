import sympy as sp

from symbopt import estimate_sos_cost, plan_sos_search, search_sos_certificate


def test_cost_planner_accepts_small_problem():
    x, y = sp.symbols("x y", real=True)
    estimate = estimate_sos_cost(x**4 + y**4, [x, y])
    plan = plan_sos_search(x**4 + y**4, [x, y])
    assert estimate.applicable
    assert estimate.relaxation_order == 2
    assert estimate.main_gram_dimension == 6
    assert plan.launch


def test_cost_planner_defers_large_dense_problem():
    variables = sp.symbols("x0:8", real=True)
    polynomial = sum(variable**8 for variable in variables)
    plan = plan_sos_search(polynomial, variables)
    assert plan.estimate.applicable
    assert not plan.launch


def test_nonpolynomial_is_not_applicable():
    x = sp.symbols("x", real=True)
    plan = plan_sos_search(sp.sin(x), [x])
    assert not plan.estimate.applicable
    assert not plan.launch


def test_direct_unconstrained_certificate_is_exact():
    x, y = sp.symbols("x y", real=True)
    result = search_sos_certificate((x + y) ** 2 + (x - y) ** 2, [x, y])
    assert result.attempted
    assert result.certified
    assert result.certificate.verified
    assert result.exact_residual_zero
    assert all(check is True for check in result.exact_psd_checks)


def test_constrained_gram_blocks():
    x = sp.symbols("x", real=True)
    # x is nonnegative on x >= 0: x = 0 + 1*x.
    result = search_sos_certificate(
        x,
        [x],
        inequalities=[x],
        ineq_mult_degree=0,
        force=True,
    )
    assert result.attempted
    assert result.certified
    assert len(result.gram_matrices) >= 2
    assert result.certificate.verified


def test_uncertified_numeric_data_is_not_exposed_as_certificate():
    x = sp.symbols("x", real=True)
    result = search_sos_certificate(-(x**2) - 1, [x], force=True)
    assert result.attempted
    assert not result.certified
    assert result.certificate is None
    assert result.raw_result is not None
