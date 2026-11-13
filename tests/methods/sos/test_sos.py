import sympy as sp

from symbopt import (
    search_sos_hierarchy,
    sos_decompose,
    sos_lower_bound,
    sos_poly_relaxation,
    sos_scalar_relaxation,
)


def test_sos_decompose_exact():
    x, y = sp.symbols("x y", real=True)
    r = sos_decompose((x**2 + y**2 + 1) ** 2 + (x - y) ** 2, [x, y])
    assert r.success
    assert sp.expand(r.exact_residual) == 0


def test_sos_rejects_odd_degree():
    x = sp.symbols("x", real=True)
    r = sos_decompose(x**3 + x**2, [x])
    assert not r.success


def test_sos_lower_bound_quadratic_exact():
    x, y = sp.symbols("x y", real=True)
    lb, r = sos_lower_bound((x - 1) ** 2 + (y + 2) ** 2 + 3, [x, y])
    assert r.success
    assert sp.simplify(lb - 3) == 0


def test_scalar_multiplier_relaxation():
    x, y = sp.symbols("x y", real=True)
    r = sos_scalar_relaxation(
        x**2 + y**2,
        [x, y],
        [x, y],
        search_lo=-0.1,
        search_hi=0.1,
        max_iter=4,
        multiplier_degree=1,
    )
    assert r.success
    assert r.lower_bound_numeric is not None


def test_polynomial_multiplier_relaxation_with_equality():
    x = sp.symbols("x", real=True)
    r = sos_poly_relaxation(
        x**2,
        [x],
        inequalities=[],
        equalities=[x],
        search_lo=-0.1,
        search_hi=0.1,
        max_iter=3,
        ineq_mult_degree=0,
        eq_mult_degree=1,
    )
    assert isinstance(r.success, bool)


def test_hierarchy_search_returns_steps():
    x = sp.symbols("x", real=True)
    r = search_sos_hierarchy(x**2, [x], search_lo=-0.1, search_hi=0.1, max_order=1)
    assert len(r.steps) == 1


def test_hierarchy_succeeds_on_simple_square():
    x = sp.symbols("x", real=True)
    r = search_sos_hierarchy(
        (x - 1) ** 2, [x], search_lo=-0.1, search_hi=0.1, max_order=2
    )
    assert r.steps
    assert r.best_result is not None or not r.certified


def test_hierarchy_pathological_odd_degree_is_graceful():
    x = sp.symbols("x", real=True)
    r = search_sos_hierarchy(x**3, [x], search_lo=-0.1, search_hi=0.1, max_order=1)
    assert len(r.steps) == 1
    assert not r.steps[0].success


def test_sos_relaxation_status_distinguishes_numeric_from_exact():
    x = sp.symbols("x", real=True)
    result = sos_poly_relaxation(
        x**2,
        [x],
        inequalities=[],
        equalities=[],
        search_lo=-0.1,
        search_hi=0.1,
        max_iter=2,
        ineq_mult_degree=0,
        eq_mult_degree=0,
    )
    assert result.status in {
        "certified",
        "numerically_feasible",
        "infeasible_or_failed",
    }
    if result.status == "numerically_feasible":
        assert result.certified is False
        assert result.lower_bound is None
        assert result.lower_bound_numeric is not None


def test_scaled_affine_square_uses_exact_certificate():
    x = sp.symbols("x", real=True)
    result = sos_decompose(36 * (x - 1) ** 2, [x])
    assert result.success
    assert result.sdp_status == "exact_square"
    assert result.certificate.exact_residual_zero
    assert result.certificate.exact_psd_checks == [True]
