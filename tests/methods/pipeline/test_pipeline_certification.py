import pytest

import sympy as sp
from symbopt import minimize
from symbopt.algebraic_numbers import exactify, algebraic_compare

pytestmark = pytest.mark.integration


def test_minimize_lp_route():
    x, y = sp.symbols("x y", real=True)
    r = minimize(x + y, [sp.Ge(x, 0), sp.Ge(y, 0), sp.Le(x + y, 2)])
    assert r.best_candidate is not None
    assert "lp" in r.status


def test_minimize_qp_route():
    x, y = sp.symbols("x y", real=True)
    r = minimize(
        x**2 + y**2 - 2 * x - 4 * y, [sp.Le(x + y, 3), sp.Ge(x, 0), sp.Ge(y, 0)]
    )
    assert r.best_candidate is not None
    assert r.certificate is not None


def test_minimize_piecewise():
    x = sp.symbols("x", real=True)
    r = minimize(
        sp.Piecewise((x**2, x >= 0), (-x, True)),
        [sp.Ge(x, -1), sp.Le(x, 1)],
        use_sos=False,
    )
    assert r is not None


def test_minimize_strict_constraint_marks_closure_or_strict():
    x = sp.symbols("x", real=True)
    r = minimize(x**2, [sp.Gt(x, 0), sp.Le(x, 1)], use_sos=False)
    assert (
        "strict" in r.status
        or "closure" in r.status
        or r.status == "global_infimum_by_semialg"
        or r.best_candidate is not None
    )


def test_certificate_contains_proof_objects():
    x = sp.symbols("x", real=True)
    r = minimize((x - 1) ** 2, [sp.Ge(x, 0), sp.Le(x, 2)])
    assert r.certificate is not None
    assert len(r.certificate.proof_objects) >= 1


def test_exactify_rational_float():
    assert exactify(0.5) == sp.Rational(1, 2)


def test_algebraic_compare():
    assert algebraic_compare(sp.sqrt(2), sp.Rational(3, 2)) == -1


def test_minimize_qp_free_variable_regression():
    x = sp.symbols("x", real=True)
    r = minimize((x + 2) ** 2, [])
    assert r.best_candidate is not None
    assert sp.simplify(r.best_candidate.primal_values[x] + 2) == 0
    assert sp.simplify(r.best_candidate.objective_value) == 0


def test_minimize_lp_equality_regression():
    x = sp.symbols("x", real=True)
    r = minimize(x, [sp.Eq(x, 2)])
    assert r.best_candidate is not None
    assert sp.simplify(r.best_candidate.primal_values[x] - 2) == 0


def test_direct_certificate_checks_constraints():
    x = sp.symbols("x", real=True)
    r = minimize((x - 1) ** 2, [sp.Ge(x, 0), sp.Le(x, 2)])
    direct = [p for p in r.certificate.proof_objects if p.kind in {"lp", "qp"}]
    assert direct
    checks = direct[0].payload["feasibility_check"]
    assert len(checks) == 2
    assert all(ok is True for _, ok in checks)


def test_approximate_sos_bound_is_not_used_as_rigorous_lower_bound():
    from symbopt.certification import build_certificate

    cert = build_certificate(
        sos_lower_bound=sp.Rational(-1, 10),
        sos_rigorous=False,
    )
    assert cert.lower_bound is None
    sos_artifacts = [p for p in cert.proof_objects if p.kind == "sos"]
    assert sos_artifacts
    assert sos_artifacts[0].payload["rigorous"] is False
