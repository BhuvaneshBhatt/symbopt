import sympy as sp

from symbopt import minimize_result
from symbopt.convex_routing import try_direct_solver
from symbopt.precision import precision_provenance


def test_inexact_backend_precision():
    x = sp.symbols("x", real=True)
    center = sp.Float("0.123456789012345678901234567890123456789", 50)
    info = precision_provenance((x - center) ** 2, [], working_precision=60)
    assert info.float64_backends_allowed is False
    assert info.routing_policy == "preserve_high_precision_inexact"


def test_inexact_direct_routing():
    x = sp.symbols("x", real=True)
    center = sp.Float("0.123456789012345678901234567890123456789", 50)
    diagnostics = []
    direct = try_direct_solver(
        (x - center) ** 2, [], [x], diagnostics, working_precision=60
    )
    assert direct is None
    assert any(item.stage == "precision_policy" for item in diagnostics)


def test_minimize_records_precision_sensitive_routing_policy():
    x = sp.symbols("x", real=True)
    center = sp.Float("0.123456789012345678901234567890123456789", 50)
    result = minimize_result(
        (x - center) ** 2,
        variables=[x],
        working_precision=60,
        use_sos=True,
    )
    assert result.working_precision == 60
    assert result.precision_provenance["float64_backends_allowed"] is False
    assert (
        result.precision_provenance["routing_policy"]
        == "preserve_high_precision_inexact"
    )
    assert any(item.stage == "precision_policy" for item in result.diagnostics)


def test_exact_high_precision_routing():
    x = sp.symbols("x", real=True)
    result = minimize_result(
        (x - sp.Rational(1, 3)) ** 2, variables=[x], working_precision=100
    )
    assert result.certified is True
    assert result.optimum_value == 0
    assert result.precision_provenance["float64_backends_allowed"] is True
