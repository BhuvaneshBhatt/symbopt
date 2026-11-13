import sympy as sp

from symbopt import (
    PipelineResult,
    PositivstellensatzCertificate,
    SOSCertificateSearchResult,
    maximize,
    minimize,
)


def test_compact_minimum():
    x = sp.symbols("x", real=True)
    result = minimize((x - 2) ** 2, variables=[x], use_sos=False)
    assert type(result) is list
    assert result == [0, {x: 2}]


def test_full_output_returns_pipeline_result():
    x = sp.symbols("x", real=True)
    result = minimize((x - 2) ** 2, variables=[x], use_sos=False, full_output=True)
    assert isinstance(result, PipelineResult)
    assert result.optimum_value == 0


def test_unbounded_extrema_use_infinite_optimizer_coordinates():
    x = sp.symbols("x", real=True)
    assert minimize(x, variables=[x], use_sos=False) == [-sp.oo, {x: -sp.oo}]
    assert maximize(x, variables=[x], use_sos=False) == [sp.oo, {x: sp.oo}]


def test_unattained_finite_infimum_uses_limiting_coordinate():
    x = sp.symbols("x", real=True)
    assert minimize(sp.exp(x), variables=[x], use_sos=False) == [0, {x: -sp.oo}]
    assert minimize(x, [x > 0], variables=[x], use_sos=False) == [0, {x: 0}]


def test_empty_feasible_set():
    x = sp.symbols("x", real=True)
    constraints = [x > 1, x < 0]
    assert minimize(x, constraints, variables=[x], use_sos=False) == [sp.oo, {}]
    assert maximize(x, constraints, variables=[x], use_sos=False) == [-sp.oo, {}]


def test_certificate_type_names():
    assert PositivstellensatzCertificate.__name__ == "PositivstellensatzCertificate"
    assert SOSCertificateSearchResult.__name__ == "SOSCertificateSearchResult"
