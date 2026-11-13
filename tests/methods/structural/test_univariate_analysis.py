import pytest
import sympy as sp

from symbopt.univariate_analysis import (
    analyze_univariate,
    clear_univariate_analysis_cache,
    try_univariate_range,
)


def test_univariate_analysis_is_cached_per_normalized_problem():
    x = sp.symbols("x", real=True)
    clear_univariate_analysis_cache()
    first = analyze_univariate(x - sp.log(x), (x > 0,), x)
    second = analyze_univariate(x - sp.log(x), (x > 0,), x)
    assert first is second
    assert first.requested_domain == sp.Interval.open(0, sp.oo)
    assert first.feasible_domain == sp.Interval.open(0, sp.oo)


def test_shared_stationary_and_limit_facts():
    x = sp.symbols("x", real=True)
    analysis = analyze_univariate(x - sp.log(x), (x > 0,), x)
    assert analysis.stationary_points == (sp.Integer(1),)
    assert analysis.endpoint_value(0, direction="+", attained=False) == sp.oo
    assert analysis.endpoint_value(sp.oo) == sp.oo


def test_shared_analysis_period_is_reused_by_periodic_routes():
    x = sp.symbols("x", real=True)
    analysis = analyze_univariate(sp.sin(2 * x) + sp.cos(4 * x), (), x)
    assert sp.simplify(analysis.period - sp.pi) == 0
    before = analysis.period
    after = analysis.period
    assert before is after


def test_range_route_uses_shared_analysis_result():
    x = sp.symbols("x", real=True)
    objective = sp.exp(x)
    constraints = (x >= 0, x <= 1)
    analysis = analyze_univariate(objective, constraints, x)
    result = try_univariate_range(objective, constraints, (x,), analysis=analysis)
    assert result.value == 1
    assert result.attained is True
    assert result.certified is True
    assert result.metadata["analysis"] == "shared univariate function-property analysis"


def test_provider_summary_does_not_force_unused_expensive_facts():
    x = sp.symbols("x", real=True)
    clear_univariate_analysis_cache()
    analysis = analyze_univariate(x**2, (), x)
    assert analysis is not None
    summary = analysis.provider_summary
    assert summary["cached_facts"] == ()
    _ = analysis.natural_domain
    summary = analysis.provider_summary
    assert "natural_domain" in summary["cached_facts"]
    assert "monotonic_partition" not in summary["cached_facts"]


def test_funcprops_period_query_accepts_set_domains():
    pytest.importorskip("funcprops")
    x = sp.symbols("x", real=True)
    clear_univariate_analysis_cache()
    analysis = analyze_univariate(sp.sin(x), (x >= 0,), x)
    assert analysis is not None
    result = analysis.period_result
    assert result is not None
    assert getattr(result, "domain", None) == sp.Interval(0, sp.oo)
    assert sp.simplify(analysis.period - 2 * sp.pi) == 0


def test_funcprops_boolean_range_condition_is_preserved():
    pytest.importorskip("funcprops")
    pytest.importorskip("semialg")
    x = sp.symbols("x", real=True)
    clear_univariate_analysis_cache()
    analysis = analyze_univariate(x**2, (x >= -1, x <= 2), x)
    assert analysis is not None
    range_data = analysis.range_condition
    assert range_data is not None
    value, condition = range_data
    assert sp.simplify(condition.subs(value, 0)) is sp.S.true
    assert sp.simplify(condition.subs(value, 5)) is sp.S.false


def test_funcprops_real_loci():
    pytest.importorskip("funcprops")
    from symbopt.univariate_analysis import certified_monotone_direction

    x = sp.Symbol("x", real=True)
    analysis = analyze_univariate(1 / x, (), x)
    assert analysis.singularity_set == sp.FiniteSet(0)
    assert analysis.singularity_analysis.poles[0].condition == sp.Eq(x, 0)
    assert analysis.natural_domain == sp.Union(
        sp.Interval.open(-sp.oo, 0), sp.Interval.open(0, sp.oo)
    )
    discontinuous = analyze_univariate(sp.sign(x), (), x)
    assert discontinuous.discontinuities[0].condition.subs(x, 0) is sp.S.true
    assert certified_monotone_direction(sp.exp(x), x, sp.S.Reals) is True
    assert certified_monotone_direction(-sp.exp(x), x, sp.S.Reals) is False
    assert certified_monotone_direction(x**2, x, sp.S.Reals) is None


def test_real_domain_fallback_without_funcprops(monkeypatch):
    import symbopt.univariate_analysis as univariate

    x = sp.Symbol("x", real=True)
    monkeypatch.setattr(univariate, "_funcprops_module", lambda: None)
    clear_univariate_analysis_cache()
    try:
        analysis = analyze_univariate(x - sp.log(x), (x > 0,), x)
        assert analysis.feasible_domain == sp.Interval.open(0, sp.oo)
        assert analysis.stationary_points == (sp.Integer(1),)

        radical = analyze_univariate(-sp.sqrt(x), (), x)
        assert radical.feasible_domain == sp.Interval(0, sp.oo)
        assert radical.range_set == sp.Interval(-sp.oo, 0)

        complex_logs = sp.log(x + sp.I) + sp.log(x - sp.I)
        assert univariate._needs_real_locus(complex_logs)
        assert univariate._needs_real_locus(sp.log(x) ** 2)
        assert univariate._needs_real_locus(sp.Abs(sp.log(x)))
        assert univariate._needs_real_locus(sp.log(x) + sp.log(-x) - sp.log(-(x**2)))
    finally:
        clear_univariate_analysis_cache()
