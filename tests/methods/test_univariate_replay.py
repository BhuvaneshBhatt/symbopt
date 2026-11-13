from dataclasses import replace

import pytest
import sympy as sp

import symbopt as so
from symbopt import (
    ComponentMinimumResult,
    GlobalMinimumCertificate,
    certify_global_minimum,
    finite_component_minimum,
    verify_univariate_certificate,
)

x, y = sp.symbols("x y", real=True)


class LyingTuple(tuple):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


fp = pytest.importorskip("funcprops")
track_domain = fp.track_domain


def assert_rejected(proof):
    assert proof.verify() is False
    if type(proof).__module__.startswith("symbopt."):
        assert so.verify_univariate_certificate(proof) is False
    else:
        assert fp.verify_certificate(proof) is False


@pytest.mark.parametrize("name", ["minimum_bounds", "maximum_bounds"])
def test_extrema_value_binding(name):
    x = sp.Symbol("x", real=True)
    c = so.certify_extrema_bounds(x**2, x, sp.Interval(-1, 1))
    assert c.verify() and c.complete
    assert_rejected(replace(c, **{name: LyingTuple((sp.Integer(-77),) * 2)}))


@pytest.mark.parametrize("outer", [True, False])
def test_deflated_root_binding(outer):
    x = sp.Symbol("x", real=True)
    D = sp.Interval.open(-sp.Rational(1, 4), sp.Rational(1, 4))
    leaf = fp.deflate_real_root(sp.sin(x) - x, x, D, 0, 3)
    assert leaf.verify() and leaf.roots == ((0, 3),)
    roots = (
        LyingTuple(((sp.Integer(77), 3),))
        if outer
        else (LyingTuple((sp.Integer(77), 3)),)
    )
    bad = replace(leaf, roots=roots)
    assert_rejected(bad)
    cover = fp.CoverageProof(leaf.context, (bad,), sp.S.EmptySet, True, 1, 3)
    assert_rejected(cover)
    with pytest.raises(ValueError, match="invalid supplied critical cover"):
        so.certify_extrema_bounds(
            -sp.cos(x) - x * x / 2, x, D, critical_covers=(cover,)
        )


def test_global_minimum_nonconstant_second_derivative():
    x = sp.symbols("x", real=True)
    c = certify_global_minimum(x**4 + x**2, x)
    assert c.point == 0 and c.unique and c.verify()


def test_certified_global_minimum():
    x = sp.symbols("x", real=True)
    c = certify_global_minimum((x - 3) ** 2 + 5, x)
    assert c.point == 3 and c.value == 5 and c.verify()


def test_minimum_component_replay():
    class LyingMinimum(ComponentMinimumResult):
        def verify(self):
            return True

    domain = sp.Interval(-1, 1)
    good = so.finite_component_minimum(x**2, x, domain, certified=True)
    assert good.verify() and good.infimum == 0
    false_result = LyingMinimum(
        **replace(good, infimum=sp.Integer(-77), minimizers=sp.FiniteSet(0)).__dict__
    )
    false = GlobalMinimumCertificate(
        x**2,
        x,
        sp.S.Zero,
        sp.Integer(-77),
        True,
        "finite-components",
        domain,
        False,
        good.domain_record,
        false_result,
    )
    assert_rejected(false)


def test_cancellation_retains_source_hole():
    f = sp.Mul(x, sp.Pow(x, -1, evaluate=False), evaluate=False)
    r = track_domain(f, x)
    assert r.effective == sp.S.Reals - sp.FiniteSet(0)
    assert r.exclusions == (sp.Ne(x, 0),)
    assert not GlobalMinimumCertificate(f, x, 1, 1, False, "convex-stationary").verify()
    optimum = finite_component_minimum(f, x, certified=True)
    assert optimum.certified and optimum.minimizers == r.effective


def test_finite_component_endpoint_minimum():
    D = sp.Union(sp.Interval(-3, -2), sp.Interval(1, 2))
    r = finite_component_minimum(x**2, x, D, certified=True)
    assert r.complete and r.certified and r.infimum == 1
    assert r.minimizers == sp.FiniteSet(1) and r.unique
    c = certify_global_minimum(x**2, x, D)
    assert c.verify() and c.point == 1 and c.requested_domain == D


def test_open_endpoint_infimum_not_minimum():
    r = finite_component_minimum(x**2, x, sp.Interval.open(1, 2), certified=True)
    assert r.certified and r.infimum == 1 and not r.attained
    assert r.minimizers is sp.S.EmptySet
    assert certify_global_minimum(x**2, x, sp.Interval.open(1, 2)) is None


def test_mixed_points_and_intervals():
    D = sp.Union(sp.Interval(2, 3), sp.FiniteSet(-1, 0))
    r = finite_component_minimum((x + 1) ** 2, x, D, certified=True)
    assert r.certified and r.infimum == 0 and r.minimizers == sp.FiniteSet(-1)


def test_pole_components_and_nonunique_minima():
    r = finite_component_minimum(x**2 + 1 / x**2, x, certified=True)
    assert r.certified and r.infimum == 2
    assert r.minimizers == sp.FiniteSet(-1, 1) and not r.unique
    c = certify_global_minimum(x**2 + 1 / x**2, x)
    assert c.verify() and not c.unique


def test_unbounded_below_is_complete_without_minimum():
    r = finite_component_minimum(1 / x, x, certified=True)
    assert r.certified and r.infimum == -sp.oo and not r.attained
    assert certify_global_minimum(1 / x, x) is None


def test_constant_open_component_has_all_points_as_minimizers():
    D = sp.Union(sp.Interval.open(-2, -1), sp.Interval.open(1, 2))
    r = finite_component_minimum(sp.Integer(3), x, D, certified=True)
    assert r.certified and r.infimum == 3 and r.minimizers == D
    c = certify_global_minimum(sp.Integer(3), x, D)
    assert c.verify() and not c.unique


def test_boolean_requested_domain_retained():
    requested = sp.And(x >= 2, x <= 3)
    c = certify_global_minimum(x**2, x, requested)
    assert c.verify() and c.point == 2
    assert c.requested_domain == requested and c.domain == sp.Interval(2, 3)


def test_empty_effective_domain_has_no_minimum():
    r = finite_component_minimum(1 / x, x, sp.FiniteSet(0), certified=True)
    assert r.certified and r.infimum == sp.oo and not r.attained
    assert certify_global_minimum(1 / x, x, sp.FiniteSet(0)) is None


def test_unsupported_result_status():
    r = finite_component_minimum(sp.sin(x) + x, x, certified=True)
    assert not r.complete and r.infimum is None and r.reason
    assert r.verify() is False and r.complete is False


def test_curvature_route_works_for_elementary_transcendentals():
    c = certify_global_minimum(sp.exp(x) + sp.exp(-x), x, x > -1)
    assert c.verify() and c.point == 0 and c.value == 2
    assert c.requested_domain == (x > -1)


def test_result_projection_keeps_provenance_and_status():
    r = finite_component_minimum(x**2, x, sp.Interval(-1, 1), certified=True)
    assert r.domain_record.effective == sp.Interval(-1, 1)
    assert r.complete is True and r.certified is True
    assert r.certificate is r and r.verify()


@pytest.mark.parametrize(
    "field,value",
    [
        ("infimum", -1),
        ("minimizers", sp.FiniteSet(99)),
        ("candidates", ()),
        ("expression", x**2 + 1),
        ("complete", 1),
    ],
)
def test_component_mutations_fail_replay(field, value):
    r = finite_component_minimum(x**2, x, sp.Interval(-1, 1), certified=True)
    bad = replace(r, **{field: value})
    assert not bad.verify()


def test_minimum_certificates_bind_source_and_metadata():
    c = certify_global_minimum(x**2, x, sp.Interval(-1, 1))
    assert verify_univariate_certificate(c, expr=x**2, var=x)
    assert not verify_univariate_certificate(c, expr=x**2 + 1, var=x)
    assert not verify_univariate_certificate(c, var=y)
    for bad in [
        replace(c, point=1),
        replace(c, unique=False),
        replace(c, coercive=True),
        replace(c, component_result=None),
        replace(c, domain=sp.S.Reals),
    ]:
        assert not bad.verify()
    assert c.complete and c.scope == "global-minimum-on-effective-domain"


def test_malformed_records_fail_safely():
    c = certify_global_minimum(x**2, x)
    assert not replace(c, domain_record=object()).verify()
    r = finite_component_minimum(x**2, x, certified=True)
    assert not replace(r, domain_record=object()).verify()


def test_coercivity_claim_on_finite_interval_cannot_replay():
    c = GlobalMinimumCertificate(
        x**2, x, 0, 0, True, "strict-convex-stationary", sp.Interval(-1, 1), True
    )
    assert not c.verify()


@pytest.mark.parametrize(
    "f,D",
    [(x**2 + sp.Float("0.1") * x, sp.S.Reals), (x**2, sp.Interval(sp.Float("0.1"), 1))],
)
def test_exact_component_route_declines_float_data(f, D):
    r = finite_component_minimum(f, x, D, certified=True)
    assert not r.complete and not r.certified and r.reason
    assert certify_global_minimum(f, x, D) is None
