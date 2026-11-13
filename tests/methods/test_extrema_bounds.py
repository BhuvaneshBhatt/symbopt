import sys
from dataclasses import replace

import pytest
import sympy as sp

import symbopt as so
from symbopt.extrema_bounds import ExtremumPiece

fp = pytest.importorskip("funcprops")

x = sp.Symbol("x", real=True)
P = fp.EnclosurePolicy(
    precision=20, max_precision=40, max_refinements=1, max_depth=7, max_cells=160
)


@pytest.fixture(scope="module")
def implicit():
    return so.certify_extrema_bounds(
        sp.exp(x) + x * x,
        x,
        sp.Interval(-1, 1),
        tolerance=sp.Rational(1, 10000),
        policy=P,
    )


def test_implicit_stationary_extrema(implicit):
    c = implicit
    assert c.verify() and so.verify_univariate_certificate(c) and c.complete
    assert c.coverage_complete and c.critical_coverage_complete and c.accuracy_complete
    assert not c.exact_complete
    w = sp.LambertW(sp.Rational(1, 2))
    value = (w * w + 2 * w).evalf(80)
    assert c.minimum_bounds[0] < value < c.minimum_bounds[1]
    assert c.maximum_bounds[0] < (sp.E + 1).evalf(80) < c.maximum_bounds[1]
    assert c.minimum_bounds[1] - c.minimum_bounds[0] <= c.tolerance
    assert c.check().outcome.value == "complete-coverage"


@pytest.mark.parametrize(
    "D,lower,upper",
    [
        (sp.Interval.open(0, 1), 0, 1),
        (sp.Union(sp.Interval(-2, -1), sp.Interval(1, 3)), 1, 9),
        (sp.FiniteSet(-1, 2), 1, 4),
    ],
)
def test_open_disconnected_and_finite_domains(D, lower, upper):
    c = so.certify_extrema_bounds(x * x, x, D, policy=P)
    assert c.verify() and c.complete
    assert c.minimum_bounds == (lower, lower) and c.maximum_bounds == (upper, upper)


def test_piecewise_seam_and_excluded_source_hole():
    f = sp.Piecewise((x * x, x < 0), (100, sp.Eq(x, 0)), (x + 10, True))
    c = so.certify_extrema_bounds(f, x, sp.Interval(-1, 1), policy=P)
    assert c.complete and c.verify()
    assert c.minimum_bounds == (0, 0) and c.maximum_bounds == (100, 100)
    raw = sp.Mul(x, sp.Pow(x, -1, evaluate=False), x * x, evaluate=False)
    c = so.certify_extrema_bounds(raw, x, sp.Interval(-1, 1), policy=P)
    assert c.complete and c.verify() and c.minimum_bounds == (0, 0)
    assert c.domain_record.effective.contains(0) is sp.S.false


def test_explicit_multiple_critical_points_are_all_used():
    c = so.certify_extrema_bounds((x * x - 2) ** 2, x, sp.Interval(-2, 2), policy=P)
    assert c.verify() and c.complete
    assert c.minimum_bounds[0] <= 0 <= c.minimum_bounds[1]
    assert c.maximum_bounds == (4, 4)


def test_partial_critical_geometry_gives_honest_value_brackets():
    cover = fp.adaptive_root_coverage(
        sp.exp(x) + 2 * x, x, sp.Interval.open(-1, 1), policy=replace(P, max_cells=0)
    ).proof
    c = so.certify_extrema_bounds(
        sp.exp(x) + x * x,
        x,
        sp.Interval(-1, 1),
        policy=replace(P, max_depth=0),
        critical_covers=(cover,),
    )
    assert c.verify() and c.coverage_complete and not c.critical_coverage_complete
    assert not c.accuracy_complete and not c.complete
    assert c.minimum_bounds[0] < sp.Rational(4, 5) < c.minimum_bounds[1]
    no_work = so.certify_extrema_bounds(
        sp.exp(x) + x * x, x, sp.Interval(-1, 1), policy=replace(P, max_cells=0)
    )
    assert no_work.verify() and not no_work.coverage_complete and not no_work.complete
    assert no_work.minimum_bounds is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: replace(c, minimum_bounds=(sp.S(-77), sp.S(-77))),
        lambda c: replace(c, maximum_bounds=(sp.S(77), sp.S(77))),
        lambda c: replace(c, pieces=c.pieces * 2),
        lambda c: replace(
            c, pieces=(replace(c.pieces[0], sites=c.pieces[0].sites[:-1]),)
        ),
        lambda c: replace(
            c, pieces=(replace(c.pieces[0], sites=c.pieces[0].sites * 2),)
        ),
        lambda c: replace(c, expression=x * x),
        lambda c: replace(c, tolerance=sp.Float("0.1")),
        lambda c: replace(c, schema_version=True),
    ],
)
def test_false_extrema_claims_rejected(implicit, mutate):
    c = mutate(implicit)
    assert not c.verify() and so.verify_univariate_certificate(c) is False


def test_nested_evidence_replay(implicit):
    class FalsePiece(ExtremumPiece):
        pass

    p = implicit.pieces[0]
    assert not replace(implicit, pieces=(FalsePiece(**p.__dict__),)).verify()

    # Observe calls without replacing any implementation used by replay.
    proposers = {
        so.certify_extrema_bounds.__code__,
        fp.adaptive_root_coverage.__code__,
        sp.solveset.__code__,
    }
    called = []

    def observe(frame, event, arg):
        if event == "call" and frame.f_code in proposers:
            called.append(frame.f_code.co_name)

    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        assert implicit.verify() and so.verify_univariate_certificate(implicit)
    finally:
        sys.setprofile(previous)
    assert called == []


def test_unsupported_pole_and_empty_domain_and_invalid_cover():
    c = so.certify_extrema_bounds(1 / x, x, sp.Interval.Lopen(0, 1), policy=P)
    assert not c.complete and not c.coverage_complete
    c = so.certify_extrema_bounds(x, x, sp.S.EmptySet, policy=P)
    assert c.verify() and not c.complete and c.minimum_bounds is None
    wrong = fp.adaptive_root_coverage(x, x, sp.Interval.open(-1, 1), policy=P).proof
    with pytest.raises(ValueError):
        so.certify_extrema_bounds(
            sp.exp(x) + x * x, x, sp.Interval(-1, 1), critical_covers=(wrong,), policy=P
        )
