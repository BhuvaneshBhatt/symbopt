from dataclasses import replace

import pytest
import sympy as sp

import symbopt as so

fp = pytest.importorskip("funcprops")


def test_partial_extrema_accuracy():
    x = sp.Symbol("x", real=True)
    policy = fp.EnclosurePolicy(
        precision=5, max_precision=5, max_refinements=0, max_depth=1, max_cells=32
    )
    proof = so.certify_extrema_bounds(
        sp.exp(x) + x * x,
        x,
        sp.Interval(-1, 1),
        tolerance=sp.Rational(1, 10**80),
        policy=policy,
    )
    assert (
        proof.verify() and proof.coverage_complete and proof.minimum_bounds is not None
    )
    assert not proof.accuracy_complete and not proof.complete
    assert (
        so.verify_univariate_certificate(
            replace(proof, minimum_bounds=(sp.S.Zero, sp.S.Zero))
        )
        is False
    )


def test_expired_budget():
    x = sp.Symbol("x", real=True)
    proof = so.certify_extrema_bounds(
        x, x, sp.Interval(0, 1), policy=fp.EnclosurePolicy(seconds=0.001)
    )
    assert proof.reason == "work-budget-exhausted"
    assert proof.minimum_bounds is None and not proof.complete and not proof.verify()


def test_completed_worker():
    x = sp.Symbol("x", real=True)
    proof = so.certify_extrema_bounds(
        x, x, sp.Interval(0, 1), policy=fp.EnclosurePolicy(seconds=30)
    )
    assert proof.minimum_bounds == (0, 0) and proof.maximum_bounds == (1, 1)
    assert proof.complete and so.verify_univariate_certificate(proof)


def test_minimum_budget():
    x = sp.Symbol("x", real=True)
    assert so.certify_global_minimum(x * x, x, work_budget=0.001) is None
    proof = so.certify_global_minimum(x * x, x, work_budget=30)
    assert proof.value == 0 and proof.point == 0 and proof.verify()


def test_budget_validates_inputs():
    x = sp.Symbol("x", real=True)
    with pytest.raises(ValueError, match="tolerance"):
        so.certify_extrema_bounds(
            x,
            x,
            sp.Interval(0, 1),
            tolerance=-1,
            policy=fp.EnclosurePolicy(seconds=0.001),
        )
    with pytest.raises(TypeError, match="policy"):
        so.certify_extrema_bounds(x, x, sp.Interval(0, 1), policy=object())
