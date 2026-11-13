"""Each stored field has an explicit mathematical or administrative role."""

from dataclasses import fields, replace

import pytest
import sympy as sp

import symbopt as so
from symbopt.extrema_bounds import ExtremumPiece, ExtremumSite
from symbopt.univariate_optimization import OptimizationCandidate

x, y = sp.symbols("x y", real=True)

FIELD_ROLES = {
    so.ComponentMinimumResult: {
        "mathematical": {
            "expression",
            "variable",
            "domain_record",
            "infimum",
            "minimizers",
            "candidates",
            "complete",
            "reason",
        },
        "administrative": {"certified"},
    },
    so.GlobalMinimumCertificate: {
        "mathematical": {
            "expression",
            "variable",
            "point",
            "value",
            "unique",
            "source",
            "domain",
            "coercive",
            "domain_record",
            "component_result",
        },
        "administrative": set(),
    },
    so.ExtremaBoundsProof: {
        "mathematical": {
            "expression",
            "variable",
            "requested",
            "domain_record",
            "tolerance",
            "pieces",
            "minimum_bounds",
            "maximum_bounds",
            "schema_version",
        },
        "administrative": {"tested", "reason"},
    },
    OptimizationCandidate: {
        "mathematical": {"point", "value", "attained", "domain", "kind"},
        "administrative": set(),
    },
    ExtremumPiece: {
        "mathematical": {"expression", "domain", "critical_cover", "sites"},
        "administrative": set(),
    },
    ExtremumSite: {
        "mathematical": {"region", "kind", "enclosure"},
        "administrative": set(),
    },
}


@pytest.mark.parametrize("record,roles", FIELD_ROLES.items())
def test_field_classification(record, roles):
    assert not roles["mathematical"] & roles["administrative"]
    assert {f.name for f in fields(record)} == roles["mathematical"] | roles[
        "administrative"
    ]


@pytest.mark.parametrize(
    "changes",
    [
        {"expression": x * x + 1},
        {"variable": y},
        {"domain_record": object()},
        {"infimum": -1},
        {"minimizers": sp.FiniteSet(77)},
        {"candidates": ()},
        {"candidates": None},
        {"complete": 1},
        {"reason": "invented"},
    ],
)
def test_component_fields(changes):
    result = so.finite_component_minimum(x * x, x, sp.Interval(-1, 1))
    assert (
        result.verify() and result.infimum == 0 and result.minimizers == sp.FiniteSet(0)
    )
    assert not so.verify_univariate_certificate(replace(result, **changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"expression": x * x + 1},
        {"variable": y},
        {"point": 1},
        {"value": -1},
        {"unique": False},
        {"source": "invented"},
        {"domain": sp.S.Reals},
        {"coercive": True},
        {"domain_record": None},
        {"component_result": None},
    ],
)
def test_minimum_fields(changes):
    proof = so.certify_global_minimum(x * x, x, sp.Interval(-1, 1))
    assert proof.point == 0 and proof.value == 0 and proof.verify()
    assert not so.verify_univariate_certificate(replace(proof, **changes))


fp = pytest.importorskip("funcprops")


def test_source_binding_and_owner():
    proof = so.certify_extrema_bounds(x * x, x, sp.Interval(-1, 1))
    assert so.verify_univariate_certificate(proof, expr=x * x, var=x)
    assert not so.verify_univariate_certificate(proof, expr=x * x + 1)
    assert not so.verify_univariate_certificate(proof, var=y)
    with pytest.raises(TypeError, match="unsupported certificate type"):
        fp.verify_certificate(proof)
    assert not so.verify_univariate_certificate(fp.track_domain(x * x, x))


@pytest.mark.parametrize(
    "changes",
    [
        {"expression": x * x + 1},
        {"variable": y},
        {"requested": sp.Interval(0, 1)},
        {"domain_record": object()},
        {"tolerance": -1},
        {"pieces": ()},
        {"minimum_bounds": (-1, -1)},
        {"maximum_bounds": (-1, -1)},
        {"schema_version": True},
    ],
)
def test_extrema_fields(changes):
    proof = so.certify_extrema_bounds(x * x, x, sp.Interval(-1, 1))
    assert proof.minimum_bounds[0] <= 0 <= proof.minimum_bounds[1]
    assert proof.maximum_bounds == (1, 1) and proof.verify()
    assert not so.verify_univariate_certificate(replace(proof, **changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"point": 77},
        {"value": -1},
        {"attained": False},
        {"domain": sp.S.Reals},
        {"kind": "invented"},
    ],
)
def test_candidate_fields(changes):
    result = so.finite_component_minimum(x * x, x, sp.Interval(-1, 1))
    first, *remaining = result.candidates
    assert first.point == -1 and first.value == 1 and first.attained
    bad = replace(result, candidates=(replace(first, **changes), *remaining))
    assert not bad.verify()


def test_checkpoint_transport():
    from symbopt.optimization_budget import _messages

    raw = sp.Mul(x, sp.Pow(x, -1, evaluate=False), evaluate=False)
    first = ("checkpoint", [raw, raw])
    second = ("return", [x * x, x * x])
    data = fp.pack_analysis_data(first) + fp.pack_analysis_data(second)
    actual = _messages(data)
    assert fp.analysis_data_equal(actual, [first, second])
    assert fp.track_domain(actual[0][1][0], x).effective.contains(0) is sp.false


def test_constant_source_binding():
    result = so.finite_component_minimum(3, x)
    assert result.infimum == 3 and result.minimizers is sp.S.Reals
    assert so.verify_univariate_certificate(result, expr=3, var=x)
    assert not so.verify_univariate_certificate(result, expr=4)
    assert not so.verify_univariate_certificate(object())
