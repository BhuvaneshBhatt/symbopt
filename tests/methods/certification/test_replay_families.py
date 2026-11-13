"""Coherent mutations of premises, witnesses and coverage across proof families."""

from dataclasses import fields, replace

import pytest
import sympy as sp

from symbopt import (
    PositivstellensatzCertificate,
    PreorderingGenerator,
    SOSCertificate,
    minimize_quadratic,
    verify_infeasibility_certificate,
    verify_quadratic_certificate,
)
from symbopt.problem_models import IntervalResult, OptimalityCertificate, ProofArtifact
from symbopt.quadratic_faces import CoordinateBounds, FaceRecord
from symbopt.quadratic_lp import LinearProof, LinearSystem, verify_linear_proof
from symbopt.verified_intervals import VerifiedBound

x, y = sp.symbols("x y", real=True)

# SOS, quadratic top-level, and univariate records also have inventories in
# their owning suites. These nested records and enclosure summaries complete
# the coverage; summaries are explicitly not independently replayable proofs.
FIELD_ROLES = {
    PositivstellensatzCertificate: {
        "sos_certificate": "witness",
        "generators": "domain_premise",
        "equalities": "domain_premise",
        "inequalities": "domain_premise",
        "groebner_basis": "derived_summary",
        "gram_bases": "witness",
        "gram_multipliers": "domain_premise",
        "ideal_remainder": "derived_summary",
        "exact_psd_checks": "derived_summary",
        "verified": "derived_summary",
    },
    PreorderingGenerator: {"indices": "domain_premise", "polynomial": "domain_premise"},
    CoordinateBounds: {"intervals": "coverage", "proofs": "witness"},
    FaceRecord: {
        "active": "coverage",
        "kind": "statement",
        "solution": "witness",
        "value": "statement",
        "proof": "witness",
    },
    LinearSystem: {
        "size": "coordinate_scope",
        "inequalities": "domain_premise",
        "lower": "domain_premise",
        "equalities": "domain_premise",
        "rhs": "domain_premise",
    },
    VerifiedBound: {
        "lower": "trusted_enclosure",
        "upper": "trusted_enclosure",
        "rigorous": "producer_status",
        "method": "provenance",
        "backend": "provenance",
        "domain": "producer_status",
    },
    IntervalResult: {
        "best_lower_bound": "trusted_enclosure",
        "best_upper_bound": "trusted_enclosure",
        "best_point": "proposal",
        "boxes_processed": "administrative",
        "remaining_boxes": "administrative",
        "feasible_boxes": "administrative",
        "status": "producer_status",
        "lower_bound_rigorous": "producer_status",
        "upper_bound_rigorous": "producer_status",
        "lower_bound_method": "provenance",
        "interval_backend": "provenance",
        "contractions_applied": "administrative",
        "contraction_method": "provenance",
    },
    OptimalityCertificate: {
        "primal_solution": "derived_summary",
        "primal_value": "derived_summary",
        "lower_bound": "derived_summary",
        "upper_bound": "derived_summary",
        "gap": "derived_summary",
        "proof_objects": "evidence_container",
        "status": "derived_summary",
    },
    ProofArtifact: {
        "kind": "provenance",
        "summary": "administrative",
        "payload": "evidence_container",
    },
}


@pytest.mark.parametrize(
    "record,roles", FIELD_ROLES.items(), ids=lambda v: getattr(v, "__name__", None)
)
def test_field_roles(record, roles):
    actual = {field.name for field in fields(record)}
    assert actual == set(roles), (
        f"unclassified={actual - set(roles)}, stale={set(roles) - actual}"
    )


@pytest.fixture
def infeasibility_proof():
    inequalities = (x, -x - 1)
    sos = SOSCertificate(
        [sp.eye(1), sp.eye(1)],
        [],
        sp.S.Zero,
        [True, True],
        True,
        polynomial=-sp.S.One,
        variables=(x,),
        gram_bases=((sp.S.One,),) * 2,
        gram_multipliers=inequalities,
        equalities=(),
    )
    return PositivstellensatzCertificate(
        sos,
        tuple(PreorderingGenerator((i,), g) for i, g in enumerate(inequalities)),
        (),
        (),
        sos.gram_bases,
        inequalities,
        sp.S.Zero,
        (True, True),
        True,
        inequalities,
    )


def test_infeasibility_source_binding(infeasibility_proof):
    proof = infeasibility_proof
    assert verify_infeasibility_certificate(proof, x, inequalities=(x, -x - 1))
    assert not verify_infeasibility_certificate(proof, x, inequalities=(x, 1 - x))
    assert not verify_infeasibility_certificate(proof, x, equalities=(x,))
    assert not verify_infeasibility_certificate(proof, y)


@pytest.mark.parametrize(
    "changes",
    [
        {"generators": ()},
        {"inequalities": (x, 1 - x)},
        {"gram_multipliers": (-1, -1)},
        {"gram_bases": ((x,), (x,))},
        {"groebner_basis": (sp.S.One,)},
        {"ideal_remainder": 1},
        {"exact_psd_checks": (False, True)},
        {"verified": False},
        {"equalities": (x,)},
    ],
)
def test_infeasibility_mutations(infeasibility_proof, changes):
    assert not verify_infeasibility_certificate(
        replace(infeasibility_proof, **changes), x
    )


@pytest.mark.parametrize("indices", [(0, 0), (-1,), (2,), (), (True,)])
def test_preordering_indices(infeasibility_proof, indices):
    proof = replace(infeasibility_proof, generators=(PreorderingGenerator(indices, x),))
    assert not verify_infeasibility_certificate(proof, x)


def test_unjustified_negative_multiplier(infeasibility_proof):
    # Every numerical summary and the identity agree on -1 = 1*(-1).
    # The feasible premise x >= 0 does not justify the multiplier -1.
    sos = replace(
        infeasibility_proof.sos_certificate,
        gram_matrices=[sp.eye(1)],
        gram_bases=((sp.S.One,),),
        gram_multipliers=(-sp.S.One,),
        exact_psd_checks=[True],
    )
    proof = replace(
        infeasibility_proof,
        sos_certificate=sos,
        inequalities=(x,),
        generators=(PreorderingGenerator((0,), x),),
        gram_bases=sos.gram_bases,
        gram_multipliers=sos.gram_multipliers,
        exact_psd_checks=(True,),
    )
    assert sos.verify()
    assert not verify_infeasibility_certificate(proof, x)


def test_linear_dual_source_binding():
    system = LinearSystem(1, ((sp.S.One,),), (sp.S.One,))
    proof = LinearProof("bound", (sp.S.One,), ())
    assert verify_linear_proof(system, (sp.S.One,), sp.S.One, proof)
    assert not verify_linear_proof(
        replace(system, lower=(sp.S.Zero,)), (sp.S.One,), sp.S.One, proof
    )
    for changes in [
        {"kind": "infeasible"},
        {"inequalities": (-sp.S.One,)},
        {"equalities": (0,)},
    ]:
        assert not verify_linear_proof(
            system, (sp.S.One,), sp.S.One, replace(proof, **changes)
        )


def test_face_source_and_coverage():
    objective = x * x - y * y
    constraints = [x >= -1, x <= 1, y >= -1, y <= 1]
    result = minimize_quadratic(
        objective, constraints, (x, y), backend="faces", certified=True
    )
    proof = result.certificate
    assert verify_quadratic_certificate(objective, proof, constraints, (x, y))
    assert not verify_quadratic_certificate(objective + 1, proof, constraints, (x, y))
    assert not verify_quadratic_certificate(
        objective, proof, [*constraints, y <= 0], (x, y)
    )
    # The claimed primal value and both bounds are changed consistently, but
    # the stationary-face evidence still proves the original value.
    changed = replace(proof, lower_bound=0, upper_bound=0, point=(0, 0))
    assert not verify_quadratic_certificate(objective, changed, constraints, (x, y))
    duplicate = replace(proof, faces=(proof.faces[0],) * len(proof.faces))
    assert not verify_quadratic_certificate(objective, duplicate, constraints, (x, y))


def test_spatial_coverage_and_bounds():
    constraints = [x >= 0, y >= 0, x + y <= 1]
    result = minimize_quadratic(
        -x * y, constraints, (x, y), backend="spatial", max_nodes=8, certified=True
    )
    proof = result.certificate
    assert verify_quadratic_certificate(-x * y, proof, constraints, (x, y))
    domain = replace(proof.domain, intervals=((0, sp.Rational(1, 4)), (0, 1)))
    root = replace(proof.nodes[0], incoming=domain.intervals, box=domain.intervals)
    narrowed = replace(proof, domain=domain, nodes=(root, *proof.nodes[1:]))
    assert not verify_quadratic_certificate(-x * y, narrowed, constraints, (x, y))
    root = proof.nodes[0]
    overlapping = replace(root, children=(root.children[0], root.children[0]))
    assert not verify_quadratic_certificate(
        -x * y,
        replace(proof, nodes=(overlapping, *proof.nodes[1:])),
        constraints,
        (x, y),
    )


def test_recession_ray_orientation():
    result = minimize_quadratic(-x, [x >= 0], x, certified=True)
    proof = result.certificate
    assert verify_quadratic_certificate(-x, proof, [x >= 0], x)
    assert not verify_quadratic_certificate(
        -x, replace(proof, direction=(-1,)), [x >= 0], x
    )
    assert not verify_quadratic_certificate(-x, proof, [x <= 0], x)


def test_nested_univariate_minimum():
    pytest.importorskip("funcprops")
    from symbopt import certify_global_minimum, verify_univariate_certificate

    proof = certify_global_minimum(x * x, x, sp.Interval(-1, 1))
    assert verify_univariate_certificate(proof, expr=x * x, var=x)
    component = proof.component_result
    candidate = replace(component.candidates[0], point=sp.S.One, value=sp.S.One)
    false_component = replace(
        component, infimum=sp.S.One, minimizers=sp.FiniteSet(1), candidates=(candidate,)
    )
    changed = replace(
        proof, point=sp.S.One, value=sp.S.One, component_result=false_component
    )
    assert not verify_univariate_certificate(changed, expr=x * x, var=x)
