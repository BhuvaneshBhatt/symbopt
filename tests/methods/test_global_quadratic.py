"""Global values, partial bounds, domain semantics and certificate replay."""

from dataclasses import fields, replace

import pytest
import sympy as sp

from symbopt import (
    minimize_quadratic,
    minimize_result,
    scip_available,
    solve_global_qp,
    verify_quadratic_certificate,
)
from symbopt.quadratic_faces import FaceCertificate
from symbopt.quadratic_geometry import EqualityCertificate
from symbopt.quadratic_lp import LinearProof
from symbopt.quadratic_spatial import SpatialCertificate, SpatialDomain, SpatialNode

x, y = sp.symbols("x y", real=True)
BOX = [x >= -1, x <= 1, y >= -1, y <= 1]


def replay(objective, result, constraints=BOX):
    return verify_quadratic_certificate(
        objective, result.certificate, constraints, (x, y)
    )


@pytest.mark.parametrize(
    "objective,constraints,value",
    [
        (x * x - y * y, BOX, -1),
        (x * x - y * y, [x >= 0, y >= 0, x + y <= 1], -1),
        ((x - y) ** 2 - y * y, [x >= 0, x <= 1, y >= 0, y <= 1], -1),
        (-x * x, [sp.Eq(x + y, 1), x >= 0, y >= 0], -1),
        (x * y, [sp.Eq(x, 2), sp.Eq(y, 3)], 6),
        (x * x + y * y, [sp.Eq(x + y, 1), sp.Eq(2 * x + 2 * y, 2)], sp.Rational(1, 2)),
        (x * x, [x >= -1, x <= 1, sp.Eq(y, 0), sp.Eq(2 * y, 0)], 0),
    ],
)
def test_faces(objective, constraints, value):
    result = minimize_quadratic(
        objective, constraints, (x, y), backend="faces", certified=True
    )
    assert (
        result.status,
        result.objective_value,
        result.lower_bound,
        result.upper_bound,
    ) == ("optimal", value, value, value)
    assert result.exact and result.certified and result.bounds_verified
    assert replay(objective, result, constraints)
    assert objective.subs(result.point_mapping) == value
    assert all(r.subs(result.point_mapping) is sp.true for r in constraints)


@pytest.mark.parametrize(
    "objective,constraints,value",
    [
        (x * x - y * y, [], -sp.oo),
        (x + (x - y) ** 2, [], -sp.oo),
        (x + (x - 2 * y) ** 2, [], -sp.oo),
        (x * x - y * y, [sp.Eq(y, 0)], 0),
        (x * x - 4 * x * y + y * y, [x >= 0, y >= 0], -sp.oo),
        (x * x + y * y, [sp.Eq(x + y, 2)], 2),
    ],
)
def test_equalities_and_rays(objective, constraints, value):
    result = minimize_quadratic(objective, constraints, (x, y), certified=True)
    assert result.objective_value == value
    assert result.exact and result.certified
    assert replay(objective, result, constraints)
    assert result.status == ("unbounded" if value == -sp.oo else "optimal")


@pytest.mark.parametrize("backend", ["faces", "auto"])
def test_infeasible(backend):
    constraints = [x >= 1, x <= 0]
    result = minimize_quadratic(
        x * y, constraints, (x, y), backend=backend, certified=True
    )
    assert result.status == "infeasible"
    assert result.lower_bound == result.upper_bound == sp.oo
    assert replay(x * y, result, constraints)


@pytest.mark.parametrize(
    "backend,budget", [("faces", {"max_faces": 0}), ("spatial", {"max_nodes": 0})]
)
def test_limits(backend, budget):
    result = minimize_quadratic(
        x * x - y * y, BOX, (x, y), backend=backend, certified=True, **budget
    )
    assert result.status in {"face_limit", "node_limit", "optimal"}
    assert result.lower_bound <= -1 <= result.upper_bound
    assert result.certified and replay(x * x - y * y, result)
    assert result.exact is (result.gap == 0)


@pytest.mark.parametrize(
    "objective,constraints,value",
    [
        (-x * y, [x >= 0, y >= 0, x + y <= 1], -sp.Rational(1, 4)),
        (x * x + y * y, [x * x + y * y >= 1], 1),
        (x * x + y * y, [sp.Eq(x * x + y * y, 3)], 3),
        (x * x - y * y, BOX, -1),
    ],
)
def test_spatial(objective, constraints, value):
    result = minimize_quadratic(
        objective, constraints, (x, y), backend="spatial", max_nodes=16, certified=True
    )
    assert result.status == "optimal"
    assert result.objective_value == result.lower_bound == result.upper_bound == value
    assert result.exact and replay(objective, result, constraints)


def test_positive_gap():
    constraints = [x * x + y * y <= 1]
    result = minimize_quadratic(
        -x * y, constraints, (x, y), backend="spatial", max_nodes=8, certified=True
    )
    assert result.lower_bound <= -sp.Rational(1, 2) <= result.upper_bound < 0
    assert result.status == "node_limit" and not result.exact
    assert result.certified and replay(-x * y, result, constraints)


def test_infeasible_spatial():
    constraints = [*BOX, sp.Le(x * x + y * y, -1, evaluate=False)]
    result = minimize_quadratic(
        x * y, constraints, (x, y), backend="spatial", certified=True
    )
    assert result.status == "infeasible"
    assert result.lower_bound == result.upper_bound == sp.oo
    assert result.certified and replay(x * y, result, constraints)


def test_matrix_api():
    result = solve_global_qp(
        [[2, 0], [0, -2]], [0, 0], bounds=[(-1, 1), (-1, 1)], certified=True
    )
    assert result.objective_value == -1 and result.exact
    assert result.point == (0, -1)
    result = solve_global_qp(
        [[0, 0], [0, 0]], [1, 0], A_eq=[[1, 0]], b_eq=[-2], certified=True
    )
    assert result.objective_value == -2 and result.point[0] == -2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backend": "bad"},
        {"max_faces": True},
        {"max_nodes": -1},
        {"time_limit": 0},
        {"time_limit": float("inf")},
        {"absolute_gap": -1},
        {"relative_gap": sp.oo},
        {"certified": 1},
        {"initial_point": (2, 2)},
    ],
)
def test_invalid_controls(kwargs):
    with pytest.raises((TypeError, ValueError)):
        minimize_quadratic(x * y, BOX, (x, y), **kwargs)


@pytest.mark.parametrize(
    "Q,c,kwargs",
    [
        ([[1, 2]], [0], {}),
        ([[1]], [1, 2], {}),
        ([[1]], [0], {"A_ub": [[1]]}),
        ([[1]], [0], {"A_eq": [[1, 2]], "b_eq": [0]}),
    ],
)
def test_matrix_shapes(Q, c, kwargs):
    with pytest.raises(ValueError):
        solve_global_qp(Q, c, **kwargs)


def test_domain_semantics():
    result = minimize_quadratic(
        x * x - y * y, variables=(x, y), bounds=[(-1, 1), (-1, 1)]
    )
    assert result.objective_value == -1 and result.exact and not result.certified
    result = minimize_quadratic(x * x - y * y, [x > 0], (x, y), backend="faces")
    assert result.status == "unsupported" and not result.bounds_verified
    result = minimize_quadratic(-x * y, [x * x - y * y >= 1], (x, y), backend="spatial")
    assert result.status == "unsupported" and result.lower_bound == -sp.oo
    result = minimize_result(
        x * x - y * y,
        variables=(x, y),
        box_bounds={x: (-1, 1), y: (-1, 1)},
        options={"quadratic": {"backend": "faces", "certified": True}},
    )
    assert result.unbounded and result.optimum_value == -sp.oo


def test_pipeline_scope():
    exact = minimize_result(
        x * x - y * y,
        BOX,
        variables=(x, y),
        options={"quadratic": {"backend": "faces", "certified": True}},
    )
    assert exact.optimum_value == -1 and exact.certified and exact.attained
    candidate = minimize_result(
        x * x - y * y,
        BOX,
        variables=(x, y),
        options={"quadratic": {"backend": "faces"}},
    )
    assert candidate.optimum_value is None and not candidate.certified
    assert candidate.best_candidate.objective_value == -1
    limited = minimize_result(
        -x * y,
        [x * x + y * y <= 1],
        variables=(x, y),
        options={
            "quadratic": {"backend": "spatial", "max_nodes": 2, "certified": True}
        },
    )
    assert not limited.certified and limited.optimum_value is None
    assert (
        limited.certificate.lower_bound
        <= -sp.Rational(1, 2)
        <= limited.certificate.upper_bound
    )


def test_integer_routing():
    n = sp.Symbol("n", integer=True)
    with pytest.raises(NotImplementedError, match="backend='scip'"):
        minimize_quadratic(n * n, variables=(n,))
    with pytest.raises(NotImplementedError):
        minimize_result(n * n, variables=(n,))


@pytest.fixture(scope="module")
def face_result():
    return minimize_quadratic(
        x * x - y * y, BOX, (x, y), backend="faces", certified=True
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("problem", None),
        ("kind", "bad"),
        ("lower_bound", -2),
        ("upper_bound", 0),
        ("point", (0, 0)),
        ("bounds", None),
        ("faces", ()),
    ],
)
def test_face_mutations(face_result, field, value):
    assert not verify_quadratic_certificate(
        x * x - y * y, replace(face_result.certificate, **{field: value}), BOX, (x, y)
    )


def test_dual_mutation(face_result):
    proof = face_result.certificate
    bounds = proof.bounds
    weights = bounds.proofs[0]
    mutated = replace(weights, inequalities=tuple(-1 for _ in weights.inequalities))
    altered = replace(bounds, proofs=(mutated, *bounds.proofs[1:]))
    assert not verify_quadratic_certificate(
        x * x - y * y, replace(proof, bounds=altered), BOX, (x, y)
    )
    candidate = next(r for r in proof.faces if r.kind == "candidate")
    altered = replace(
        candidate, solution=tuple(sp.Float(v) for v in candidate.solution)
    )
    records = tuple(altered if r is candidate else r for r in proof.faces)
    assert not verify_quadratic_certificate(
        x * x - y * y, replace(proof, faces=records), BOX, (x, y)
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", "bad"),
        ("anchor", (sp.Float(0), sp.Float(0))),
        ("direction", (0, 0)),
        ("problem", None),
    ],
)
def test_ray_mutations(field, value):
    result = minimize_quadratic(x * x - y * y, variables=(x, y), certified=True)
    assert not verify_quadratic_certificate(
        x * x - y * y, replace(result.certificate, **{field: value}), variables=(x, y)
    )


@pytest.fixture(scope="module")
def spatial_result():
    return minimize_quadratic(
        -x * y,
        [x >= 0, y >= 0, x + y <= 1],
        (x, y),
        backend="spatial",
        max_nodes=8,
        certified=True,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("problem", None),
        ("domain", None),
        ("nodes", ()),
        ("lower_bound", -1),
        ("upper_bound", 0),
        ("point", (0, 0)),
    ],
)
def test_spatial_mutations(spatial_result, field, value):
    assert not verify_quadratic_certificate(
        -x * y,
        replace(spatial_result.certificate, **{field: value}),
        [x >= 0, y >= 0, x + y <= 1],
        (x, y),
    )


def test_partition_mutations(spatial_result):
    proof = spatial_result.certificate
    root = proof.nodes[0]
    assert root.children
    for altered in [
        replace(root, children=(0, 0)),
        replace(root, midpoint=10),
        replace(root, lower=1),
        replace(root, incoming=((0, 2), (0, 1))),
    ]:
        changed = replace(proof, nodes=(altered, *proof.nodes[1:]))
        assert not verify_quadratic_certificate(
            -x * y, changed, [x >= 0, y >= 0, x + y <= 1], (x, y)
        )


def test_certificate_field_inventory():
    mathematical = {
        FaceCertificate: {
            "problem",
            "kind",
            "lower_bound",
            "upper_bound",
            "point",
            "bounds",
            "faces",
            "complete",
            "infeasibility",
        },
        EqualityCertificate: {"problem", "kind", "anchor", "direction"},
        SpatialCertificate: {
            "problem",
            "domain",
            "nodes",
            "lower_bound",
            "upper_bound",
            "point",
        },
        SpatialDomain: {"kind", "intervals", "linear_bounds", "relation", "witness"},
        SpatialNode: {
            "incoming",
            "box",
            "lower",
            "kind",
            "proof",
            "children",
            "split",
            "midpoint",
        },
        LinearProof: {"kind", "inequalities", "equalities"},
    }
    # No mathematical field may be added without an explicit mutation/replay decision.
    for kind, names in mathematical.items():
        assert {f.name for f in fields(kind)} == names


@pytest.mark.skipif(not scip_available(), reason="optional PySCIPOpt is not installed")
@pytest.mark.parametrize("integer", [False, True])
def test_scip(integer):
    u = sp.Symbol("u", integer=True) if integer else x
    result = minimize_quadratic(
        (u - sp.Rational(1, 3)) ** 2 - y * y,
        [u >= 0, u <= 1, y >= -1, y <= 1],
        (u, y),
        backend="scip",
        certified=True,
    )
    expected = -sp.Rational(8, 9) if integer else -1
    assert abs(float(result.objective_value - expected)) < 1e-6
    assert result.lower_bound <= float(expected) + 1e-6
    assert not result.certified and not result.exact and not result.bounds_verified
    assert not verify_quadratic_certificate(
        (u - sp.Rational(1, 3)) ** 2 - y * y, result.certificate, variables=(u, y)
    )


@pytest.mark.skipif(not scip_available(), reason="optional PySCIPOpt is not installed")
def test_scip_free_variables():
    result = minimize_quadratic((x + 2) ** 2 + y * y, variables=(x, y), backend="scip")
    assert abs(float(result.point[0] + 2)) < 1e-5
    result = minimize_quadratic(-x * y, [x * x + y * y <= 1], (x, y), backend="scip")
    assert abs(float(result.objective_value + sp.Rational(1, 2))) < 1e-6


def test_valid_certificate_variations(face_result):
    # Removing the completeness claim remains sound when interval bounds already meet.
    assert replay(
        x * x - y * y,
        replace(
            face_result, certificate=replace(face_result.certificate, complete=False)
        ),
    )
    result = minimize_quadratic(x * x - y * y, variables=(x, y), certified=True)
    assert verify_quadratic_certificate(
        x * x - y * y, replace(result.certificate, anchor=(1, 1)), variables=(x, y)
    )


def test_farkas_kind():
    constraints = [x >= 1, x <= 0]
    result = minimize_quadratic(x * y, constraints, (x, y), certified=True)
    original = result.certificate
    fake = LinearProof("bound", (0, 0), ())
    assert not verify_quadratic_certificate(
        x * y, replace(original, infeasibility=fake), constraints, (x, y)
    )


@pytest.mark.parametrize(
    "sense,objective,value",
    [
        ("min", x, -1),
        ("max", x, 1),
        ("max", x * x, 1),
    ],
)
def test_open_ellipsoid(sense, objective, value):
    from symbopt import maximize_result

    operation = minimize_result if sense == "min" else maximize_result
    result = operation(objective, [x * x + y * y < 1], variables=(x, y), use_sos=False)
    assert result.optimum_value == value and result.certified
    assert result.attained is False and result.best_candidate is None


def test_exact_real_coefficients():
    result = minimize_result(
        sp.E * x * x - sp.pi * y,
        [x * x + sp.sqrt(2) * y * y <= sp.pi],
        variables=(x, y),
        use_sos=False,
    )
    assert (
        sp.simplify(
            result.optimum_value + sp.pi ** sp.Rational(3, 2) / 2 ** sp.Rational(1, 4)
        )
        == 0
    )
    assert result.certified and result.attained
    assert result.best_candidate.original_variable_values[x] == 0


@pytest.mark.parametrize("coefficient", [-7, 1, sp.sqrt(2)])
def test_tangent_poles(coefficient):
    result = minimize_result(
        coefficient * sp.tan(2 * x + 1) + sp.sin(3 * x) + 2 * sp.cos(x),
        variables=x,
        use_sos=False,
    )
    assert (
        result.unbounded and result.optimum_value == -sp.oo and result.attained is False
    )


@pytest.mark.parametrize(
    "constraints", [[x > 0, x <= 0], [x + y > 2, x + y <= 2], [x > 1, x < 0]]
)
def test_strict_affine_infeasible(constraints):
    result = minimize_result(
        1 + y / (1 + x + y), constraints, variables=(x, y), use_sos=False
    )
    assert result.infeasible and result.certified and result.optimum_value == sp.oo


def test_strict_affine_feasible():
    result = minimize_result(x * x, [x > 0, x < 1], variables=x, use_sos=False)
    assert result.optimum_value == 0 and result.attained is False


def test_spatial_domain_mutations():
    result = minimize_quadratic(
        x * x + y * y, [x * x + y * y >= 1], (x, y), backend="spatial", certified=True
    )
    proof = result.certificate
    assert proof.domain.kind == "sublevel"
    for domain in [
        replace(proof.domain, kind="bad"),
        replace(proof.domain, intervals=((-2, 2), (-1, 1))),
        replace(proof.domain, witness=(0, 0)),
    ]:
        assert not verify_quadratic_certificate(
            x * x + y * y, replace(proof, domain=domain), [x * x + y * y >= 1], (x, y)
        )
    result = minimize_quadratic(
        -x * y,
        [x * x + y * y <= 1],
        (x, y),
        backend="spatial",
        max_nodes=0,
        certified=True,
    )
    proof = result.certificate
    assert proof.domain.kind == "ellipsoid"
    domain = replace(proof.domain, relation=10)
    assert not verify_quadratic_certificate(
        -x * y, replace(proof, domain=domain), [x * x + y * y <= 1], (x, y)
    )


def test_face_record_mutations(face_result):
    proof = face_result.certificate
    index = next(i for i, r in enumerate(proof.faces) if r.kind == "candidate")
    record = proof.faces[index]
    for changed in [
        replace(record, active=(100,)),
        replace(record, value=-100),
        replace(record, kind="bad"),
    ]:
        records = list(proof.faces)
        records[index] = changed
        assert not verify_quadratic_certificate(
            x * x - y * y, replace(proof, faces=tuple(records)), BOX, (x, y)
        )


@pytest.mark.skipif(not scip_available(), reason="optional PySCIPOpt is not installed")
@pytest.mark.parametrize(
    "assumptions,target", [({"even": True}, 0), ({"odd": True}, 1)]
)
def test_scip_parity(assumptions, target):
    n = sp.Symbol("n", **assumptions)
    result = minimize_quadratic(
        (n - sp.Rational(2, 3)) ** 2, variables=n, backend="scip"
    )
    assert abs(float(result.point[0] - target)) < 1e-6
    assert not result.certified
    result = minimize_result(
        (n - sp.Rational(2, 3)) ** 2,
        variables=n,
        options={"quadratic": {"backend": "scip"}},
    )
    assert not result.certified and result.optimum_value is None
    assert (
        abs(
            float(
                result.best_candidate.objective_value
                - (sp.Rational(2, 3) - target) ** 2
            )
        )
        < 1e-6
    )


@pytest.mark.skipif(not scip_available(), reason="optional PySCIPOpt is not installed")
def test_strict_integer_quadratic():
    n, m = sp.symbols("n m", integer=True)
    result = minimize_quadratic(
        n + m, [n * n + m * m < 100], (n, m), backend="scip", max_nodes=2048
    )
    assert abs(float(result.objective_value + 14)) < 1e-6
    assert result.status == "optimal" and not result.exact
    assert float(sum(v * v for v in result.point)) < 100
    n = sp.Symbol("n", integer=True, positive=True)
    result = minimize_quadratic(n * n, variables=n, backend="scip")
    assert abs(float(result.point[0] - 1)) < 1e-6


@pytest.mark.skipif(not scip_available(), reason="optional PySCIPOpt is not installed")
def test_continuous_strict_refusal():
    result = minimize_quadratic(x * x, [x > 0], variables=x, backend="scip")
    assert result.status == "unsupported" and not result.certified
