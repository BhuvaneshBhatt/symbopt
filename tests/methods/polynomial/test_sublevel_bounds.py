from dataclasses import replace

import pytest
import sympy as sp

from symbopt import algebraize, polynomial_sublevel_bounds
from symbopt.sublevel_bounds import _sublevel_certificate, restrict_polynomial_sublevel

x, y = sp.symbols("x y", real=True)


@pytest.mark.parametrize(
    "objective",
    [
        x**4 + y**4 + x * y - 3 * x + 2 * y + 7,
        3 * x**4 + 3 * y**4 - 2 * x**3 * y + sp.sqrt(2) * x,
        x**4 + y**4 + 100 * x * x * y * y - 2 * x * y,
    ],
)
def test_exterior_bound(objective):
    level = sp.S(4)
    certificate = _sublevel_certificate(objective, (x, y), level)
    assert certificate.verify()
    bounds = polynomial_sublevel_bounds(objective, (x, y), level)
    radius = bounds[x][1]
    assert bounds == {x: (-radius, radius), y: (-radius, radius)}
    a = min(certificate.leading_margins)
    B = certificate.remainder_bound
    assert a * radius - B > abs(level)
    for point in ((radius, 0), (-radius, radius), (radius + 1, -radius - 1)):
        assert objective.subs(dict(zip((x, y), point))) > level


@pytest.mark.parametrize(
    "objective",
    [
        x**3 + y * y,
        (x - y) ** 4,
        x**4 + y * y,
        x**4 - y**4,
        (x * x + (x * y - 1) ** 2),
        x * x + sp.pi * y * y,
    ],
)
def test_unsupported_leading_form(objective):
    assert polynomial_sublevel_bounds(objective, (x, y), 0) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("degree", 2),
        ("radius", sp.S.Zero),
        ("upper_bound", sp.S(10) ** 8),
        ("remainder_bound", sp.S.Zero),
        ("leading_margins", (sp.S(100), sp.S(100))),
        ("objective", x**4 - y**4),
    ],
)
def test_certificate_mutation(field, value):
    certificate = _sublevel_certificate(x**4 + y**4 + x * y + 3, (x, y), 4)
    assert certificate.verify()
    assert replace(certificate, **{field: value}).verify() is False


def test_feasible_level_required():
    problem = algebraize(x**4 + y**4 + x * y, [x > 5, y > 5], variables=(x, y))
    bounds, proof = restrict_polynomial_sublevel(problem, {})
    assert bounds == {} and proof is None
    unconstrained = algebraize(x**4 + y**4 + x * y, [], variables=(x, y))
    bounds, proof = restrict_polynomial_sublevel(unconstrained, {})
    assert proof["upper_bound"] == 0 and proof["feasible_point"] == {x: 0, y: 0}
    assert all(left < 0 < right for left, right in bounds.values())


def test_no_infeasible_certificate():
    problem = algebraize(x**4 + y**4 + x * y, [x > 0, x < 0], variables=(x, y))
    assert restrict_polynomial_sublevel(problem, {}) == ({}, None)


def test_automatic_interval_box():
    from symbopt import minimize_result

    result = minimize_result(
        x**4 + y**4 + x * y,
        [y**4 + x >= -1],
        variables=(x, y),
        use_sos=False,
        max_active_sets=0,
        interval_max_boxes=4,
    )
    assert result.inferred_box_bounds == {x: (-4, 4), y: (-4, 4)}
    assert result.interval_result is not None
    proof = next(
        p
        for p in result.certificate.proof_objects
        if p.kind == "polynomial_sublevel_bounds"
    )
    assert proof.payload["upper_bound"] == 0
    assert proof.payload["feasible_point"] == {x: 0, y: 0}
    assert result.certified is False
