"""Exact endpoint and vertex rules for Cartesian interval domains."""

from functools import cmp_to_key

import sympy as sp

from .rational_optimization import _compare
from .structural_results import StructuralGlobalResult


def _best_corners(context, kind, corners, **proof):
    values = [
        (sp.simplify(context.objective.subs(p, simultaneous=True)), p) for p in corners
    ]
    if any(v.is_real is not True or v.is_finite is not True for v, _ in values):
        return None
    try:
        best = min((v for v, _ in values), key=cmp_to_key(_compare))
    except (NotImplementedError, sp.PolynomialError, ValueError):
        return None
    points = []
    for value, point in values:
        if _compare(value, best) == 0:
            result = context.result(kind, point, **proof)
            if result is not None:
                points.append(result.points[0])
    if not points:
        for point in context.seeds():
            result = context.result(kind, point, **proof)
            if result is not None and _compare(result.value, best) == 0:
                points.append(result.points[0])
                break
    attained = True if points else None
    # With a strict monotone coordinate every excluded extreme is unattainable.
    if not points and proof.get("strict_excluded_coordinate"):
        attained = False
    return StructuralGlobalResult(kind, best, tuple(points), attained, metadata=proof)


def try_multiaffine_box(context):
    """Evaluate small-box vertices, requiring denominator sign throughout closure."""
    corners = context.corners()
    if not corners:
        return None
    p, q = sp.fraction(sp.cancel(context.objective))
    polys = [sp.Poly(e, *context.variables) for e in (p, q)]
    if any(poly.degree(v) > 1 for poly in polys for v in context.variables):
        return None
    denominators = [sp.simplify(q.subs(point, simultaneous=True)) for point in corners]
    if not (
        all(v.is_positive is True for v in denominators)
        or all(v.is_negative is True for v in denominators)
    ):
        return None
    # Any guard not implied by the rectangle causes rectangle recognition to fail.
    strict_scalar = False
    if len(context.variables) == 1:
        numerator, denominator = polys
        determinant = numerator.nth(1) * denominator.nth(0) - numerator.nth(
            0
        ) * denominator.nth(1)
        # A nonconstant linear-fractional scalar map has derivative det/q**2.
        # With no pole, an excluded minimizing endpoint cannot be attained inside.
        strict_scalar = determinant.is_zero is False
    return _best_corners(
        context,
        "multiaffine_box",
        corners,
        denominator_values=tuple(denominators),
        vertex_count=len(corners),
        domain=context.rectangle,
        strict_excluded_coordinate=strict_scalar,
    )


def try_monotone_box(context):
    """Fix coordinates whose derivative sign is proved on a finite product box."""
    if not context.rectangle or not all(i.finite for i in context.rectangle.values()):
        return None
    if any(
        node.func.is_Function
        and node.func not in {sp.exp, sp.log, sp.Abs, sp.sin, sp.cos, sp.sinh, sp.cosh}
        for node in sp.preorder_traversal(context.objective)
    ):
        return None
    point = {}
    derivatives = {}
    strict_excluded = False
    for variable, interval in context.rectangle.items():
        derivative = sp.diff(context.objective, variable)
        sign = context.sign(derivative)
        if sign is None:
            return None
        derivatives[variable] = derivative
        if sign == 0:
            point[variable] = interval.sample()
        elif sign > 0:
            point[variable] = interval.left
            strict_excluded |= interval.left_open and sign == 1
        else:
            point[variable] = interval.right
            strict_excluded |= interval.right_open and sign == -1
    # Endpoint substitution is used only for expressions continuous on closure.
    values = [context.objective.subs(point, simultaneous=True)]
    if any(v.is_real is not True or v.is_finite is not True for v in values):
        return None
    # Exclude poles inside the box, even when endpoint evaluations are finite.
    _p, q = sp.fraction(sp.cancel(context.objective))
    if q != 1 and context.sign(q) not in {1, -1}:
        poly = sp.Poly(q, *context.variables)
        corners = context.corners()
        if not corners or any(poly.degree(v) > 1 for v in context.variables):
            return None
        q_values = [q.subs(c, simultaneous=True) for c in corners]
        if not (
            all(v.is_positive is True for v in q_values)
            or all(v.is_negative is True for v in q_values)
        ):
            return None
    return _best_corners(
        context,
        "coordinate_monotone",
        (point,),
        derivatives=derivatives,
        strict_excluded_coordinate=strict_excluded,
        domain=context.rectangle,
    )
