"""Exact quadratic optimization with real symbolic coefficients and affine bounds."""

import sympy as sp

from .relation_checks import normalize_relational
from .structural_results import StructuralGlobalResult
from .symbolic_cache import affine_coefficients, cached_poly, simplify_scalar


def _truth(condition):
    condition = sp.simplify(condition)
    if condition is sp.S.true:
        return True
    if condition is sp.S.false:
        return False
    return None


def _result(value, point=None, *, attained=True, **metadata):
    return StructuralGlobalResult(
        kind="parametric_quadratic",
        value=simplify_scalar(value),
        points=() if point is None else (point,),
        attained=attained,
        metadata=metadata,
    )


def _interval(constraints, variable):
    """Normalize affine scalar constraints while retaining strict endpoints."""
    lower, upper, strict, equalities = [], [], [], []
    for relation in constraints:
        kind, residual = normalize_relational(relation)
        data = affine_coefficients(residual, (variable,))
        if data is None:
            return None
        (slope,), constant = data
        if slope.is_real is not True or constant.is_real is not True:
            return None
        if slope == 0:
            if _truth(relation) is True:
                continue
            return None
        if slope.is_positive is not True and slope.is_negative is not True:
            return None
        endpoint = sp.simplify(-constant / slope)
        if kind == "eq":
            equalities.append(endpoint)
        elif kind in {"ge", "gt"}:
            (lower if slope.is_positive else upper).append(endpoint)
            if kind == "gt":
                strict.append(relation)
        else:
            return None
    left = sp.Max(*lower) if lower else -sp.oo
    right = sp.Min(*upper) if upper else sp.oo
    if equalities:
        point = equalities[0]
        if not all(_truth(sp.Eq(point, value)) is True for value in equalities):
            return None
        return point, point, strict
    return left, right, strict


def _scalar_quadratic(poly, constraints, variable):
    """Minimize a real quadratic on a proved interval without specializing parameters."""
    if any(coefficient.is_real is not True for coefficient in poly.coeffs()):
        return None
    interval = _interval(constraints, variable)
    if interval is None:
        return None
    left, right, strict = interval
    if _truth(left <= right) is not True:
        if _truth(left > right) is True:
            return _result(sp.oo, attained=False, infeasible=True)
        return None
    a, b = (sp.simplify(poly.nth(i)) for i in (2, 1))
    expression = poly.as_expr()
    if left == right:
        feasible = [_truth(rel.subs(variable, left)) for rel in constraints]
        if all(value is True for value in feasible):
            return _result(
                expression.subs(variable, left), {variable: left}, proof="singleton"
            )
        if any(value is False for value in feasible):
            return _result(sp.oo, attained=False, infeasible=True)
        return None
    if strict and _truth(left < right) is not True:
        # A potentially collapsed interval can be empty at an open endpoint.
        return None

    finite = left is not -sp.oo and right is not sp.oo
    if a == 0:
        if b == 0:
            # An interior representative also works for open intervals.
            point = (
                (left + right) / 2
                if finite
                else left + 1
                if left is not -sp.oo
                else right - 1
                if right is not sp.oo
                else sp.S.Zero
            )
        elif b.is_positive is True:
            point = left
        elif b.is_negative is True:
            point = right
        elif finite and not strict:
            point = sp.Piecewise((left, b >= 0), (right, True))
        else:
            return None
    elif a.is_nonnegative is True:
        vertex = sp.simplify(-b / (2 * a))
        if vertex.is_real is not True or sp.simplify(2 * a * vertex + b) != 0:
            return None
        point = sp.Min(sp.Max(vertex, left), right)
    elif a.is_negative is True and not finite:
        return _result(-sp.oo, attained=False, proof="negative curvature")
    elif finite and not strict:
        fleft, fright = (
            expression.subs(variable, left),
            expression.subs(variable, right),
        )
        endpoint = sp.Piecewise((left, fleft <= fright), (right, True))
        if a.is_nonpositive is True:
            point = endpoint
        else:
            # The vertex is evaluated only in the positive-curvature branch.
            vertex = sp.Min(sp.Max(-b / (2 * a), left), right)
            point = sp.Piecewise((vertex, a > 0), (endpoint, True))
    else:
        return None

    if point in (sp.oo, -sp.oo):
        return _result(-sp.oo, attained=False, proof="nonzero affine slope")
    value = simplify_scalar(expression.subs(variable, point))
    checks = [_truth(relation.subs(variable, point)) for relation in strict]
    attained = False if False in checks else None if None in checks else True
    return _result(
        value,
        {variable: point} if attained is True else None,
        attained=attained,
        proof="quadratic interval minimum",
        interval=(left, right),
    )


def try_parametric_quadratic(objective, constraints, variables):
    """Solve real parameterized quadratics with proved curvature or scalar affine bounds.

    Unconstrained positive-semidefinite forms require a real stationary point
    valid for every allowed parameter value. Scalar closed bounded problems
    also allow conditional curvature. Undecidable feasibility or denominators
    leave the problem to another route.
    """
    variables = tuple(variables)
    parameters = objective.free_symbols - set(variables)
    for relation in constraints:
        parameters.update(relation.free_symbols - set(variables))
    if not parameters or not variables:
        return None
    try:
        poly = cached_poly(objective, variables)
    except (sp.PolynomialError, TypeError, ValueError):
        return None
    if poly.total_degree() > 2 or any(c.is_real is not True for c in poly.coeffs()):
        return None
    if len(variables) == 1 and constraints:
        return _scalar_quadratic(poly, constraints, variables[0])
    if constraints:
        return None

    hessian = sp.hessian(poly.as_expr(), variables)
    if any(hessian[i, i].is_negative is True for i in range(len(variables))):
        return _result(-sp.oo, attained=False, proof="negative curvature")
    convex = hessian.is_positive_semidefinite
    if convex is False:
        return _result(
            -sp.oo, attained=False, proof="non-positive-semidefinite quadratic"
        )
    if convex is not True:
        return None
    origin = dict.fromkeys(variables, sp.S.Zero)
    gradient = sp.Matrix([sp.diff(poly.as_expr(), v) for v in variables])
    linear = gradient.subs(origin)
    if hessian.is_zero_matrix:
        if all(value == 0 for value in linear):
            return _result(poly.as_expr(), {}, proof="constant quadratic")
        if any(value.is_zero is False for value in linear):
            return _result(-sp.oo, attained=False, proof="nonzero affine slope")
        return None
    solutions = sp.linsolve((hessian, -linear), variables)
    if solutions is sp.S.EmptySet:
        for direction in hessian.nullspace():
            slope = sp.simplify((linear.T * direction)[0])
            if (
                all(value.is_real is True for value in direction)
                and slope.is_real is True
                and slope.is_zero is False
            ):
                return _result(
                    -sp.oo, attained=False, proof="nonzero slope on a null direction"
                )
        return None
    if not isinstance(solutions, sp.FiniteSet) or len(solutions) != 1:
        return None
    solution = next(iter(solutions))
    point = {
        v: sp.simplify(value.subs(origin)) for v, value in zip(variables, solution)
    }
    if any(value.is_real is not True for value in point.values()):
        return None
    if any(
        sp.simplify(value.subs(point, simultaneous=True)) != 0 for value in gradient
    ):
        return None
    return _result(
        poly.as_expr().subs(point, simultaneous=True),
        point,
        proof="positive-semidefinite stationary quadratic",
        hessian=hessian,
    )
