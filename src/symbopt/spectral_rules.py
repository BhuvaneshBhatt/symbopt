"""Exact quadratic, Rayleigh-quotient and ellipsoid optimization rules."""

import sympy as sp

from .rational_optimization import _compare, _real_roots
from .relation_checks import normalize_relational
from .rule_utils import (
    RuleContext,
    affine_data,
    linear_solution,
    polynomial_degree_bound,
)
from .structural_results import StructuralGlobalResult
from .symbolic_cache import (
    cached_gradient,
    cached_hessian,
    cached_poly,
    structural_coordinate,
)


def quadratic_data(expression, variables):
    """Return Q,b,c with f=x.T*Q*x+b.T*x+c for a small real quadratic."""
    if len(variables) > 8 or sp.count_ops(expression) > 100:
        return None
    degree = polynomial_degree_bound(expression, variables)
    if degree is None or degree > 2:
        return None
    poly = cached_poly(expression, tuple(variables))
    if poly.total_degree() > 2:
        return None
    zero = dict.fromkeys(variables, sp.S.Zero)
    matrix = sp.ImmutableMatrix(cached_hessian(expression, tuple(variables))) / 2
    linear = sp.Matrix(
        [d.subs(zero) for d in cached_gradient(expression, tuple(variables))]
    )
    constant = expression.subs(zero)
    if any(value.is_real is not True for value in (*matrix, *linear, constant)):
        return None
    return matrix, linear, constant


def try_exact_quadratic(context):
    """Solve an unconstrained quadratic by exact linear algebra and PSD checks."""
    if context.constraints:
        return None
    data = quadratic_data(context.objective, context.variables)
    if data is None:
        return None
    matrix, linear, _constant = data
    if matrix.is_positive_semidefinite is not True:
        return None
    equations = tuple(2 * matrix * sp.Matrix(context.variables) + linear)
    point = linear_solution(equations, context.variables)
    if point is None:
        return None
    return context.result(
        "exact_quadratic",
        point,
        hessian=2 * matrix,
        stationarity_residuals=tuple(sp.simplify(e.subs(point)) for e in equations),
        psd=True,
        method="exact_linear_stationarity",
    )


def _ellipsoid(relation, variables):
    kind, residual = normalize_relational(relation)
    if kind not in {"ge", "gt", "eq"}:
        return None
    for sign in (1, -1) if kind == "eq" else (1,):
        data = quadratic_data(sign * residual, variables)
        if data is None:
            continue
        negative, linear, constant = data
        matrix = -negative
        if matrix.is_positive_definite is not True:
            continue
        center = matrix.inv() * linear / 2
        level = sp.simplify(constant + (center.T * matrix * center)[0])
        if level.is_nonnegative is True and all(v.is_real is True for v in center):
            return matrix, center, level, kind
    return None


def try_linear_ellipsoid(context):
    """Apply the exact support-function formula of a positive-definite ellipsoid."""
    data = affine_data(context.objective, context.variables)
    if data is None:
        return None
    coefficients, constant = data
    a = sp.Matrix(coefficients)
    if any(v.is_real is not True for v in (*a, constant)):
        return None
    for relation in context.constraints:
        ellipsoid = _ellipsoid(relation, context.variables)
        if ellipsoid is None:
            continue
        matrix, center, level, kind = ellipsoid
        inverse_a = matrix.inv() * a
        dual_norm = sp.simplify((a.T * inverse_a)[0])
        if kind == "gt" and level == 0 and len(context.constraints) == 1:
            return StructuralGlobalResult(
                "linear_ellipsoid",
                sp.oo,
                metadata={"infeasible": True, "positive_definite": matrix},
            )
        if dual_norm == 0:
            vector = center.copy()
            if kind == "eq" and level.is_positive is True:
                vector[0] += sp.sqrt(level / matrix[0, 0])
        elif dual_norm.is_positive is True:
            vector = center - sp.sqrt(level / dual_norm) * inverse_a
        else:
            continue
        point = dict(zip(context.variables, vector))
        result = context.result(
            "linear_ellipsoid", point, matrix=matrix, center=center, level=level
        )
        if result is not None:
            return result
        if (
            len(context.constraints) == 1
            and kind == "gt"
            and level.is_positive is True
            and dual_norm.is_positive is True
        ):
            value = sp.simplify(
                constant + (a.T * center)[0] - sp.sqrt(level * dual_norm)
            )
            return StructuralGlobalResult(
                "linear_ellipsoid",
                value,
                attained=False,
                metadata={
                    "matrix": matrix,
                    "center": center,
                    "level": level,
                    "limiting_point": point,
                    "approach": "center + t*(limiting_point-center), t -> 1 from below",
                },
            )
    return None


def try_rayleigh(context):
    """Use a smallest generalized eigenvalue, checking any constrained witness."""
    variables = context.variables
    if not 2 <= len(variables) <= 6:
        return None
    p, q = sp.fraction(sp.cancel(context.objective))
    numerator = quadratic_data(p, variables)
    denominator = quadratic_data(q, variables)
    if numerator is None:
        return None
    a, linear, offset = numerator
    if any(v != 0 for v in linear):
        return None
    scale = sp.S.One
    if (
        denominator is not None
        and denominator[0].is_positive_definite is True
        and all(v == 0 for v in denominator[1])
        and denominator[2] == 0
        and offset == 0
    ):
        b = denominator[0]
    else:
        if q != 1:
            return None
        b = None
        for relation in context.constraints:
            if not isinstance(relation, sp.Equality):
                continue
            ellipsoid = _ellipsoid(relation, variables)
            if ellipsoid is None:
                continue
            matrix, center, level, _ = ellipsoid
            if all(v == 0 for v in center) and level.is_positive is True:
                b, scale = matrix, level
                break
        if b is None:
            return None
    if any(value.free_symbols for value in (*a, *b)):
        return None
    parameter = structural_coordinate(
        "eigenvalue", (context.objective, context.constraints, context.variables)
    )
    determinant = (a - parameter * b).det()
    roots = _real_roots(determinant, parameter)
    if not roots:
        return None
    value = roots[0]
    vectors = (a - value * b).nullspace()
    for vector in vectors:
        norm = sp.simplify((vector.T * b * vector)[0])
        if norm.is_positive is not True:
            continue
        candidates = [
            vector,
            sp.sqrt(scale / norm) * vector,
            -sp.sqrt(scale / norm) * vector,
        ]
        for candidate in candidates:
            point = dict(zip(variables, candidate))
            result = context.result(
                "rayleigh_quotient", point, matrix=a, metric=b, eigenvalue=value
            )
            expected = scale * value + offset if q == 1 else value
            if result is not None and _compare(result.value, expected) == 0:
                return result
    return None


def try_quadratic_ellipsoid(context):
    """Match a PSD quadratic correction that vanishes at a linear support point.

    The affine support value bounds the objective below because the remaining
    quadratic is nonnegative everywhere. Equality at the support point proves
    the global value even with exact nonalgebraic coefficients.
    """
    data = quadratic_data(context.objective, context.variables)
    if data is None:
        return None
    matrix, linear, constant = data
    if matrix.is_zero_matrix or matrix.is_positive_semidefinite is not True:
        return None
    affine = (linear.T * sp.Matrix(context.variables))[0] + constant
    support = try_linear_ellipsoid(
        RuleContext(affine, context.constraints, context.variables)
    )
    if support is None:
        return None
    point = (
        support.points[0] if support.points else support.metadata.get("limiting_point")
    )
    if point is None:
        return None
    vector = sp.Matrix([point[v] for v in context.variables])
    if any(sp.simplify(v) != 0 for v in matrix * vector):
        return None
    return StructuralGlobalResult(
        "quadratic_ellipsoid",
        support.value,
        support.points,
        support.attained,
        metadata={**support.metadata, "psd_correction": matrix},
    )


def try_centered_ellipsoid(context):
    """Optimize a centered homogeneous quadratic on an open or closed ellipsoid."""
    if len(context.constraints) != 1 or not 2 <= len(context.variables) <= 4:
        return None
    data = quadratic_data(context.objective, context.variables)
    ellipsoid = _ellipsoid(context.constraints[0], context.variables)
    if data is None or ellipsoid is None:
        return None
    matrix, linear, offset = data
    metric, center, level, kind = ellipsoid
    if any(v != 0 for v in (*linear, *center)) or any(
        v.free_symbols for v in (*matrix, *metric)
    ):
        return None
    if kind != "eq" and matrix.is_positive_semidefinite is True:
        return context.result(
            "centered_ellipsoid", dict(zip(context.variables, center)), metric=metric
        )
    parameter = structural_coordinate(
        "ellipsoid_eigenvalue",
        (context.objective, context.constraints, context.variables),
    )
    roots = _real_roots((matrix - parameter * metric).det(), parameter)
    if not roots:
        return None
    eigenvalue = roots[0]
    for vector in (matrix - eigenvalue * metric).nullspace():
        norm = sp.simplify((vector.T * metric * vector)[0])
        if norm.is_positive is not True:
            continue
        point = dict(zip(context.variables, sp.sqrt(level / norm) * vector))
        value = sp.simplify(offset + level * eigenvalue)
        if kind == "gt" and level.is_positive is True:
            return StructuralGlobalResult(
                "centered_ellipsoid",
                value,
                attained=False,
                metadata={
                    "metric": metric,
                    "eigenvalue": eigenvalue,
                    "limiting_point": point,
                },
            )
        return context.result(
            "centered_ellipsoid", point, metric=metric, eigenvalue=eigenvalue
        )
    return None
