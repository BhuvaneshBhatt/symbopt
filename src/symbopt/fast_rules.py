"""Budgeted exact structural rules before general optimization algorithms."""

from dataclasses import replace
from functools import lru_cache
from math import comb

import sympy as sp

from .affine_feasibility import try_strict_affine_infeasible
from .box_rules import try_monotone_box, try_multiaffine_box
from .convex_rules import try_concave_endpoints, try_convex_stationary
from .elementary_rules import (
    try_nonnegative_sum,
    try_positive_inequality,
    try_unbounded_ray,
)
from .piecewise_linear_rules import epigraph_reduction, try_weighted_median
from .pole_rules import try_tangent_pole
from .quadratic_routing import try_quadratic_faces
from .rule_utils import RULE_ERRORS, RuleContext, affine_data, polynomial_degree_bound
from .shape_reductions import (
    CoordinateReduction,
    affine_reduction,
    power_reduction,
    radial_reduction,
    ridge_reduction,
    solve_reduction,
)
from .spectral_rules import (
    try_centered_ellipsoid,
    try_linear_ellipsoid,
    try_quadratic_ellipsoid,
    try_rayleigh,
)
from .structural_results import StructuralGlobalResult
from .symbolic_cache import cached_hessian


def expansion_size(expression, limit=256):
    """Bound expanded support size before requesting polynomial conversion."""
    if expression.is_Add:
        return min(limit + 1, sum(expansion_size(a, limit) for a in expression.args))
    if expression.is_Mul:
        size = 1
        for arg in expression.args:
            size *= expansion_size(arg, limit)
            if size > limit:
                return limit + 1
        return size
    if expression.is_Pow and expression.exp.is_Integer and expression.exp > 0:
        size = expansion_size(expression.base, limit)
        if size > limit:
            return limit + 1
        return min(limit + 1, comb(int(expression.exp) + size - 1, size - 1))
    return 1


def preferred_direct(context):
    """Select direct LP/QP solving for closed affine quadratic problems."""
    try:
        degree = polynomial_degree_bound(context.objective, context.variables)
        if degree is None or degree > 2:
            return False
        parameters = context.objective.free_symbols - set(context.variables)
        return (
            degree <= 2
            and (
                degree <= 1
                or bool(parameters)
                or cached_hessian(
                    context.objective, context.variables
                ).is_positive_semidefinite
                is True
            )
            and all(
                isinstance(r, (sp.Equality, sp.GreaterThan, sp.LessThan))
                and affine_data(r.lhs - r.rhs, context.variables) is not None
                for r in context.constraints
            )
        )

    except sp.PolynomialError:
        return False


@lru_cache(maxsize=1024)
def _rule_profile(objective, constraints, variables):
    """Cache immutable work-budget and direct-route recognition facts."""
    expressions = (objective, *(r.lhs - r.rhs for r in constraints))
    eligible = (
        bool(variables)
        and len(variables) <= 12
        and not any(v.is_real is False for v in variables)
    )
    eligible = eligible and not any(e.has(sp.Float) for e in expressions)
    eligible = eligible and sum(int(sp.count_ops(e)) for e in expressions) <= 240
    if not eligible:
        return False, False, False
    small = all(expansion_size(e) <= 256 for e in expressions)
    return (
        True,
        small,
        small and preferred_direct(RuleContext(objective, constraints, variables)),
    )


@lru_cache(maxsize=1024)
def _recognize_fast_rules(objective, constraints, variables, depth):
    """Recognize a proof or equivalent reduction without invoking another solver."""
    eligible, small, direct = _rule_profile(objective, constraints, variables)
    if not eligible:
        return None
    for relation in constraints:
        if relation.func(*relation.args) is sp.false:
            return StructuralGlobalResult(
                "infeasible_relation",
                sp.oo,
                attained=False,
                metadata={"infeasible": True, "contradiction": relation},
            )
    context = RuleContext(objective, constraints, variables)
    infeasible = try_strict_affine_infeasible(context)
    if infeasible is not None:
        return infeasible
    if direct:
        return None
    for rule in (try_tangent_pole, try_unbounded_ray, try_nonnegative_sum):
        try:
            result = rule(context)
            if result is not None:
                return result
        except RULE_ERRORS:
            continue
    if not small:
        return None
    if depth < 3 and len(variables) == 1 and objective.is_rational_function(*variables):
        try:
            result = try_positive_inequality(context)
            if result is not None:
                return result
            reduction = power_reduction(context)
            if reduction is not None:
                return reduction
        except RULE_ERRORS:
            pass
    for rule in (
        try_weighted_median,
        try_positive_inequality,
        try_linear_ellipsoid,
        try_rayleigh,
        try_quadratic_ellipsoid,
        try_centered_ellipsoid,
        try_multiaffine_box,
        try_monotone_box,
        try_concave_endpoints,
        try_convex_stationary,
        try_quadratic_faces,
    ):
        try:
            result = rule(context)
            if result is not None:
                return result
        except RULE_ERRORS:
            continue
    if depth >= 3:
        return None
    for recognize in (
        affine_reduction,
        power_reduction,
        ridge_reduction,
        radial_reduction,
        epigraph_reduction,
    ):
        try:
            reduction = recognize(context)
            if reduction is not None:
                return reduction
        except RULE_ERRORS:
            continue
    return None


def try_fast_rules(objective, constraints, variables, solve, *, depth=0):
    """Try bounded exact rules, then solve at most one selected coordinate reduction.

    Recognition depends only on immutable problem data, so repeat queries reuse
    both matches and refusals. Copy a matched plan before exposing its mutable
    proof metadata or points. Inner solver results never enter this cache.
    """
    variables, constraints = tuple(variables), tuple(constraints)
    plan = _recognize_fast_rules(objective, constraints, variables, depth)
    if plan is None:
        return None
    if isinstance(plan, CoordinateReduction):
        plan = replace(
            plan, lifts=_copy_rule_data(plan.lifts), proof=_copy_rule_data(plan.proof)
        )
        try:
            return solve_reduction(
                RuleContext(objective, constraints, variables), plan, solve
            )
        except RULE_ERRORS:
            return None
    return replace(
        plan,
        points=_copy_rule_data(plan.points),
        metadata=_copy_rule_data(plan.metadata),
    )


def _copy_rule_data(value):
    """Copy plan containers and matrices while sharing immutable expressions."""
    if isinstance(value, dict):
        return {key: _copy_rule_data(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy_rule_data(item) for item in value)
    if isinstance(value, list):
        return [_copy_rule_data(item) for item in value]
    if isinstance(value, sp.MatrixBase):
        return value.copy()
    return value
