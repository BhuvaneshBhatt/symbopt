"""Adapt global quadratic evidence to the symbolic optimization portfolio."""

import sympy as sp

from .quadratic_faces import solve_faces
from .quadratic_models import quadratic_controls, quadratic_problem
from .rule_utils import affine_data, polynomial_degree_bound
from .structural_results import StructuralGlobalResult


def quadratic_structural_result(result):
    """Keep incomplete bounds and numerical incumbents distinct from exact optima."""
    return StructuralGlobalResult(
        f"quadratic_{result.backend}",
        result.objective_value
        if result.objective_value is not None
        else result.lower_bound,
        (result.point_mapping,) if result.point else (),
        result.attained,
        result.exact and result.certified,
        {
            **result.metadata,
            "quadratic_result": result,
            "primal_exact": bool(result.point and result.bounds_verified),
            "infeasible": result.exact and result.status == "infeasible",
            "unbounded": result.exact and result.status == "unbounded",
        },
    )


def try_quadratic_faces(context):
    """Use a small complete face search after the cheaper structural rules."""
    variables = context.variables
    if not 2 <= len(variables) <= 4 or len(context.constraints) > 8:
        return None
    if polynomial_degree_bound(context.objective, variables) != 2:
        return None
    if any(
        affine_data(r.lhs - r.rhs, variables) is None
        or not isinstance(r, (sp.Equality, sp.GreaterThan, sp.LessThan))
        for r in context.constraints
    ):
        return None
    problem = quadratic_problem(
        context.objective, context.constraints, variables, use_symbol_assumptions=False
    )
    result = solve_faces(problem, quadratic_controls(max_faces=256, certified=True))
    return quadratic_structural_result(result) if result.exact else None
