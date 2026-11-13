"""Exact strict-affine infeasibility checks for the symbolic portfolio."""

import sympy as sp

from .quadratic_lp import affine_system, exact_linear_minimize
from .structural_results import StructuralGlobalResult


def try_strict_affine_infeasible(context):
    """Prove an empty strict polyhedron by maximizing a shared positive slack.

    A finite conjunction of strict rows has a feasible point exactly when the
    relaxed system permits a common slack in (0,1]. Rational dual/Farkas
    witnesses justify the decisive LP result; unchecked statuses are ignored.
    """
    if (
        len(context.variables) > 4
        or len(context.constraints) > 8
        or not any(
            isinstance(r, (sp.StrictGreaterThan, sp.StrictLessThan))
            for r in context.constraints
        )
    ):
        return None
    slack = sp.Dummy("strict_slack", real=True)
    relations = []
    for relation in context.constraints:
        if isinstance(relation, sp.StrictGreaterThan):
            relations.append(sp.Ge(relation.lhs - relation.rhs, slack, evaluate=False))
        elif isinstance(relation, sp.StrictLessThan):
            relations.append(sp.Ge(relation.rhs - relation.lhs, slack, evaluate=False))
        else:
            relations.append(relation)
    relations.extend((sp.Ge(slack, 0, evaluate=False), sp.Le(slack, 1, evaluate=False)))
    system = affine_system(relations, (*context.variables, slack))
    if system is None:
        return None
    objective = (sp.S.Zero,) * len(context.variables) + (-sp.S.One,)
    result = exact_linear_minimize(objective, system)
    if result.status == "infeasible" or (
        result.status == "optimal" and result.value == 0
    ):
        return StructuralGlobalResult(
            "strict_affine_infeasible",
            sp.oo,
            attained=False,
            metadata={
                "infeasible": True,
                "slack_system": system,
                "slack_objective": objective,
                "slack_proof": result.proof,
                "slack_value": result.value,
            },
        )
    return None
