"""Unbounded trigonometric objectives with an isolated pole and bounded remainder."""

import sympy as sp

from .rule_utils import affine_data
from .structural_results import StructuralGlobalResult


def try_tangent_pole(context):
    """Prove escape at a single affine tangent pole under bounded perturbations."""
    if len(context.variables) != 1 or context.constraints:
        return None
    variable = context.variables[0]
    tangents = context.objective.atoms(sp.tan)
    if len(tangents) != 1:
        return None
    tangent = next(iter(tangents))
    coefficient = context.objective.expand().coeff(tangent)
    if (
        coefficient.free_symbols
        or coefficient.is_real is not True
        or coefficient.is_zero is not False
    ):
        return None
    data = affine_data(tangent.args[0], (variable,))
    if data is None:
        return None
    (slope,), shift = data
    if (
        slope.is_real is not True
        or slope.is_zero is not False
        or shift.is_real is not True
        or shift.free_symbols
        or slope.free_symbols
    ):
        return None
    remainder = sp.expand(context.objective - coefficient * tangent)
    if remainder == 0:
        return None
    for term in sp.Add.make_args(remainder):
        if not term.has(variable):
            if term.is_finite is not True or term.is_real is not True:
                return None
            continue
        scalar, function = term.as_independent(variable, as_Add=False)
        if (
            scalar.is_real is not True
            or scalar.is_finite is not True
            or function.func not in (sp.sin, sp.cos)
            or function.args[0].is_real is not True
        ):
            return None
    return StructuralGlobalResult(
        "tangent_pole",
        -sp.oo,
        attained=False,
        metadata={
            "pole": (sp.pi / 2 - shift) / slope,
            "coefficient": coefficient,
            "bounded_remainder": remainder,
        },
    )
