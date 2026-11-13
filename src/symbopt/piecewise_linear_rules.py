"""Weighted medians and exact epigraph reductions for convex piecewise linear data."""

from functools import cmp_to_key

import sympy as sp

from .rational_optimization import _compare
from .rule_utils import affine_data, truth
from .shape_reductions import CoordinateReduction
from .symbolic_cache import structural_coordinate


def try_weighted_median(context):
    """Find weighted median endpoints by exact ordering of affine absolute values."""
    if len(context.variables) != 1 or not context.rectangle:
        return None
    x = context.variables[0]
    interval = context.rectangle[x]
    if (interval.left, interval.right) != (-sp.oo, sp.oo):
        return None
    atoms = []
    offset = sp.S.Zero
    for term in sp.Add.make_args(context.objective):
        coefficient, dependent = term.as_independent(x, as_Add=False)
        if not dependent.has(x):
            offset += term
            continue
        if dependent.func is not sp.Abs or coefficient.is_positive is not True:
            return None
        data = affine_data(dependent.args[0], (x,))
        if data is None:
            return None
        (slope,), constant = data
        if slope.is_real is not True or slope.is_zero is not False:
            return None
        atoms.append((sp.simplify(-constant / slope), coefficient * abs(slope)))
    if not atoms:
        return None
    atoms.sort(key=cmp_to_key(lambda a, b: _compare(a[0], b[0])))
    total = sum(w for _, w in atoms)
    cumulative = sp.S.Zero
    for i, (center, weight) in enumerate(atoms):
        cumulative += weight
        if truth(2 * cumulative >= total) is True:
            right = (
                atoms[i + 1][0]
                if i + 1 < len(atoms) and truth(2 * cumulative == total) is True
                else center
            )
            result = context.result(
                "weighted_median",
                {x: center},
                median_interval=(center, right),
                weights=tuple(atoms),
            )
            if result is not None and right != center:
                other = context.result("weighted_median", {x: right})
                if other is not None:
                    from dataclasses import replace

                    return replace(result, points=(*result.points, *other.points))
            return result
        if truth(2 * cumulative < total) is not True:
            return None
    return None


def epigraph_reduction(context):
    """Replace positive affine Abs/Max terms by a linear epigraph without branches."""
    objective = sp.S.Zero
    constraints = list(context.constraints)
    auxiliary = []
    for term in sp.Add.make_args(context.objective):
        coefficient, dependent = term.as_independent(*context.variables, as_Add=False)
        if affine_data(term, context.variables) is not None:
            objective += term
            continue
        if coefficient.is_positive is not True:
            return None
        if dependent.func is sp.Abs:
            expressions = (dependent.args[0], -dependent.args[0])
        elif dependent.func is sp.Max:
            expressions = dependent.args
        else:
            return None
        if any(affine_data(expr, context.variables) is None for expr in expressions):
            return None
        variable = structural_coordinate(
            "epigraph",
            (context.objective, context.constraints, context.variables, len(auxiliary)),
        )
        auxiliary.append(variable)
        objective += coefficient * variable
        constraints.extend(sp.Ge(variable, expr) for expr in expressions)
    if not auxiliary or len(auxiliary) > 12:
        return None
    # Coupled affine constraints are retained, including strict constraints.
    if any(
        affine_data(r.lhs - r.rhs, context.variables) is None
        for r in context.constraints
    ):
        return None
    lift = {v: v for v in context.variables}
    return CoordinateReduction(
        "linear_epigraph",
        objective,
        tuple(constraints),
        (*context.variables, *auxiliary),
        (lift,),
        {"epigraph_variables": tuple(auxiliary), "equivalent_minimum": True},
    )
