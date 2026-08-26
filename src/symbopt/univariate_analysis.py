from __future__ import annotations

import sympy as sp
from sympy.calculus.util import continuous_domain, function_range
from sympy.core.relational import Relational

from .structural_results import StructuralGlobalResult


def _univariate_domain(constraints, variable):
    domain = sp.S.Reals
    for relation in constraints:
        if not isinstance(relation, Relational):
            return None
        other_symbols = relation.free_symbols - {variable}
        if other_symbols:
            return None
        lhs, rhs = relation.lhs, relation.rhs
        if lhs == variable and rhs.is_extended_real:
            if isinstance(relation, sp.GreaterThan):
                piece = sp.Interval(rhs, sp.oo)
            elif isinstance(relation, sp.StrictGreaterThan):
                piece = sp.Interval.open(rhs, sp.oo)
            elif isinstance(relation, sp.LessThan):
                piece = sp.Interval(-sp.oo, rhs)
            elif isinstance(relation, sp.StrictLessThan):
                piece = sp.Interval.open(-sp.oo, rhs)
            elif isinstance(relation, sp.Equality):
                piece = sp.FiniteSet(rhs)
            else:
                return None
        elif rhs == variable and lhs.is_extended_real:
            if isinstance(relation, sp.GreaterThan):
                piece = sp.Interval(-sp.oo, lhs)
            elif isinstance(relation, sp.StrictGreaterThan):
                piece = sp.Interval.open(-sp.oo, lhs)
            elif isinstance(relation, sp.LessThan):
                piece = sp.Interval(lhs, sp.oo)
            elif isinstance(relation, sp.StrictLessThan):
                piece = sp.Interval.open(lhs, sp.oo)
            elif isinstance(relation, sp.Equality):
                piece = sp.FiniteSet(lhs)
            else:
                return None
        else:
            return None
        domain = sp.Intersection(domain, piece)
    return domain


def _set_attains(range_set, value) -> bool:
    if value in (sp.oo, -sp.oo):
        return False
    try:
        contains = range_set.contains(value)
    except (NotImplementedError, TypeError, ValueError):
        return False
    return contains is sp.S.true or contains is True


def _point_for_value(objective, variable, domain, value):
    if value in (sp.oo, -sp.oo):
        return ()
    try:
        solutions = sp.solveset(sp.Eq(objective, value), variable, domain=domain)
    except (NotImplementedError, ValueError, TypeError):
        return ()
    if isinstance(solutions, sp.FiniteSet):
        for root in solutions:
            if root.is_real is not False:
                return ({variable: sp.simplify(root)},)
    if isinstance(solutions, sp.ImageSet):
        try:
            lam = solutions.lamda
            base_sets = solutions.base_sets
            if len(lam.variables) == 1 and base_sets == (sp.S.Integers,):
                root = sp.simplify(lam.expr.subs(lam.variables[0], 0))
                if domain.contains(root) is sp.S.true:
                    return ({variable: root},)
        except (AttributeError, TypeError, ValueError):
            pass
    if isinstance(solutions, sp.Union):
        for component in solutions.args:
            if isinstance(component, sp.ImageSet):
                try:
                    lam = component.lamda
                    base_sets = component.base_sets
                    if len(lam.variables) == 1 and base_sets == (sp.S.Integers,):
                        for index in (0, 1, -1, 2, -2):
                            root = sp.simplify(lam.expr.subs(lam.variables[0], index))
                            if domain.contains(root) is sp.S.true:
                                return ({variable: root},)
                except (AttributeError, TypeError, ValueError):
                    continue
    return ()


def _range_result(kind, objective, constraints, variable, domain):
    try:
        natural = continuous_domain(objective, variable, sp.S.Reals)
        feasible_domain = sp.Intersection(domain, natural)
        if feasible_domain is sp.S.EmptySet:
            return None
        image = function_range(objective, variable, feasible_domain)
        infimum = sp.simplify(image.inf)
    except (NotImplementedError, ValueError, TypeError, AttributeError, RecursionError):
        return None
    attained = _set_attains(image, infimum)
    points = (
        _point_for_value(objective, variable, feasible_domain, infimum)
        if attained
        else ()
    )
    return StructuralGlobalResult(
        kind=kind,
        value=infimum,
        points=points,
        attained=attained,
        certified=True,
        metadata={
            "domain": sp.sstr(feasible_domain),
            "range": sp.sstr(image),
            "analysis": "exact SymPy continuous-domain/function-range",
        },
    )
