from __future__ import annotations

import sympy as sp

from .structural_results import StructuralGlobalResult
from .univariate_analysis import _range_result, analyze_univariate


def _domain_components(domain):
    if isinstance(domain, sp.Interval):
        return (domain,)
    if isinstance(domain, sp.FiniteSet):
        return tuple(sp.FiniteSet(item) for item in domain)
    if isinstance(domain, sp.Union):
        components = []
        for part in domain.args:
            if isinstance(part, (sp.Interval, sp.FiniteSet)):
                components.append(part)
            else:
                return ()
        return tuple(components)
    if domain is sp.S.Reals:
        return (sp.Interval(-sp.oo, sp.oo),)
    return ()


def _boundary_values(objective, variable, component, analysis=None):
    values = []
    if isinstance(component, sp.FiniteSet):
        for point in component:
            try:
                values.append(
                    (
                        sp.simplify(objective.subs(variable, point)),
                        point,
                        True,
                        "isolated",
                    )
                )
            except (TypeError, ValueError):
                continue
        return values
    for endpoint, direction, closed in (
        (component.start, "+", not component.left_open),
        (component.end, "-", not component.right_open),
    ):
        try:
            if endpoint is -sp.oo:
                value = (
                    analysis.endpoint_value(-sp.oo)
                    if analysis is not None
                    else sp.simplify(sp.limit(objective, variable, -sp.oo))
                )
            elif endpoint is sp.oo:
                value = (
                    analysis.endpoint_value(sp.oo)
                    if analysis is not None
                    else sp.simplify(sp.limit(objective, variable, sp.oo))
                )
            elif closed:
                value = sp.simplify(objective.subs(variable, endpoint))
            else:
                value = (
                    analysis.endpoint_value(
                        endpoint, direction=direction, attained=False
                    )
                    if analysis is not None
                    else sp.simplify(
                        sp.limit(objective, variable, endpoint, dir=direction)
                    )
                )
        except (NotImplementedError, ValueError, TypeError, RecursionError):
            continue
        values.append(
            (value, endpoint if closed else None, closed, f"boundary:{endpoint}")
        )
    return values


def _stationary_points(objective, variable, component, analysis=None):
    if analysis is not None and component == analysis.feasible_domain:
        return analysis.stationary_points
    try:
        roots = sp.solveset(
            sp.Eq(sp.diff(objective, variable), 0), variable, domain=component
        )
    except (NotImplementedError, ValueError, TypeError, RecursionError):
        return None
    if isinstance(roots, sp.FiniteSet):
        return tuple(sp.simplify(root) for root in roots if root.is_real is not False)
    if roots is sp.S.EmptySet:
        return ()
    return None


def try_transcendental(objective, constraints, variables, analysis=None):
    """Certify common univariate transcendental extrema from real-domain geometry.

    This route explicitly intersects user constraints with SymPy's real
    continuous domain, records singularities/poles, detects one-sided unbounded
    limits, and uses exact convexity plus exhaustive stationary points on each
    interval component.  It declines nonconvex finite problems.
    """
    variables = tuple(variables)
    if len(variables) != 1:
        return None
    variable = variables[0]
    if not (objective.has(sp.exp, sp.log) or sp.fraction(sp.cancel(objective))[1] != 1):
        return None
    analysis = analysis or analyze_univariate(objective, constraints, variable)
    if analysis is None:
        return None
    natural = analysis.natural_domain
    domain = analysis.feasible_domain
    if natural is None or domain is None:
        return None
    if domain is sp.S.EmptySet:
        return None
    components = _domain_components(domain)
    if not components:
        return None

    poles = analysis.singularity_set
    if poles is None:
        poles = sp.S.EmptySet

    entries = []
    convex_components = []
    for component in components:
        entries.extend(_boundary_values(objective, variable, component, analysis))
        if isinstance(component, sp.FiniteSet):
            convex_components.append(True)
            continue
        # A -oo one-sided/domain limit proves the global infimum immediately;
        # no convexity premise is required for that component.
        if any(value is -sp.oo for value, _, _, _ in entries):
            return StructuralGlobalResult(
                kind="transcendental_domain",
                value=-sp.oo,
                points=(),
                attained=False,
                certified=True,
                metadata={
                    "domain": sp.sstr(domain),
                    "natural_domain": sp.sstr(natural),
                    "singularities": sp.sstr(poles),
                    "analysis": "shared univariate endpoint analysis proves unbounded below",
                    "providers": analysis.provider_summary,
                },
            )
        try:
            from sympy.calculus.util import is_convex

            convex = bool(is_convex(objective, variable, domain=component))
        except (NotImplementedError, ValueError, TypeError, RecursionError):
            convex = False
        convex_components.append(convex)
        if not convex:
            return None
        roots = _stationary_points(objective, variable, component, analysis)
        if roots is None:
            return None
        for root in roots:
            try:
                entries.append(
                    (
                        sp.simplify(objective.subs(variable, root)),
                        root,
                        True,
                        "stationary",
                    )
                )
            except (TypeError, ValueError):
                return None

    finite_entries = [entry for entry in entries if entry[0] is not sp.oo]
    if not finite_entries:
        return None
    if any(entry[0] is -sp.oo for entry in finite_entries):
        return StructuralGlobalResult(
            kind="transcendental_domain",
            value=-sp.oo,
            points=(),
            attained=False,
            certified=True,
            metadata={
                "domain": sp.sstr(domain),
                "natural_domain": sp.sstr(natural),
                "singularities": sp.sstr(poles),
                "convex_components": tuple(convex_components),
            },
        )
    try:
        best_value = min(entry[0] for entry in finite_entries)
    except (TypeError, ValueError):
        return None
    best_value = sp.simplify(best_value)
    points = tuple(
        {variable: sp.simplify(point)}
        for value, point, closed, _ in finite_entries
        if point is not None and closed and sp.simplify(value - best_value) == 0
    )
    return StructuralGlobalResult(
        kind="transcendental_domain",
        value=best_value,
        points=points,
        attained=bool(points),
        certified=True,
        metadata={
            "domain": sp.sstr(domain),
            "natural_domain": sp.sstr(natural),
            "singularities": sp.sstr(poles),
            "convex_components": tuple(convex_components),
            "analysis": "shared univariate domain/partition + exact convexity/stationarity",
            "providers": analysis.provider_summary,
        },
    )


def _common_asymptotic_form(objective, variable) -> bool:
    _numerator, denominator = sp.fraction(sp.cancel(objective))
    rational = denominator != 1 and denominator.has(variable)
    radical = any(
        isinstance(node, sp.Pow)
        and node.exp.is_Rational
        and node.exp.q % 2 == 0
        and node.base.has(variable)
        for node in sp.preorder_traversal(objective)
    )
    return bool(rational or objective.has(sp.log, sp.exp) or radical)


def try_asymptotic_solver(objective, constraints, variables, analysis=None):
    """Exact range/asymptotic analysis for common univariate elementary forms."""
    variables = tuple(variables)
    if len(variables) != 1:
        return None
    variable = variables[0]
    if not _common_asymptotic_form(objective, variable):
        return None
    analysis = analysis or analyze_univariate(objective, constraints, variable)
    if analysis is None:
        return None
    return _range_result(
        "univariate_asymptotic",
        objective,
        constraints,
        variable,
        analysis=analysis,
    )
