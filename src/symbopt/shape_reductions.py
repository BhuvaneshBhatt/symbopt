"""Verified coordinate reductions to smaller symbolic optimization problems."""

from dataclasses import dataclass
from functools import reduce
from math import gcd

import sympy as sp

from .rule_utils import affine_data, truth
from .structural_results import StructuralGlobalResult
from .symbolic_cache import structural_coordinate


@dataclass
class CoordinateReduction:
    """An equivalent smaller objective/domain with exact maps back to input variables."""

    kind: str
    objective: sp.Expr
    constraints: tuple
    variables: tuple
    lifts: tuple
    proof: dict


def _relations(relations, substitution):
    result = []
    for relation in relations:
        value = relation.subs(substitution, simultaneous=True)
        if value is sp.true:
            continue
        result.append(sp.Eq(0, 1, evaluate=False) if value is sp.false else value)
    return tuple(result)


def affine_reduction(context):
    """Eliminate fixed-rank affine equalities with a real, surjective parameterization."""
    equalities = [
        r.lhs - r.rhs for r in context.constraints if isinstance(r, sp.Equality)
    ]
    if not equalities or len(context.variables) > 8:
        return None
    data = [affine_data(eq, context.variables) for eq in equalities]
    if any(d is None for d in data):
        return None
    # Parameter-dependent ranks require strata, even when a generic solution
    # happens to satisfy the equations (a*x=0 is the basic counterexample).
    if any(
        c.free_symbols or c.is_algebraic is not True or c.is_real is not True
        for d in data
        for c in d[0]
    ):
        return None
    solutions = sp.linsolve(equalities, context.variables)
    if not isinstance(solutions, sp.FiniteSet) or len(solutions) != 1:
        return None
    solution = tuple(next(iter(solutions)))
    free = tuple(
        v for v in context.variables if any(value.has(v) for value in solution)
    )
    if len(free) >= len(context.variables):
        return None
    aliases = {
        v: structural_coordinate(
            v.name,
            ("affine", context.objective, context.constraints, context.variables, v),
        )
        for v in free
    }
    lift = {
        v: sp.simplify(value.xreplace(aliases))
        for v, value in zip(context.variables, solution)
    }
    if any(value.is_real is not True for value in lift.values()):
        return None
    if any(
        truth(sp.Eq(eq.subs(lift, simultaneous=True), 0)) is not True
        for eq in equalities
    ):
        return None
    objective = sp.simplify(context.objective.subs(lift, simultaneous=True))
    constraints = _relations(context.constraints, lift)
    return CoordinateReduction(
        "affine_elimination",
        objective,
        constraints,
        tuple(aliases.values()),
        (lift,),
        {"parameterization": lift, "fixed_rank": True},
    )


def power_reduction(context):
    """Use the gcd of exponents, preserving the image and all real inverse branches."""
    if len(context.variables) != 1:
        return None
    x = context.variables[0]

    def terms(expression):
        result = []
        for term in sp.Add.make_args(sp.expand(expression)):
            power = term.as_powers_dict().get(x, sp.S.Zero)
            if power.is_Integer is not True or power < 0:
                return None
            coefficient = term / x**power
            if coefficient.has(x):
                return None
            result.append((int(power), coefficient))
        return result

    numerator, denominator = sp.fraction(sp.together(context.objective))
    polys = [terms(p) for p in (numerator, denominator)]
    if any(poly is None for poly in polys):
        return None
    exponents = [power for poly in polys for power, _ in poly if power]
    k = reduce(gcd, exponents, 0)
    if k < 2:
        return None
    t = structural_coordinate(
        "power", (context.objective, context.constraints, context.variables)
    )

    def convert(expression):
        p, q = sp.fraction(sp.together(expression))
        converted = []
        for part in (p, q):
            poly = terms(part)
            if poly is None or any(power % k for power, _ in poly):
                return None
            converted.append(sum(c * t ** (power // k) for power, c in poly))
        return converted[0] / converted[1]

    objective = convert(context.objective)
    constraints = [sp.Ge(t, 0)] if k % 2 == 0 else []
    signs = {-1, 0, 1}
    for relation in context.constraints:
        residual = relation.lhs - relation.rhs
        if isinstance(relation, sp.Unequality) and residual in (x, -x):
            constraints.append(sp.Ne(t, 0))
            continue
        converted_residual = convert(residual)
        if converted_residual is None:
            # Simple one-sided sign domains map directly under powers.
            if residual in (x, -x) and not isinstance(
                relation, (sp.Equality, sp.Unequality)
            ):
                if k % 2 == 0:
                    # Intersect inverse-branch signs before taking the image:
                    # x >= 0 and x <= 0 has image {0}, not [0, infinity).
                    signs.intersection_update(
                        sign for sign in (-1, 0, 1) if relation.subs(x, sign) is sp.true
                    )
                    continue
                constraints.append(relation.func(t if residual == x else -t, 0))
                continue
            return None
        condition = relation.func(converted_residual, 0)
        if condition is not sp.true:
            constraints.append(
                sp.Eq(0, 1, evaluate=False) if condition is sp.false else condition
            )
    root = sp.real_root(t, k) if k % 2 else t ** sp.Rational(1, k)
    if k % 2:
        lifts = ({x: root},)
    else:
        if not signs:
            return None
        if signs == {0}:
            constraints.append(sp.Eq(t, 0))
            lifts = ({x: sp.S.Zero},)
        else:
            if 0 not in signs:
                constraints.append(sp.Gt(t, 0))
            lifts = tuple({x: sign * root} for sign in (-1, 1) if sign in signs)
    return CoordinateReduction(
        "power_substitution",
        objective,
        tuple(constraints),
        (t,),
        lifts,
        {"power": k, "forward": x**k, "image_nonnegative": k % 2 == 0},
    )


def ridge_reduction(context):
    """Reduce an affine ridge, using either its fibers or an exact box image."""
    if len(context.variables) < 2:
        return None
    candidates = [
        n
        for n in sp.preorder_traversal(context.objective)
        if n.is_Add and len(n.free_symbols.intersection(context.variables)) >= 2
    ]
    t = structural_coordinate(
        "ridge", (context.objective, context.constraints, context.variables)
    )
    for form in candidates[:12]:
        data = affine_data(form, context.variables)
        if data is None:
            continue
        coefficients, constant = data
        pivots = [
            i
            for i, c in enumerate(coefficients)
            if c.is_zero is False and c.is_real is True
        ]
        if not pivots or constant.is_real is not True:
            continue
        pivot = pivots[0]
        others = sum(
            c * v
            for i, (c, v) in enumerate(zip(coefficients, context.variables))
            if i != pivot
        )
        substitution = {
            context.variables[pivot]: (t - constant - others) / coefficients[pivot]
        }
        objective = sp.simplify(context.objective.subs(substitution, simultaneous=True))
        if objective.free_symbols.intersection(context.variables):
            continue
        constraints = _relations(context.constraints, substitution)
        if not any(r.free_symbols.intersection(context.variables) for r in constraints):
            lift = dict.fromkeys(context.variables, sp.S.Zero)
            lift[context.variables[pivot]] = (t - constant) / coefficients[pivot]
            if all(value.is_real is True for value in lift.values()):
                return CoordinateReduction(
                    "affine_ridge",
                    objective,
                    constraints,
                    (t,),
                    (lift,),
                    {"forward": form},
                )
        if not context.rectangle or not all(
            i.finite for i in context.rectangle.values()
        ):
            continue
        low, high = {}, {}
        low_open = high_open = False
        for v, c in zip(context.variables, coefficients):
            interval = context.rectangle[v]
            if c == 0:
                low[v] = high[v] = interval.sample()
            elif c.is_positive is True or c.is_negative is True:
                positive = c.is_positive is True
                low[v] = interval.left if positive else interval.right
                high[v] = interval.right if positive else interval.left
                low_open |= interval.left_open if positive else interval.right_open
                high_open |= interval.right_open if positive else interval.left_open
            else:
                break
        else:
            left, right = form.subs(low), form.subs(high)
            if truth(left < right) is not True:
                continue
            fraction = (t - left) / (right - left)
            lift = {
                v: (1 - fraction) * low[v] + fraction * high[v]
                for v in context.variables
            }
            domain = (
                sp.Gt(t, left) if low_open else sp.Ge(t, left),
                sp.Lt(t, right) if high_open else sp.Le(t, right),
            )
            return CoordinateReduction(
                "affine_ridge",
                objective,
                domain,
                (t,),
                (lift,),
                {"forward": form, "image": (left, right)},
            )
    return None


def radial_reduction(context):
    """Recognize dependence on the squared Euclidean radius by an exact identity."""
    if len(context.variables) < 2:
        return None
    t = structural_coordinate(
        "radius_squared", (context.objective, context.constraints, context.variables)
    )
    squared = sum(v * v for v in context.variables)
    axis = dict.fromkeys(context.variables, sp.S.Zero)
    axis[context.variables[0]] = sp.sqrt(t)
    objective = sp.simplify(context.objective.subs(axis, simultaneous=True))
    if not objective.has(t) or objective.free_symbols.intersection(context.variables):
        return None
    if sp.simplify(context.objective - objective.subs(t, squared)) != 0:
        return None
    constraints = [sp.Ge(t, 0)]
    for relation in context.constraints:
        residual = relation.lhs - relation.rhs
        reduced = sp.simplify(residual.subs(axis, simultaneous=True))
        if sp.simplify(residual - reduced.subs(t, squared)) != 0:
            return None
        value = relation.func(reduced, 0)
        if value is not sp.true:
            constraints.append(
                sp.Eq(0, 1, evaluate=False) if value is sp.false else value
            )
    return CoordinateReduction(
        "radial_reduction",
        objective,
        tuple(constraints),
        (t,),
        (axis,),
        {"forward": squared, "image": "nonnegative reals"},
    )


def solve_reduction(context, reduction, solve):
    """Solve the reduced problem once, preserving proof and inverse-branch checks."""
    if not reduction.variables:
        point = reduction.lifts[0]
        return context.result(reduction.kind, point, **reduction.proof)
    analytic = None
    if len(reduction.variables) == 1:
        from .quadratic_optimization import _scalar_quadratic
        from .rule_utils import polynomial_degree_bound
        from .symbolic_cache import cached_poly

        degree = polynomial_degree_bound(reduction.objective, reduction.variables)
        if degree is not None and degree <= 2:
            analytic = _scalar_quadratic(
                cached_poly(reduction.objective, reduction.variables),
                reduction.constraints,
                reduction.variables[0],
            )
    if analytic is not None:
        value, attained = analytic.value, analytic.attained
        inner_points = analytic.points
        certificate = analytic
    else:
        result = solve(reduction.objective, reduction.constraints, reduction.variables)
        if not result.certified or result.optimum_value is None:
            return None
        value, attained = result.optimum_value, result.attained
        inner_points = tuple(
            c.original_variable_values or c.primal_values for c in result.candidates
        )
        certificate = result.certificate
    points = []
    for inner in inner_points:
        for lift in reduction.lifts:
            point = {
                v: sp.simplify(value.subs(inner, simultaneous=True))
                for v, value in lift.items()
            }
            checked = context.result(reduction.kind, point)
            if (
                checked is not None
                and truth(sp.Eq(checked.value, value)) is True
                and checked.points[0] not in points
            ):
                points.append(checked.points[0])
    return StructuralGlobalResult(
        reduction.kind,
        value,
        tuple(points),
        attained,
        metadata={
            **reduction.proof,
            "reduced_objective": reduction.objective,
            "reduced_constraints": reduction.constraints,
            "inner_certificate": certificate,
        },
    )
