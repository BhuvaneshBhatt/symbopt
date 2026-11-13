"""Composition-based curvature and exact sufficient KKT certificates."""

import sympy as sp

from .relation_checks import normalize_relational
from .rule_utils import affine_data, checked_point, linear_solution, truth
from .spectral_rules import quadratic_data
from .structural_results import StructuralGlobalResult


def curvature(expression, context):
    """Recognize a bounded grammar of globally valid convex/concave compositions."""
    if not expression.free_symbols.intersection(context.variables):
        return "affine" if expression.is_real is True else None
    if affine_data(expression, context.variables) is not None:
        return "affine" if context.real(expression) else None
    if expression.is_Add:
        kinds = [curvature(arg, context) for arg in expression.args]
        if all(k in {"affine", "convex"} for k in kinds):
            return "convex"
        if all(k in {"affine", "concave"} for k in kinds):
            return "concave"
        return None
    coefficient, dependent = expression.as_independent(*context.variables, as_Add=False)
    if coefficient != 1:
        kind = curvature(dependent, context)
        if coefficient.is_nonnegative is True:
            return kind
        if coefficient.is_nonpositive is True:
            return {
                "affine": "affine",
                "convex": "concave",
                "concave": "convex",
                None: None,
            }[kind]
        return None
    if expression.func is sp.exp:
        return (
            "convex"
            if curvature(expression.args[0], context) in {"affine", "convex"}
            else None
        )
    if expression.func is sp.log:
        inner = expression.args[0]
        if context.sign(inner) == 1 and curvature(inner, context) in {
            "affine",
            "concave",
        }:
            return "concave"
    if (
        expression.func is sp.Abs
        and affine_data(expression.args[0], context.variables) is not None
    ):
        return "convex"
    if expression.func is sp.Max:
        return (
            "convex"
            if all(
                curvature(arg, context) in {"affine", "convex"}
                for arg in expression.args
            )
            else None
        )
    if expression.is_Pow and not expression.exp.free_symbols.intersection(
        context.variables
    ):
        base, power = expression.args
        kind = curvature(base, context)
        if (
            power.is_Integer
            and power > 1
            and power % 2 == 0
            and kind == "affine"
            and context.real(base)
        ):
            return "convex"
        sign = context.sign(base)
        if sign in {0, 1, 2} and power.is_real is True:
            if (
                power.is_negative is True
                and sign == 1
                and kind in {"affine", "concave"}
            ):
                return "convex"
            if truth(power >= 1) is True and kind in {"affine", "convex"}:
                return "convex"
            if (
                truth(power > 0) is True
                and truth(power < 1) is True
                and kind in {"affine", "concave"}
            ):
                return "concave"
    data = quadratic_data(expression, context.variables)
    if data is not None:
        if data[0].is_positive_semidefinite is True:
            return "convex"
        if (-data[0]).is_positive_semidefinite is True:
            return "concave"
    return None


def smooth_at(expression, point):
    """Check branch differentiability before treating symbolic derivatives as gradients.

    Derivatives assigned at an Abs kink or a tied Max are not sufficient KKT
    data. In particular, three default Heaviside tie weights need not sum to one.
    """
    for atom in expression.atoms(sp.Abs):
        if atom.args[0].subs(point).is_zero is not False:
            return False
    for atom in expression.atoms(sp.Max):
        values = [arg.subs(point) for arg in atom.args]
        if not any(
            all(
                i == j or truth(value > other) is True for j, other in enumerate(values)
            )
            for i, value in enumerate(values)
        ):
            return False
    return True


def convex_kkt_certificate(context, point):
    """Check sufficient convex KKT conditions, with no constraint-qualification claim."""
    if curvature(context.objective, context) not in {"affine", "convex"}:
        return None
    checked = checked_point(
        context.objective, context.constraints, context.variables, point
    )
    if checked is None:
        return None
    point, value = checked
    if not smooth_at(context.objective, point):
        return None
    equality_gradients, active_gradients = [], []
    for relation in context.constraints:
        kind, residual = normalize_relational(relation)
        if kind == "neq":
            # Deleting points preserves a convex objective bound, but the
            # remaining constraints must still define the convex comparison set.
            continue
        if kind == "eq":
            if affine_data(residual, context.variables) is None:
                return None
            equality_gradients.append(
                sp.Matrix([sp.diff(residual, v).subs(point) for v in context.variables])
            )
        elif kind in {"ge", "gt"}:
            if curvature(residual, context) not in {"affine", "concave"}:
                return None
            if truth(sp.Eq(residual.subs(point), 0)) is True:
                if not smooth_at(residual, point):
                    return None
                active_gradients.append(
                    sp.Matrix(
                        [sp.diff(residual, v).subs(point) for v in context.variables]
                    )
                )
        else:
            return None
    gradient = sp.Matrix(
        [sp.diff(context.objective, v).subs(point) for v in context.variables]
    )
    if any(
        g.has(sp.nan, sp.zoo, sp.Derivative) or g.is_real is not True for g in gradient
    ):
        return None
    multipliers = tuple(
        sp.Dummy("lambda", real=True) for _ in equality_gradients
    ) + tuple(sp.Dummy("mu", real=True) for _ in active_gradients)
    if len(multipliers) > 24:
        return None
    columns = [*equality_gradients, *(-g for g in active_gradients)]
    if columns:
        equations = tuple(
            gradient + sp.Matrix.hstack(*columns) * sp.Matrix(multipliers)
        )
        solution = linear_solution(equations, multipliers)
        if solution is None or any(
            solution[m].is_nonnegative is not True
            for m in multipliers[len(equality_gradients) :]
        ):
            # Degeneracy can leave multiplier freedom. A rational feasibility LP
            # selects nonnegative multipliers; exact substitution is still decisive.
            try:
                from sympy.solvers.simplex import (
                    InfeasibleLPError,
                    UnboundedLPError,
                    lpmin,
                )
            except ImportError:
                return None

            conditions = [sp.Eq(e, 0) for e in equations]
            conditions += [sp.Ge(m, 0) for m in multipliers[len(equality_gradients) :]]
            if any(
                not c.is_Rational
                for e in equations
                for c in sp.Poly(e, *multipliers).coeffs()
            ):
                return None
            try:
                _, solution = lpmin(0, conditions)
            except (InfeasibleLPError, UnboundedLPError):
                return None
            solution = {m: solution.get(m, sp.S.Zero) for m in multipliers}
            if any(
                solution[m].is_nonnegative is not True
                for m in multipliers[len(equality_gradients) :]
            ) or any(truth(sp.Eq(e.subs(solution), 0)) is not True for e in equations):
                return None

    else:
        equations = tuple(gradient)
        solution = {}
        if any(truth(sp.Eq(g, 0)) is not True for g in gradient):
            return None
    return StructuralGlobalResult(
        "convex_kkt",
        value,
        (point,),
        True,
        metadata={
            "curvature": "convex",
            "multipliers": solution,
            "stationarity": tuple(sp.simplify(g.subs(solution)) for g in equations),
            "sufficient_conditions": True,
        },
    )


def try_convex_stationary(context):
    """Try affine stationarity and a few exact seeds before any nonlinear solving."""
    if curvature(context.objective, context) not in {"affine", "convex"}:
        return None
    gradient = [sp.diff(context.objective, v) for v in context.variables]
    stationary = linear_solution(gradient, context.variables)
    points = ([stationary] if stationary is not None else []) + list(context.seeds())
    for point in points:
        result = convex_kkt_certificate(context, point)
        if result is not None:
            return result
    return None


def try_concave_endpoints(context):
    """A concave function on a finite box has a minimum at a closure vertex."""
    from .box_rules import _best_corners

    corners = context.corners()
    if not corners or curvature(context.objective, context) not in {
        "affine",
        "concave",
    }:
        return None
    return _best_corners(
        context,
        "concave_vertices",
        corners,
        curvature="concave",
        domain=context.rectangle,
    )
