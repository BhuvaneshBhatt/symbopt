"""Leading-term, visible positivity and classical-inequality certificates."""

from dataclasses import replace
from itertools import product

import sympy as sp

from .rule_utils import affine_data, linear_solution, truth
from .structural_results import StructuralGlobalResult


def leading_term(expression, variable):
    """Find a rational asymptotic leading term without expanding integer powers."""
    if not expression.has(variable):
        return (0, expression) if expression.is_real is True else None
    if expression == variable:
        return 1, sp.S.One
    if expression.is_Add or expression.is_Mul:
        terms = [leading_term(arg, variable) for arg in expression.args]
        if any(t is None for t in terms):
            return None
        if expression.is_Mul:
            return sum(t[0] for t in terms), sp.Mul(*(t[1] for t in terms))
        degree = max(t[0] for t in terms)
        coefficient = sp.simplify(sum(c for d, c in terms if d == degree))
        if coefficient.is_zero is not False:
            return None
        return degree, coefficient
    if expression.is_Pow and expression.exp.is_Integer:
        base = leading_term(expression.base, variable)
        if base is not None and base[1].is_zero is False:
            return base[0] * int(expression.exp), base[1] ** expression.exp
    return None


def try_unbounded_ray(context):
    """Prove escape to minus infinity along an eventually feasible affine ray."""
    t = sp.Dummy("ray", positive=True)
    directions = []
    for position in range(len(context.variables)):
        directions.append(
            tuple(int(i == position) for i in range(len(context.variables)))
        )
    if len(context.variables) > 1:
        directions.append((1,) * len(context.variables))
    for direction, sign in product(directions, (1, -1)):
        path = {v: sign * d * t for v, d in zip(context.variables, direction)}
        expression = context.objective.subs(path, simultaneous=True)
        lead = leading_term(expression, t)
        if lead is None or lead[0] <= 0 or lead[1].is_negative is not True:
            continue
        feasible = True
        for relation in context.constraints:
            residual = relation.lhs - relation.rhs
            if isinstance(relation, (sp.LessThan, sp.StrictLessThan)):
                residual = -residual
            reduced = sp.cancel(residual.subs(path, simultaneous=True))
            if reduced == 0:
                holds = isinstance(relation, (sp.Equality, sp.GreaterThan, sp.LessThan))
            else:
                data = leading_term(reduced, t)
                holds = data is not None and (
                    isinstance(relation, sp.Unequality)
                    and data[1].is_zero is False
                    or not isinstance(relation, (sp.Equality, sp.Unequality))
                    and data[1].is_positive is True
                )
            if not holds:
                feasible = False
                break
        if feasible:
            return StructuralGlobalResult(
                "leading_ray",
                -sp.oo,
                metadata={
                    "path_parameter": t,
                    "path": path,
                    "leading_degree": lead[0],
                    "leading_coefficient": lead[1],
                    "eventually_feasible": True,
                },
            )
    return None


def try_nonnegative_sum(context):
    """Match a visible lower bound and verify a simultaneous attaining point."""
    offset = sp.S.Zero
    zeros = []
    for term in sp.Add.make_args(context.objective):
        if not term.free_symbols.intersection(context.variables):
            offset += term
            continue
        coefficient, dependent = term.as_independent(*context.variables, as_Add=False)
        if coefficient.is_nonnegative is not True:
            return None
        if (
            dependent.is_Pow
            and dependent.exp.is_Integer
            and dependent.exp > 0
            and dependent.exp % 2 == 0
        ):
            base = dependent.base
        elif dependent.func is sp.Abs:
            base = dependent.args[0]
        elif context.sign(dependent) in {0, 1, 2}:
            base = dependent
        else:
            return None
        if not context.real(base):
            return None
        zeros.append(base)
    if not zeros or offset.is_real is not True:
        return None
    equations = [g for g in zeros if affine_data(g, context.variables) is not None]
    equations += [
        r.lhs - r.rhs for r in context.constraints if isinstance(r, sp.Equality)
    ]
    point = linear_solution(equations, context.variables)
    points = ([point] if point is not None else []) + list(context.seeds())
    found = []
    proof = None
    for point in points:
        if all(
            truth(sp.Eq(g.subs(point, simultaneous=True), 0)) is True for g in zeros
        ):
            result = context.result(
                "nonnegative_sum",
                point,
                lower_bound=offset,
                zero_conditions=tuple(zeros),
                affine_zero_conditions=all(
                    affine_data(g, context.variables) is not None for g in zeros
                ),
            )
            if result is not None and truth(sp.Eq(result.value, offset)) is True:
                proof = result
                if result.points[0] not in found:
                    found.append(result.points[0])
    return replace(proof, points=tuple(found)) if proof is not None else None


def try_positive_inequality(context):
    """Recognize weighted two-power AM-GM and reciprocal allocation patterns."""
    if len(context.variables) == 1 and context.rectangle:
        x = context.variables[0]
        interval = context.rectangle[x]
        if interval.left != 0 or not interval.left_open or interval.right != sp.oo:
            return None
        offset = sp.S.Zero
        powers = []
        for term in sp.Add.make_args(sp.expand(context.objective)):
            exponent = term.as_powers_dict().get(x, sp.S.Zero)
            coefficient = sp.simplify(term / x**exponent)
            if coefficient.has(x) or coefficient.is_real is not True:
                return None
            if exponent == 0:
                offset += coefficient
            else:
                powers.append((coefficient, exponent))
        if len(powers) == 2:
            positive = next(((a, p) for a, p in powers if p.is_positive is True), None)
            negative = next(((b, q) for b, q in powers if q.is_negative is True), None)
            if positive and negative:
                a, p = positive
                b, minus_q = negative
                q = -minus_q
                if a.is_positive is True and b.is_positive is True:
                    point = {x: (b * q / (a * p)) ** (1 / (p + q))}
                    return context.result(
                        "weighted_amgm",
                        point,
                        positive_powers=(p, q),
                        coefficients=(a, b),
                    )
    if len(context.variables) < 2:
        return None
    objective = sp.expand(context.objective)
    coefficients = []
    residual = objective
    for x in context.variables:
        coefficient = objective.coeff(x, -1)
        if (
            coefficient.free_symbols.intersection(context.variables)
            or coefficient.is_positive is not True
        ):
            return None
        coefficients.append(coefficient)
        residual -= coefficient / x
    if residual.free_symbols.intersection(context.variables):
        return None
    # All inequalities must give positivity; the single equality fixes total mass.
    equalities = [r for r in context.constraints if isinstance(r, sp.Equality)]
    if len(equalities) != 1:
        return None
    data = affine_data(equalities[0].lhs - equalities[0].rhs, context.variables)
    if (
        data is None
        or not data[0]
        or any(c != data[0][0] for c in data[0])
        or data[0][0] == 0
    ):
        return None
    total = sp.simplify(-data[1] / data[0][0])
    if total.is_positive is not True:
        return None
    # Positivity is checked from explicit axis constraints, independently of equality.
    remaining = tuple(r for r in context.constraints if not isinstance(r, sp.Equality))
    from .rule_utils import RuleContext

    domain = RuleContext(context.objective, remaining, context.variables).rectangle
    if domain is None or any(
        i.left != 0 or not i.left_open or i.right != sp.oo for i in domain.values()
    ):
        return None
    roots = [sp.sqrt(a) for a in coefficients]
    point = {x: total * root / sum(roots) for x, root in zip(context.variables, roots)}
    return context.result(
        "cauchy_schwarz", point, total=total, positive_coefficients=tuple(coefficients)
    )
