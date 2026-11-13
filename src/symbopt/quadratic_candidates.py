"""Feasible exact incumbents from inexpensive sampling and numerical proposals."""

from itertools import product

import sympy as sp

from .rational_optimization import _compare


def improve_incumbent(problem, points, incumbent=()):
    """Accept proposals only after exact original-domain substitution checks."""
    point = incumbent
    value = problem.value(point) if point else sp.oo
    for candidate in points:
        candidate = tuple(map(sp.sympify, candidate))
        if not problem.feasible(candidate):
            continue
        proposed = problem.value(candidate)
        if _compare(proposed, value) < 0:
            point, value = candidate, proposed
    return point


def sampled_points(problem, bounds=None):
    """Generate a small coordinate grid and exact roots of sliced quadratic equalities."""
    n = len(problem.variables)
    if bounds is None:
        bases = [(sp.S.Zero,) * n]
        for scale in (1, 2, 4, 8):
            bases += [
                tuple(sp.S(sign * scale) if i == j else sp.S.Zero for i in range(n))
                for j in range(n)
                for sign in (-1, 1)
            ]
            bases += [(sp.S(scale),) * n, (-sp.S(scale),) * n]
    else:
        midpoint = tuple((a + b) / 2 for a, b in bounds)
        bases = [midpoint, tuple(a for a, b in bounds), tuple(b for a, b in bounds)]
        for i, (a, b) in enumerate(bounds):
            for endpoint in (a, b, sp.S.Zero if a <= 0 <= b else (a + b) / 2):
                bases.append(
                    tuple(endpoint if j == i else x for j, x in enumerate(midpoint))
                )
        if n <= 4:
            bases += list(product(*bounds))
    yield from dict.fromkeys(bases)
    equalities = [
        r.lhs - r.rhs for r in problem.constraints if isinstance(r, sp.Equality)
    ]
    yield from _equality_points(problem.variables, equalities, bases[: 1 + 2 * n])


def _equality_points(variables, equalities, points):
    """Recover exact real coordinates from quadratic equality slices.

    Other coordinates are fixed at each proposal. This produces candidates,
    whose full feasibility is checked by the caller; identically zero slices
    impose no particular coordinate and produce no additional proposal.
    """
    seen_slices = set()
    for point in points:
        for expression in equalities:
            for i, variable in enumerate(variables):
                # The coordinate being solved for is discarded. Proposals
                # differing only in that coordinate define the same slice.
                key = (expression, variable, point[:i] + point[i + 1 :])
                if key in seen_slices:
                    continue
                seen_slices.add(key)
                substitutions = {
                    v: p for j, (v, p) in enumerate(zip(variables, point)) if j != i
                }
                poly = sp.Poly(expression.subs(substitutions), variable)
                a, b, c = (poly.nth(k) for k in (2, 1, 0))
                if a == 0:
                    roots = (-c / b,) if b != 0 else ()
                else:
                    discriminant = b * b - 4 * a * c
                    if discriminant.is_nonnegative is not True:
                        roots = ()
                    elif discriminant == 0:
                        roots = (-b / (2 * a),)
                    else:
                        radical = sp.sqrt(discriminant)
                        denominator = 2 * a
                        roots = (
                            (-b - radical) / denominator,
                            (-b + radical) / denominator,
                        )
                for root in roots:
                    yield tuple(
                        root if j == i else value for j, value in enumerate(point)
                    )


def numerical_proposals(problem, bounds=None, seed=()):
    """Use SLSQP only to propose points; rational/algebraic checks decide acceptance."""
    import numpy as np
    from scipy.optimize import minimize

    variables = problem.variables
    if not seed:
        seed = (
            tuple((a + b) / 2 for a, b in bounds)
            if bounds
            else (sp.S.Zero,) * len(variables)
        )
    objective = sp.lambdify(variables, problem.objective, "numpy")
    gradient = sp.lambdify(
        variables, [sp.diff(problem.objective, v) for v in variables], "numpy"
    )
    conditions = []
    for relation in problem.constraints:
        residual = relation.lhs - relation.rhs
        if isinstance(relation, sp.LessThan):
            residual = -residual
        function = sp.lambdify(variables, residual, "numpy")
        derivative = sp.lambdify(
            variables, [sp.diff(residual, v) for v in variables], "numpy"
        )
        conditions.append(
            {
                "type": "eq" if isinstance(relation, sp.Equality) else "ineq",
                "fun": lambda x, f=function: float(f(*x)),
                "jac": lambda x, g=derivative: np.asarray(g(*x), dtype=float),
            }
        )
    try:
        result = minimize(
            lambda x: float(objective(*x)),
            np.array([float(v) for v in seed]),
            jac=lambda x: np.asarray(gradient(*x), dtype=float),
            constraints=conditions,
            bounds=[(float(a), float(b)) for a, b in bounds] if bounds else None,
            method="SLSQP",
            options={"maxiter": 80, "ftol": 1e-12},
        )
    except (ValueError, TypeError, ArithmeticError):
        return
    if not np.all(np.isfinite(result.x)):
        return
    rounded = tuple(sp.Rational(float(v)).limit_denominator(10**6) for v in result.x)
    yield rounded
    if seed:
        for denominator in (10**4, 10**6, 10**8):
            yield tuple(
                (1 - sp.Rational(1, denominator)) * v + sp.Rational(1, denominator) * p
                for v, p in zip(rounded, seed)
            )
    # Recover a real algebraic coordinate when a rational proposal is close to a
    # quadratic equality; this preserves exact feasibility on curved boundaries.
    equalities = [
        r.lhs - r.rhs for r in problem.constraints if isinstance(r, sp.Equality)
    ]
    yield from _equality_points(variables, equalities, (rounded,))
