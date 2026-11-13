from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import sympy as sp
from sympy.core.relational import Relational

from .algebraic_numbers import algebraic_compare
from .lp import solve_lp
from .numeric_utils import eval_float
from .relation_checks import normalize_relational, relation_holds
from .structural_results import StructuralGlobalResult
from .symbolic_cache import affine_coefficients as _affine_coefficients


def _symbolic_linear_rels(constraints, variables):
    rows = []
    for relation in constraints:
        if not isinstance(relation, Relational):
            return None
        kind, residual = normalize_relational(relation)
        if kind not in {"eq", "ge", "gt"}:
            return None
        data = _affine_coefficients(residual, variables)
        if data is None:
            return None
        coeffs, constant = data
        if any(value.free_symbols for value in (*coeffs, constant)):
            return None
        rows.append((kind, coeffs, constant))
    return rows


def _linear_arrays(rows):
    inequalities = []
    ineq_rhs = []
    equalities = []
    eq_rhs = []
    for kind, coeffs, constant in rows:
        numeric_coeffs = [eval_float(value) for value in coeffs]
        numeric_constant = eval_float(constant)
        if kind in {"ge", "gt"}:
            inequalities.append([-value for value in numeric_coeffs])
            ineq_rhs.append(numeric_constant)
        else:
            equalities.append(numeric_coeffs)
            eq_rhs.append(-numeric_constant)
    return (
        np.asarray(inequalities, dtype=float) if inequalities else None,
        np.asarray(ineq_rhs, dtype=float) if ineq_rhs else None,
        np.asarray(equalities, dtype=float) if equalities else None,
        np.asarray(eq_rhs, dtype=float) if eq_rhs else None,
    )


def _exact_relation_checks(constraints, point):
    for relation in constraints:
        value = sp.simplify(relation.subs(point))
        if value is sp.S.true or value is True:
            continue
        if value is sp.S.false or value is False:
            return False
        if not relation_holds(relation, point, tol=1e-10):
            return False
    return True


def _certify_linear_optimum(objective, constraints, variables, point) -> bool:
    """Certify a feasible LP point by exact KKT/dual multipliers."""
    if not _exact_relation_checks(constraints, point):
        return False
    equalities = []
    active_inequalities = []
    for relation in constraints:
        kind, residual = normalize_relational(relation)
        value = sp.simplify(residual.subs(point))
        if kind == "eq":
            equalities.append(residual)
        elif kind == "ge" and value == 0:
            active_inequalities.append(residual)
        elif kind != "ge":
            return False
    lam = tuple(sp.Dummy(f"lam{i}", real=True) for i in range(len(equalities)))
    mu = tuple(sp.Dummy(f"mu{i}", real=True) for i in range(len(active_inequalities)))
    lagrangian = objective
    for symbol, residual in zip(lam, equalities):
        lagrangian += symbol * residual
    for symbol, residual in zip(mu, active_inequalities):
        lagrangian -= symbol * residual
    equations = [sp.Eq(sp.diff(lagrangian, var), 0) for var in variables]
    try:
        solutions = sp.solve(equations, [*lam, *mu], dict=True, simplify=True)
    except (NotImplementedError, TypeError, ValueError):
        return False
    if not equations and not mu:
        return True
    for solution in solutions:
        if any(symbol not in solution for symbol in mu):
            continue
        if all(
            sp.ask(sp.Q.nonnegative(sp.simplify(solution[symbol]))) is True
            for symbol in mu
        ):
            return True
    return False


def _affine_extreme(expr, constraints, variables, maximize=False):
    rows = _symbolic_linear_rels(constraints, variables)
    objective_data = _affine_coefficients(expr, variables)
    if rows is None or objective_data is None:
        return None
    coeffs, constant = objective_data
    if any(value.free_symbols for value in (*coeffs, constant)):
        return None
    A, b, Aeq, beq = _linear_arrays(rows)
    result = solve_lp(
        np.asarray([eval_float(value) for value in coeffs], dtype=float),
        A_ub=A,
        b_ub=b,
        A_eq=Aeq,
        b_eq=beq,
        maximize=maximize,
        bounds=[(None, None)] * len(variables),
    )
    if result.status != "optimal":
        return result.status, None, None
    point = {
        var: sp.nsimplify(value, rational=True)
        for var, value in zip(variables, result.x)
    }
    if not _exact_relation_checks(constraints, point):
        return "numeric_only", None, None
    certified_objective = -expr if maximize else expr
    if not _certify_linear_optimum(certified_objective, constraints, variables, point):
        return "numeric_only", None, None
    value = sp.simplify(expr.subs(point))
    return "optimal", value, point


def _strict_lp_feasible(constraints, positive_forms):
    """Decide rational open-domain feasibility with an exact bounded margin LP.

    Return None when the installed SymPy lacks the rational LP backend or the
    coefficients are outside its exact rational scope.
    """
    try:
        from sympy.solvers.simplex import InfeasibleLPError, UnboundedLPError, lpmax
    except ImportError:
        return None
    margin = sp.Dummy("strict_margin", real=True)
    conditions = [*constraints, margin >= 0, margin <= 1]
    conditions.extend(form >= margin for form in positive_forms)
    symbols = tuple(
        sorted(
            set().union(*(condition.free_symbols for condition in conditions)),
            key=sp.default_sort_key,
        )
    )
    rows = _symbolic_linear_rels(conditions, symbols)
    if rows is None or not all(
        value.is_Rational for _, row, constant in rows for value in (*row, constant)
    ):
        return None
    try:
        value, _ = lpmax(margin, conditions)
    except InfeasibleLPError:
        return False
    except (UnboundedLPError, TypeError, ValueError):
        return None
    return value.is_positive


def _fractional_branch(
    objective: sp.Expr,
    constraints: Sequence[Relational],
    variables: Sequence[sp.Symbol],
    sign: int,
    *,
    split_branch: bool,
) -> StructuralGlobalResult | None:
    """Solve one denominator-sign branch on the closure via Charnes--Cooper."""
    numerator, denominator = map(sp.expand, sp.fraction(sp.cancel(objective)))
    numerator_data = _affine_coefficients(numerator, variables)
    denominator_data = _affine_coefficients(denominator, variables)
    if numerator_data is None or denominator_data is None:
        return None
    branch_constraints = list(constraints)
    if split_branch:
        # Use the closure of the strict sign branch. denominator_norm == 1 below
        # excludes the singular hyperplane for finite transformed points; t == 0
        # represents an asymptotic boundary/recession value.
        branch_constraints.append(sp.Ge(sign * denominator, 0))
    rows = _symbolic_linear_rels(branch_constraints, variables)
    if rows is None:
        return None

    ncoeff, nconst = numerator_data
    dcoeff, dconst = denominator_data
    if sign < 0:
        ncoeff = tuple(-value for value in ncoeff)
        nconst = -nconst
        dcoeff = tuple(-value for value in dcoeff)
        dconst = -dconst

    yvars = tuple(
        sp.Dummy(f"fractional_y{i}", real=True) for i in range(len(variables))
    )
    scale = sp.Dummy("fractional_scale", real=True)
    transformed_constraints = []
    positive_forms = [scale]
    for kind, coeffs, constant in rows:
        lhs = sp.Add(*[coeff * y for coeff, y in zip(coeffs, yvars)]) + constant * scale
        transformed_constraints.append(sp.Eq(lhs, 0) if kind == "eq" else sp.Ge(lhs, 0))
        if kind == "gt":
            positive_forms.append(lhs)
    denominator_norm = (
        sp.Add(*[coeff * y for coeff, y in zip(dcoeff, yvars)]) + dconst * scale
    )
    transformed_constraints.append(sp.Eq(denominator_norm, 1))
    transformed_constraints.append(sp.Ge(scale, 0, evaluate=False))
    transformed_objective = (
        sp.Add(*[coeff * y for coeff, y in zip(ncoeff, yvars)]) + nconst * scale
    )

    all_tvars = (*yvars, scale)
    trows = _symbolic_linear_rels(transformed_constraints, all_tvars)
    if trows is None:
        return None
    A, b, Aeq, beq = _linear_arrays(trows)
    objective_data = _affine_coefficients(transformed_objective, all_tvars)
    if objective_data is None:
        return None
    coeffs, constant = objective_data
    lp = solve_lp(
        np.asarray([eval_float(value) for value in coeffs], dtype=float),
        A_ub=A,
        b_ub=b,
        A_eq=Aeq,
        b_eq=beq,
        maximize=False,
        bounds=[(None, None)] * len(yvars) + [(0, None)],
    )
    exact_point = None
    if lp.status == "unbounded":
        try:
            from sympy.solvers.simplex import InfeasibleLPError, UnboundedLPError, lpmin
        except ImportError:
            return None
        # Floating-point solvers can discard very small coefficients. Their
        # status alone is not a recession certificate for the exact problem.
        if not all(
            value.is_Rational
            for _, row, constant in trows
            for value in (*row, constant)
        ):
            return None
        if not all(value.is_Rational for value in (*coeffs, constant)):
            return None
        try:
            _, exact_point = lpmin(transformed_objective, transformed_constraints)
        except UnboundedLPError:
            if _strict_lp_feasible(transformed_constraints, positive_forms) is not True:
                return None
            return StructuralGlobalResult(
                kind="linear_fractional",
                value=-sp.oo,
                points=(),
                attained=False,
                certified=True,
                metadata={
                    "transformation": "Charnes-Cooper",
                    "denominator_sign": sign,
                    "denominator_sign_split": split_branch,
                    "asymptotic": True,
                    "lp_status": "unbounded",
                    "unboundedness_proof": "exact rational LP and open-domain feasibility",
                },
            )
        except (InfeasibleLPError, TypeError, ValueError):
            return None
    elif lp.status != "optimal":
        return None

    transformed_point = (
        {var: exact_point.get(var, sp.S.Zero) for var in all_tvars}
        if exact_point is not None
        else {
            var: sp.nsimplify(value, rational=True)
            for var, value in zip(all_tvars, lp.x)
        }
    )
    active = []
    for rel in transformed_constraints:
        kind, residual = normalize_relational(rel)
        if kind == "eq" or abs(eval_float(residual.subs(transformed_point))) <= 1e-8:
            active.append(sp.Eq(residual, 0))
    try:
        sols = sp.solve(active, all_tvars, dict=True, simplify=True)
    except (NotImplementedError, ValueError, TypeError):
        sols = []
    best_exact = transformed_point
    for solution in sols:
        if all(
            var in solution and not solution[var].free_symbols.intersection(all_tvars)
            for var in all_tvars
        ):
            candidate = {var: sp.simplify(solution[var]) for var in all_tvars}
            if _exact_relation_checks(transformed_constraints, candidate):
                best_exact = candidate
                break

    tvalue = sp.simplify(best_exact[scale])
    value = sp.simplify(transformed_objective.subs(best_exact))
    exact_transformed = _exact_relation_checks(transformed_constraints, best_exact)
    if not exact_transformed or not _certify_linear_optimum(
        transformed_objective, transformed_constraints, all_tvars, best_exact
    ):
        return None

    metadata = {
        "transformation": "Charnes-Cooper",
        "denominator_sign": sign,
        "denominator_sign_split": split_branch,
        "transformed_scale": sp.sstr(tvalue),
    }
    if tvalue == 0:
        if _strict_lp_feasible(transformed_constraints, positive_forms) is not True:
            return None
        metadata["asymptotic"] = True
        return StructuralGlobalResult(
            kind="linear_fractional",
            value=value,
            points=(),
            attained=False,
            certified=True,
            metadata=metadata,
        )
    if sp.ask(sp.Q.positive(tvalue)) is not True:
        return None

    point = {
        original: sp.simplify(best_exact[y] / tvalue)
        for original, y in zip(variables, yvars)
    }
    original_value = sp.simplify(objective.subs(point))
    if original_value.has(sp.zoo, sp.nan) or original_value in (sp.oo, -sp.oo):
        return None
    if sp.simplify(original_value - value) != 0:
        return None
    attained = _exact_relation_checks(constraints, point)
    if split_branch:
        den_value = sp.simplify(denominator.subs(point))
        attained = attained and (sp.ask(sp.Q.positive(sign * den_value)) is True)
    if not attained and not _strict_lp_feasible(
        transformed_constraints, positive_forms
    ):
        return None
    return StructuralGlobalResult(
        kind="linear_fractional",
        value=original_value,
        points=(point,) if attained else (),
        attained=attained,
        certified=True,
        metadata=metadata,
    )


def _smaller_structural(left, right):
    if left is None:
        return right
    if right is None:
        return left
    if left.value is -sp.oo or right.value is sp.oo:
        return left
    if right.value is -sp.oo or left.value is sp.oo:
        return right
    comparison = algebraic_compare(left.value, right.value)
    if comparison is None:
        return None
    if comparison < 0:
        return left
    if comparison > 0:
        return right
    points = (*left.points, *right.points)
    return StructuralGlobalResult(
        kind="linear_fractional",
        value=left.value,
        points=points,
        attained=left.attained or right.attained,
        certified=True,
        metadata={
            "denominator_sign_split": True,
            "branches": (left.metadata, right.metadata),
        },
    )


def _implied_denominator_exclusion(relation, denominator, variables):
    """Remove only exclusions proved to follow from the objective denominator."""
    if not isinstance(relation, sp.Unequality):
        return False
    residual = sp.simplify(relation.lhs - relation.rhs)
    if residual == 0 or residual.free_symbols - set(variables):
        return False
    try:
        quotient = sp.Poly(sp.cancel(denominator / residual), *variables)
    except sp.PolynomialError:
        return False
    # A formal polynomial with an infinite or unresolved coefficient cannot
    # establish divisibility over a field (for example a parameter at zero).
    return all(c.is_finite is True for c in quotient.coeffs())


def try_linear_fractional(
    objective: sp.Expr,
    constraints: Sequence[Relational],
    variables: Sequence[sp.Symbol],
) -> StructuralGlobalResult | None:
    """Solve affine linear-fractional programs, splitting denominator signs when needed."""
    variables = tuple(variables)
    if not variables:
        return None
    numerator, denominator = map(sp.expand, sp.fraction(sp.cancel(objective)))
    if denominator == 1 or not denominator.free_symbols.intersection(variables):
        return None
    numerator_data = _affine_coefficients(numerator, variables)
    denominator_data = _affine_coefficients(denominator, variables)
    constraints = tuple(
        r
        for r in constraints
        if not _implied_denominator_exclusion(r, denominator, variables)
    )
    rows = _symbolic_linear_rels(constraints, variables)
    if numerator_data is None or denominator_data is None or rows is None:
        return None
    if any(
        value.free_symbols
        for value in (
            *numerator_data[0],
            numerator_data[1],
            *denominator_data[0],
            denominator_data[1],
        )
    ):
        return None

    # Preserve the cheap fixed-sign route first. Strict constraints are optimized
    # on their closure, with attainment checked against the original relations.
    min_status, den_min, _ = _affine_extreme(
        denominator,
        [
            sp.Ge(r.lhs, r.rhs)
            if isinstance(r, sp.StrictGreaterThan)
            else sp.Le(r.lhs, r.rhs)
            if isinstance(r, sp.StrictLessThan)
            else r
            for r in constraints
        ],
        variables,
    )
    max_status, den_max, _ = _affine_extreme(
        denominator,
        [
            sp.Ge(r.lhs, r.rhs)
            if isinstance(r, sp.StrictGreaterThan)
            else sp.Le(r.lhs, r.rhs)
            if isinstance(r, sp.StrictLessThan)
            else r
            for r in constraints
        ],
        variables,
        maximize=True,
    )
    if (
        min_status == "optimal"
        and den_min is not None
        and sp.ask(sp.Q.positive(den_min)) is True
    ):
        return _fractional_branch(
            objective, constraints, variables, 1, split_branch=False
        )
    if (
        max_status == "optimal"
        and den_max is not None
        and sp.ask(sp.Q.negative(den_max)) is True
    ):
        return _fractional_branch(
            objective, constraints, variables, -1, split_branch=False
        )

    positive = _fractional_branch(
        objective, constraints, variables, 1, split_branch=True
    )
    negative = _fractional_branch(
        objective, constraints, variables, -1, split_branch=True
    )
    # A failed solve is different from a proved empty sign branch. Either
    # nonempty branch can lower the global infimum, unless -infinity is proved.
    if not any(
        branch is not None and branch.value is -sp.oo for branch in (positive, negative)
    ):
        for sign, branch in ((1, positive), (-1, negative)):
            if branch is not None:
                continue
            closure = []
            positive_forms = [sign * denominator]
            for kind, coeffs, constant in rows:
                residual = sum(c * v for c, v in zip(coeffs, variables)) + constant
                closure.append(
                    sp.Eq(residual, 0) if kind == "eq" else sp.Ge(residual, 0)
                )
                if kind == "gt":
                    positive_forms.append(residual)
            if _strict_lp_feasible(closure, positive_forms) is not False:
                return None
        if positive is None and negative is None:
            # The nonzero denominator partitions every source point into these
            # two sign branches. Both open-domain margin LPs proved emptiness.
            return StructuralGlobalResult(
                kind="linear_fractional",
                value=sp.oo,
                points=(),
                attained=False,
                certified=True,
                metadata={
                    "denominator_sign_split": True,
                    "infeasibility_proof": "exact rational margin LPs on both sign branches",
                },
            )
    result = _smaller_structural(positive, negative)
    if result is not None and result.metadata.get("denominator_sign_split") is not True:
        result = StructuralGlobalResult(
            kind=result.kind,
            value=result.value,
            points=result.points,
            attained=result.attained,
            certified=result.certified,
            metadata={**dict(result.metadata), "denominator_sign_split": True},
        )
    return result
