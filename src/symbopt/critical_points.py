from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import sympy as sp
from .degeneracy import classify_degenerate
from sympy.core.relational import Relational
from .relation_checks import normalize_relational


@dataclass
class CriticalPointResult:
    solution: Dict[sp.Symbol, sp.Expr]
    classification: str
    lagrangian: Optional[sp.Expr] = None
    hessian_at_point: Optional[sp.Matrix] = None
    eigenvalues: Optional[List[sp.Expr]] = None
    assumptions: Tuple[sp.Expr, ...] = ()

    def pretty(self) -> str:
        lines = [
            f"solution: {self.solution}",
            f"classification: {self.classification}",
        ]
        if self.lagrangian is not None:
            lines.append(f"lagrangian: {sp.sstr(self.lagrangian)}")
        if self.hessian_at_point is not None:
            lines.append(f"hessian_at_point: {sp.sstr(self.hessian_at_point)}")
        if self.eigenvalues is not None:
            lines.append(f"eigenvalues: {[sp.sstr(ev) for ev in self.eigenvalues]}")
        if self.assumptions:
            lines.append(f"assumptions: {[sp.sstr(a) for a in self.assumptions]}")
        return "\n".join(lines)


def normalize_constraints(constraints: Sequence[Relational]):
    eqs, ineqs_ge, ineqs_gt = [], [], []
    for rel in constraints:
        kind, residual = normalize_relational(rel)
        if kind == "eq":
            eqs.append(residual)
        elif kind == "ge":
            ineqs_ge.append(residual)
        elif kind == "gt":
            ineqs_gt.append(residual)
        else:
            raise TypeError(
                "Unequality constraints are not supported by critical_points."
            )
    return eqs, ineqs_ge, ineqs_gt


def _safe_simplify(expr: sp.Expr, assumptions: Sequence[sp.Expr]):
    out = sp.simplify(expr)
    if assumptions:
        try:
            out = sp.refine(out, sp.And(*assumptions))
        except (TypeError, ValueError, ArithmeticError, NotImplementedError):
            pass
    return sp.simplify(out)


def _is_positive(expr: sp.Expr, assumptions: Sequence[sp.Expr]) -> bool:
    expr = _safe_simplify(expr, assumptions)
    if expr.is_positive is True:
        return True
    try:
        return (
            sp.ask(
                sp.Q.positive(expr),
                assumptions=sp.And(*assumptions) if assumptions else True,
            )
            is True
        )
    except (TypeError, ValueError, ArithmeticError, NotImplementedError):
        return False


def _is_negative(expr: sp.Expr, assumptions: Sequence[sp.Expr]) -> bool:
    expr = _safe_simplify(expr, assumptions)
    if expr.is_negative is True:
        return True
    try:
        return (
            sp.ask(
                sp.Q.negative(expr),
                assumptions=sp.And(*assumptions) if assumptions else True,
            )
            is True
        )
    except (TypeError, ValueError, ArithmeticError, NotImplementedError):
        return False


def _is_zero(expr: sp.Expr, assumptions: Sequence[sp.Expr]) -> bool:
    expr = _safe_simplify(expr, assumptions)
    if expr.is_zero is True:
        return True
    try:
        return (
            sp.ask(
                sp.Q.zero(expr),
                assumptions=sp.And(*assumptions) if assumptions else True,
            )
            is True
        )
    except (TypeError, ValueError, ArithmeticError, NotImplementedError):
        return False


def _counts_by_sign(values: Sequence[sp.Expr], assumptions: Sequence[sp.Expr]):
    counts = {}
    for value in values:
        if _is_positive(value, assumptions):
            counts[1] = counts.get(1, 0) + 1
        elif _is_negative(value, assumptions):
            counts[-1] = counts.get(-1, 0) + 1
        elif _is_zero(value, assumptions):
            counts[0] = counts.get(0, 0) + 1
        else:
            counts[999] = counts.get(999, 0) + 1
    return counts


def classify_hessian(
    hessian: sp.Matrix,
    vars: Sequence[sp.Symbol],
    solution: Dict[sp.Symbol, sp.Expr],
    assumptions: Sequence[sp.Expr] = (),
    f: Optional[sp.Expr] = None,
):
    hpt = sp.Matrix(hessian.subs(solution))
    hpt = sp.Matrix(
        [
            [_safe_simplify(hpt[i, j], assumptions) for j in range(hpt.shape[1])]
            for i in range(hpt.shape[0])
        ]
    )
    try:
        evdict = hpt.eigenvals()
        eigenvalues = []
        for ev, mult in evdict.items():
            eigenvalues.extend([_safe_simplify(ev, assumptions)] * int(mult))
    except (TypeError, ValueError, ArithmeticError, NotImplementedError):
        return "Inconclusive", hpt, []
    counts = _counts_by_sign(eigenvalues, assumptions)
    n = len(vars)
    if counts.get(1, 0) == n:
        return "LocalMinimum", hpt, eigenvalues
    if counts.get(-1, 0) == n:
        return "LocalMaximum", hpt, eigenvalues
    if counts.get(1, 0) + counts.get(-1, 0) == n:
        return "SaddlePoint", hpt, eigenvalues
    if counts.get(0, 0) > 0:
        if f is not None:
            try:
                deg_class, _ = classify_degenerate(f, vars, solution, hpt)
                if deg_class is not None:
                    return deg_class, hpt, eigenvalues
            except (TypeError, ValueError, ArithmeticError, NotImplementedError):
                pass
        return "Degenerate", hpt, eigenvalues
    return "Inconclusive", hpt, eigenvalues


def _critical_points_free(f: sp.Expr, vars: Sequence[sp.Symbol], domain: str = "Reals"):
    vars = list(vars)
    grad = [sp.diff(f, v) for v in vars]
    try:
        sols = sp.solve([sp.Eq(g, 0) for g in grad], vars, dict=True)
    except (TypeError, ValueError, ArithmeticError, NotImplementedError):
        sols = []
    hessian = sp.hessian(f, vars)
    return [
        CriticalPointResult(
            solution=sol,
            classification=classify_hessian(hessian, vars, sol, ())[0],
            lagrangian=f,
            hessian_at_point=classify_hessian(hessian, vars, sol, ())[1],
            eigenvalues=classify_hessian(hessian, vars, sol, ())[2],
            assumptions=(),
        )
        for sol in sols
    ]


def _classify_constrained(
    lagrangian: sp.Expr,
    vars: Sequence[sp.Symbol],
    solution: Dict[sp.Symbol, sp.Expr],
    equalities: Sequence[sp.Expr],
    inequalities: Sequence[sp.Expr],
    multipliers: Sequence[sp.Symbol],
    assumptions: Sequence[sp.Expr],
    objective: sp.Expr,
):
    """Classify a constrained stationary point on the feasible tangent space."""
    hessian = sp.hessian(lagrangian, vars)
    hpt = sp.Matrix(hessian.subs(solution))
    hpt = sp.Matrix(
        [
            [_safe_simplify(hpt[i, j], assumptions) for j in range(hpt.cols)]
            for i in range(hpt.rows)
        ]
    )

    tangent_rows = []
    uncertain_active = False

    for constraint in equalities:
        gradient = [
            _safe_simplify(sp.diff(constraint, var).subs(solution), assumptions)
            for var in vars
        ]
        tangent_rows.append(gradient)

    for constraint, multiplier in zip(inequalities, multipliers):
        value = _safe_simplify(constraint.subs(solution), assumptions)
        if not _is_zero(value, assumptions):
            continue

        multiplier_value = _safe_simplify(
            solution.get(multiplier, multiplier),
            assumptions,
        )
        if _is_positive(multiplier_value, assumptions):
            gradient = [
                _safe_simplify(
                    sp.diff(constraint, var).subs(solution),
                    assumptions,
                )
                for var in vars
            ]
            tangent_rows.append(gradient)
        else:
            uncertain_active = True

    if not tangent_rows:
        classification, _, eigenvalues = classify_hessian(
            hessian,
            vars,
            solution,
            assumptions,
            f=objective,
        )
        if uncertain_active and classification in {"SaddlePoint", "LocalMaximum"}:
            classification = "Inconclusive"
        return classification, hpt, eigenvalues

    jacobian = sp.Matrix(tangent_rows)
    tangent_basis = jacobian.nullspace()
    if not tangent_basis:
        return "LocalMinimum", sp.zeros(0, 0), []

    basis = sp.Matrix.hstack(*tangent_basis)
    reduced = sp.simplify(basis.T * hpt * basis)
    try:
        evdict = reduced.eigenvals()
        eigenvalues = []
        for eigenvalue, multiplicity in evdict.items():
            eigenvalues.extend(
                [_safe_simplify(eigenvalue, assumptions)] * int(multiplicity)
            )
    except (TypeError, ValueError, ArithmeticError, NotImplementedError):
        return "Inconclusive", reduced, []

    counts = _counts_by_sign(eigenvalues, assumptions)
    size = reduced.rows
    if counts.get(1, 0) == size:
        return "LocalMinimum", reduced, eigenvalues
    if counts.get(-1, 0) == size:
        return (
            ("Inconclusive" if uncertain_active else "LocalMaximum"),
            reduced,
            eigenvalues,
        )
    if counts.get(1, 0) + counts.get(-1, 0) == size:
        return (
            ("Inconclusive" if uncertain_active else "SaddlePoint"),
            reduced,
            eigenvalues,
        )
    if counts.get(0, 0) > 0:
        return "Degenerate", reduced, eigenvalues
    return "Inconclusive", reduced, eigenvalues


def _critical_points_constr(
    f: sp.Expr,
    constraints: Sequence[Relational],
    vars: Sequence[sp.Symbol],
    domain: str = "Reals",
    lambda_prefix: str = "lam",
    mu_prefix: str = "mu",
    karush_john_multiplier: Optional[str] = None,
):
    vars = list(vars)
    eqs, ineqs_ge, ineqs_gt = normalize_constraints(constraints)
    ineqs = list(ineqs_ge) + list(ineqs_gt)
    strict_assumptions = [sp.Gt(h, 0) for h in ineqs_gt]

    lam = [sp.Symbol(f"{lambda_prefix}_{i + 1}", real=True) for i in range(len(eqs))]
    mu = [
        sp.Symbol(f"{mu_prefix}_{j + 1}", real=True, nonnegative=True)
        for j in range(len(ineqs))
    ]
    rho = (
        sp.Symbol(str(karush_john_multiplier), real=True, nonnegative=True)
        if karush_john_multiplier is not None
        else None
    )

    if rho is not None:
        L = (
            rho * f
            + sum(mult * residual for mult, residual in zip(lam, eqs))
            - sum(m * h for m, h in zip(mu, ineqs))
        )
        allvars = [rho] + lam + mu + vars
    else:
        L = (
            f
            + sum(mult * residual for mult, residual in zip(lam, eqs))
            - sum(m * h for m, h in zip(mu, ineqs))
        )
        allvars = lam + mu + vars

    equations = []
    equations.extend([sp.Eq(sp.expand(sp.diff(L, v)), 0) for v in vars])
    equations.extend([sp.Eq(g, 0) for g in eqs])
    equations.extend([sp.Eq(sp.expand(m * h), 0) for m, h in zip(mu, ineqs)])
    if rho is not None:
        kj_multipliers = [rho, *lam, *mu]
        equations.append(
            sp.Eq(
                sp.expand(sum(multiplier**2 for multiplier in kj_multipliers)),
                1,
            )
        )

    try:
        raw_solutions = sp.solve(equations, allvars, dict=True)
    except (TypeError, ValueError, ArithmeticError, NotImplementedError):
        raw_solutions = []

    sols = []
    for solution in raw_solutions:
        if any(var not in solution for var in vars):
            continue
        feasible = True
        for constraint in ineqs_ge:
            value = _safe_simplify(constraint.subs(solution), ())
            if not (value.is_nonnegative is True or _is_zero(value, ())):
                feasible = False
                break
        if not feasible:
            continue
        for constraint in ineqs_gt:
            value = _safe_simplify(constraint.subs(solution), ())
            if value.is_positive is not True:
                feasible = False
                break
        if not feasible:
            continue
        for multiplier in mu:
            value = _safe_simplify(solution.get(multiplier, multiplier), ())
            if value.is_nonnegative is not True:
                feasible = False
                break
        if rho is not None:
            value = _safe_simplify(solution.get(rho, rho), ())
            if value.is_nonnegative is not True:
                feasible = False
        if feasible:
            sols.append(solution)

    assumptions = tuple(
        list(constraints)
        + strict_assumptions
        + ([sp.Ge(rho, 0)] if rho is not None else [])
    )
    out = []
    for sol in sols:
        if rho is not None:
            kj_multipliers = [rho, *lam, *mu]
            if all(
                multiplier in sol and sp.simplify(sol[multiplier]) == 0
                for multiplier in kj_multipliers
            ):
                continue

        classification, hpt, evs = _classify_constrained(
            L,
            vars,
            sol,
            eqs,
            ineqs,
            mu,
            assumptions,
            f,
        )
        out.append(
            CriticalPointResult(
                solution=sol,
                classification=classification,
                lagrangian=L,
                hessian_at_point=hpt,
                eigenvalues=evs,
                assumptions=assumptions,
            )
        )
    return out


def critical_points(
    problem,
    vars: Sequence[sp.Symbol],
    domain: str = "Reals",
    lambda_prefix: str = "lam",
    mu_prefix: str = "mu",
    karush_john_multiplier: Optional[str] = None,
):
    """Compute and classify symbolic critical points.

    The function handles unconstrained objectives and constrained ``(f, cons)``
    pairs. Constrained classification uses the feasible tangent space when
    available and falls back to an inconclusive result rather than overstating a
    degenerate case."""
    if isinstance(problem, (list, tuple)) and len(problem) == 2:
        f, constraints = problem
        if constraints in (True, None, [], ()):
            return _critical_points_free(f, vars, domain=domain)
        return _critical_points_constr(
            f,
            constraints,
            vars,
            domain=domain,
            lambda_prefix=lambda_prefix,
            mu_prefix=mu_prefix,
            karush_john_multiplier=karush_john_multiplier,
        )
    return _critical_points_free(problem, vars, domain=domain)


def check_second_order(
    problem,
    vars: Sequence[sp.Symbol],
    solution: Dict[sp.Symbol, sp.Expr],
    target: str = "Min",
) -> bool:
    results = critical_points(problem, vars)
    matches = []
    for r in results:
        ok = True
        for k, v in solution.items():
            if k not in r.solution or sp.simplify(r.solution[k] - v) != 0:
                ok = False
                break
        if ok:
            matches.append(r)
    if not matches:
        return False
    if target.lower().startswith("min"):
        return any(r.classification == "LocalMinimum" for r in matches)
    if target.lower().startswith("max"):
        return any(r.classification == "LocalMaximum" for r in matches)
    return False


def check_kkt(
    problem,
    vars: Sequence[sp.Symbol],
    point_rules: Dict[sp.Symbol, sp.Expr],
    domain: str = "Reals",
    lambda_prefix: str = "lam",
    mu_prefix: str = "mu",
    karush_john_multiplier: Optional[str] = None,
) -> bool:
    f, constraints = problem
    critpts = _critical_points_constr(
        f,
        constraints,
        vars,
        domain=domain,
        lambda_prefix=lambda_prefix,
        mu_prefix=mu_prefix,
        karush_john_multiplier=karush_john_multiplier,
    )
    if not critpts:
        return False
    matches = []
    for cp in critpts:
        ok = True
        for v in vars:
            if (
                v not in point_rules
                or v not in cp.solution
                or sp.simplify(cp.solution[v] - point_rules[v]) != 0
            ):
                ok = False
                break
        if ok:
            matches.append(cp)
    if not matches:
        return False
    return all(
        m.classification in ("LocalMinimum", "FlatMinimumCandidate", "Inconclusive")
        for m in matches
    )
