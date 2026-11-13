from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import sympy as sp
from sympy.core.relational import Relational

from ._symbols import fresh_symbols
from .degeneracy import classify_degenerate
from .relation_checks import normalize_relational
from .semialgebraic import exact_feasible, prove_nonnegative_exact


@dataclass
class CriticalPointResult:
    """A stationary candidate and its local curvature classification and assumptions."""

    solution: dict[sp.Symbol, sp.Expr]
    classification: str
    lagrangian: sp.Expr | None = None
    hessian_at_point: sp.Matrix | None = None
    eigenvalues: list[sp.Expr] | None = None
    assumptions: tuple[sp.Expr, ...] = ()

    def pretty(self) -> str:
        """Format the stored mathematical result and diagnostic metadata for inspection."""
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


@dataclass(frozen=True)
class CriticalConeResult:
    """Exact second-order sign analysis on a normalized critical cone."""

    classification: str
    quadratic_form: sp.Expr
    constraints: tuple[sp.Expr, ...]
    nonnegative: bool | None
    nonpositive: bool | None
    zero_direction_exists: bool | None
    positive_direction_exists: bool | None
    negative_direction_exists: bool | None


def analyze_critical_cone(
    objective: sp.Expr,
    lagrangian: sp.Expr,
    variables: Sequence[sp.Symbol],
    solution: dict[sp.Symbol, sp.Expr],
    equalities: Sequence[sp.Expr],
    inequalities: Sequence[sp.Expr],
    multipliers: Sequence[sp.Symbol],
    assumptions: Sequence[sp.Expr] = (),
) -> CriticalConeResult:
    """Analyze the Lagrangian Hessian on the exact critical cone.

    Direction variables are normalized to the unit sphere, converting strict
    positivity on nonzero critical directions into a compact semialgebraic
    decision problem.  ``g >= 0`` is the inequality convention.
    """
    variables = tuple(variables)
    directions = tuple(sp.Dummy(f"d{i}", real=True) for i in range(len(variables)))
    dvec = sp.Matrix(directions)
    constraints = []
    cone_inequalities = []

    def dot_gradient(expr):
        return sp.expand(
            sum(
                sp.diff(expr, var).subs(solution) * direction
                for var, direction in zip(variables, directions)
            )
        )

    for equality in equalities:
        constraints.append(sp.Eq(dot_gradient(equality), 0))

    for inequality, multiplier in zip(inequalities, multipliers):
        value = _safe_simplify(inequality.subs(solution), assumptions)
        if not _is_zero(value, assumptions):
            continue
        derivative = dot_gradient(inequality)
        multiplier_value = _safe_simplify(
            solution.get(multiplier, multiplier), assumptions
        )
        if _is_positive(multiplier_value, assumptions):
            constraints.append(sp.Eq(derivative, 0))
        else:
            relation = sp.Ge(derivative, 0)
            constraints.append(relation)
            cone_inequalities.append(relation)

    constraints.append(sp.Eq(dot_gradient(objective), 0))
    constraints.append(sp.Eq(sum(direction**2 for direction in directions), 1))
    hessian = sp.Matrix(sp.hessian(lagrangian, variables).subs(solution))
    quadratic = sp.expand((dvec.T * hessian * dvec)[0])

    nonnegative = prove_nonnegative_exact(quadratic, constraints, directions).proven
    nonpositive = prove_nonnegative_exact(-quadratic, constraints, directions).proven
    zero_exists = exact_feasible([*constraints, sp.Eq(quadratic, 0)], directions)
    positive_exists = exact_feasible([*constraints, sp.Gt(quadratic, 0)], directions)
    negative_exists = exact_feasible([*constraints, sp.Lt(quadratic, 0)], directions)

    # Without semialg, a purely linear-subspace critical cone can still be
    # classified exactly by reducing the Hessian to its nullspace basis.
    if (
        not cone_inequalities
        and nonnegative is None
        and nonpositive is None
        and positive_exists in (None, False)
        and negative_exists in (None, False)
    ):
        rows = []
        for relation in constraints[:-1]:  # omit sphere normalization
            if isinstance(relation, sp.Equality):
                rows.append(
                    [
                        sp.expand(relation.lhs - relation.rhs).coeff(d)
                        for d in directions
                    ]
                )
        jacobian = sp.Matrix(rows) if rows else sp.zeros(0, len(directions))
        basis_vectors = jacobian.nullspace()
        if basis_vectors:
            basis = sp.Matrix.hstack(*basis_vectors)
            reduced = sp.simplify(basis.T * hessian * basis)
            try:
                signs = []
                for eigenvalue, multiplicity in reduced.eigenvals().items():
                    sign = sp.sign(sp.simplify(eigenvalue))
                    signs.extend([sign] * int(multiplicity))
                if signs and all(sign == 1 for sign in signs):
                    nonnegative, zero_exists, positive_exists, negative_exists = (
                        True,
                        False,
                        True,
                        False,
                    )
                elif signs and all(sign == -1 for sign in signs):
                    nonpositive, zero_exists, positive_exists, negative_exists = (
                        True,
                        False,
                        False,
                        True,
                    )
                elif 1 in signs and -1 in signs:
                    positive_exists, negative_exists = True, True
                elif signs and all(sign in (0, 1) for sign in signs):
                    nonnegative, zero_exists = True, 0 in signs
                elif signs and all(sign in (0, -1) for sign in signs):
                    nonpositive, zero_exists = True, 0 in signs
            except (TypeError, ValueError, NotImplementedError):
                pass

    if nonnegative is True and zero_exists is False:
        classification = "SecondOrderSufficientMinimum"
    elif nonpositive is True and zero_exists is False:
        classification = "SecondOrderSufficientMaximum"
    elif positive_exists is True and negative_exists is True:
        classification = "IndefiniteCriticalCone"
    elif negative_exists is True:
        classification = "SecondOrderMinimumViolation"
    elif positive_exists is True and nonnegative is True:
        classification = "SecondOrderNecessaryMinimum"
    elif nonnegative is True and nonpositive is True:
        classification = "FlatCriticalCone"
    else:
        classification = "Inconclusive"
    return CriticalConeResult(
        classification,
        quadratic,
        tuple(constraints),
        nonnegative,
        nonpositive,
        zero_exists,
        positive_exists,
        negative_exists,
    )


def normalize_constraints(constraints: Sequence[Relational]):
    """Normalize equality and inequality residuals using the nonnegative convention."""
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
    solution: dict[sp.Symbol, sp.Expr],
    assumptions: Sequence[sp.Expr] = (),
    f: sp.Expr | None = None,
):
    """Classify the exact Hessian sign when symbolic eigenvalue information is sufficient."""
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
    solution: dict[sp.Symbol, sp.Expr],
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

    if uncertain_active:
        try:
            cone = analyze_critical_cone(
                objective,
                lagrangian,
                vars,
                solution,
                equalities,
                inequalities,
                multipliers,
                assumptions,
            )
            if cone.classification == "SecondOrderSufficientMinimum":
                return "LocalMinimum", hpt, []
            if cone.classification == "SecondOrderSufficientMaximum":
                return "LocalMaximum", hpt, []
            if cone.classification == "IndefiniteCriticalCone":
                return "SaddlePoint", hpt, []
            if cone.classification == "SecondOrderMinimumViolation":
                return "Inconclusive", hpt, []
        except (
            ImportError,
            TypeError,
            ValueError,
            ArithmeticError,
            NotImplementedError,
        ):
            pass

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
    karush_john_multiplier: str | None = None,
):
    """Construct and solve constrained stationary systems for KKT/Karush--John analysis."""
    vars = list(vars)
    eqs, ineqs_ge, ineqs_gt = normalize_constraints(constraints)
    ineqs = list(ineqs_ge) + list(ineqs_gt)
    strict_assumptions = [sp.Gt(h, 0) for h in ineqs_gt]

    occupied = f.free_symbols | set(vars)
    for residual in (*eqs, *ineqs):
        occupied.update(residual.free_symbols)
    lam = fresh_symbols(
        (f"{lambda_prefix}_{i + 1}" for i in range(len(eqs))), occupied, real=True
    )
    mu = fresh_symbols(
        (f"{mu_prefix}_{i + 1}" for i in range(len(ineqs))),
        (*occupied, *lam),
        real=True,
        nonnegative=True,
    )
    rho = (
        fresh_symbols(
            (str(karush_john_multiplier),),
            (*occupied, *lam, *mu),
            real=True,
            nonnegative=True,
        )[0]
        if karush_john_multiplier is not None
        else None
    )

    if rho is not None:
        L = (
            rho * f
            + sum(mult * residual for mult, residual in zip(lam, eqs))
            - sum(m * h for m, h in zip(mu, ineqs))
        )
        allvars = [rho, *lam, *mu, *vars]
    else:
        L = (
            f
            + sum(mult * residual for mult, residual in zip(lam, eqs))
            - sum(m * h for m, h in zip(mu, ineqs))
        )
        allvars = [*lam, *mu, *vars]

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
    karush_john_multiplier: str | None = None,
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
    solution: dict[sp.Symbol, sp.Expr],
    target: str = "Min",
) -> bool:
    """Apply the supported unconstrained or reduced-Hessian local second-order tests."""
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
    point_rules: dict[sp.Symbol, sp.Expr],
    domain: str = "Reals",
    lambda_prefix: str = "lam",
    mu_prefix: str = "mu",
    karush_john_multiplier: str | None = None,
) -> bool:
    """Check the candidate stationarity, feasibility and multiplier conditions of the KKT system."""
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
