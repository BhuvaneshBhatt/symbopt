from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import sympy as sp
from sympy.core.relational import Equality, Relational

from .symbolic_cache import cached_gradient, cached_hessian, cached_poly
from .diagnostics import DiagnosticKind, SolverDiagnostic
from .lp import solve_lp
from .problem_models import LiftedProblem
from .numeric_utils import eval_float
from .qp import solve_convex_qp
from .precision import float64_backends_allowed
from .relation_checks import normalize_relational, relation_holds
from .verified_intervals import EvalConfig, enclose_expression


@dataclass
class DirectSolveResult:
    kind: str
    point: Dict[sp.Symbol, sp.Expr]
    objective_value: sp.Expr
    multipliers: Dict[sp.Symbol, sp.Expr] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)


def _poly_degree(expr, vars) -> Optional[int]:
    try:
        return cached_poly(sp.expand(expr), tuple(vars)).total_degree()
    except (sp.PolynomialError, ValueError, TypeError):
        return None


def _extract_linear_system(constraints, vars, digits: int = 30):
    inequalities = []
    ineq_rhs = []
    equalities = []
    eq_rhs = []
    zero = {var: 0 for var in vars}

    for relation in constraints:
        kind, residual = normalize_relational(relation)
        if _poly_degree(residual, vars) not in {0, 1}:
            return None
        gradient = cached_gradient(residual, tuple(vars))
        coeffs = [eval_float(component, digits) for component in gradient]
        constant = eval_float(residual.subs(zero), digits)
        if kind == "ge":
            inequalities.append([-value for value in coeffs])
            ineq_rhs.append(constant)
        elif kind == "eq":
            equalities.append(coeffs)
            eq_rhs.append(-constant)
        else:
            return None

    return (
        np.asarray(inequalities, dtype=float) if inequalities else None,
        np.asarray(ineq_rhs, dtype=float) if ineq_rhs else None,
        np.asarray(equalities, dtype=float) if equalities else None,
        np.asarray(eq_rhs, dtype=float) if eq_rhs else None,
    )


def _active_constraints(constraints, point, tol=1e-7):
    equalities = []
    active_ineq = []
    for relation in constraints:
        kind, residual = normalize_relational(relation)
        if kind == "eq":
            equalities.append((relation, residual))
            continue
        if kind != "ge":
            continue
        try:
            value = abs(eval_float(residual.subs(point), 40))
        except (ValueError, TypeError, ArithmeticError):
            continue
        if value <= tol:
            active_ineq.append((relation, residual))
    return equalities, active_ineq


def _exact_feasible(constraints, point):
    results = []
    exact_all = True
    numeric_all = True
    for relation in constraints:
        try:
            simplified = sp.simplify(relation.subs(point))
        except (TypeError, ValueError, ArithmeticError) as exc:
            results.append((sp.sstr(relation), None, str(exc)))
            exact_all = False
            numeric_all = False
            continue
        if simplified is sp.S.true or simplified is True:
            results.append((sp.sstr(relation), True, "exact"))
        elif simplified is sp.S.false or simplified is False:
            results.append((sp.sstr(relation), False, "exact"))
            exact_all = False
            numeric_all = False
        else:
            holds = relation_holds(relation, point, 1e-12)
            results.append((sp.sstr(relation), holds, "numeric fallback"))
            exact_all = False
            numeric_all = numeric_all and bool(holds)
    return exact_all, numeric_all, results


def _recover_exact_kkt(objective, constraints, vars, numeric_point):
    equalities, active_ineq = _active_constraints(constraints, numeric_point)
    lam = sp.symbols(f"_lam0:{len(equalities)}", real=True)
    mu = sp.symbols(f"_mu0:{len(active_ineq)}", nonnegative=True)
    lagrangian = objective
    for symbol, (_, residual) in zip(lam, equalities):
        lagrangian += symbol * residual
    for symbol, (_, residual) in zip(mu, active_ineq):
        lagrangian -= symbol * residual

    equations = [sp.diff(lagrangian, var) for var in vars]
    equations += [residual for _, residual in equalities]
    equations += [residual for _, residual in active_ineq]
    unknowns = [*vars, *lam, *mu]
    solutions = sp.solve(
        equations,
        unknowns,
        dict=True,
        simplify=True,
    )

    best = None
    best_distance = float("inf")
    for solution in solutions:
        if not all(var in solution for var in vars):
            continue
        point = {var: sp.simplify(solution[var]) for var in vars}
        if any(point[var].free_symbols.intersection(vars) for var in vars):
            continue
        exact_feasible, numeric_feasible, checks = _exact_feasible(constraints, point)
        if not numeric_feasible:
            continue
        multipliers = {
            symbol: sp.simplify(solution[symbol])
            for symbol in [*lam, *mu]
            if symbol in solution
        }
        multiplier_signs = [
            (
                symbol in multipliers
                and sp.ask(sp.Q.nonnegative(multipliers[symbol])) is True
            )
            for symbol in mu
        ]
        if not all(multiplier_signs):
            continue
        try:
            distance = sum(
                abs(eval_float(point[var], 40) - float(numeric_point[var]))
                for var in vars
            )
        except (ValueError, TypeError, ArithmeticError):
            distance = 0.0
        if distance < best_distance:
            best_distance = distance
            best = (
                point,
                multipliers,
                checks,
                lagrangian,
                exact_feasible,
            )
    return best


def _postprocess_direct(
    kind,
    objective,
    constraints,
    vars,
    numeric_point,
    numeric_value,
    diagnostics=None,
    convexity_exact=True,
):
    fallback = {
        var: sp.nsimplify(value, rational=True) for var, value in numeric_point.items()
    }
    exact_point = fallback
    multipliers = {}
    recovered = False
    exact_feasible, numeric_feasible, checks = _exact_feasible(constraints, fallback)
    metadata = {
        "numeric_point": {sp.sstr(k): float(v) for k, v in numeric_point.items()},
        "active_constraints": [
            sp.sstr(relation)
            for relation, _ in _active_constraints(constraints, numeric_point)[1]
        ],
        "exact_recovered": False,
        "feasibility_checks": checks,
        "convexity_exact": bool(convexity_exact),
        "rigorous_global": False,
    }

    try:
        recovered_data = _recover_exact_kkt(objective, constraints, vars, numeric_point)
        if recovered_data is not None:
            (
                exact_point,
                multipliers,
                checks,
                lagrangian,
                exact_feasible,
            ) = recovered_data
            recovered = True
            numeric_feasible = True
            metadata.update(
                {
                    "exact_recovered": True,
                    "feasibility_checks": checks,
                    "lagrangian": sp.sstr(lagrangian),
                    "stationarity_residuals": [
                        sp.sstr(
                            sp.simplify(
                                sp.diff(lagrangian, var).subs(
                                    {**exact_point, **multipliers}
                                )
                            )
                        )
                        for var in vars
                    ],
                    "multipliers": {
                        sp.sstr(symbol): sp.sstr(value)
                        for symbol, value in multipliers.items()
                    },
                }
            )
    except (ValueError, TypeError, NotImplementedError, sp.SympifyError) as exc:
        if diagnostics is not None:
            diagnostics.append(
                SolverDiagnostic.from_exception(
                    kind,
                    "exact_postprocess",
                    DiagnosticKind.CERTIFICATE_FAILURE,
                    exc,
                )
            )

    value = sp.simplify(objective.subs(exact_point))
    if value.free_symbols:
        value = sp.nsimplify(numeric_value, rational=True)
    metadata["exact_feasible"] = exact_feasible
    metadata["numeric_feasible"] = numeric_feasible
    metadata["exact_recovered"] = recovered
    metadata["rigorous_global"] = bool(recovered and exact_feasible and convexity_exact)
    return DirectSolveResult(kind, exact_point, value, multipliers, metadata)


def try_direct_solver(
    objective,
    constraints,
    vars,
    diagnostics: Optional[List[SolverDiagnostic]] = None,
    working_precision: Optional[int] = None,
):
    if not vars:
        return None
    if not float64_backends_allowed(objective, constraints, working_precision):
        if diagnostics is not None:
            diagnostics.append(
                SolverDiagnostic(
                    "routing",
                    "precision_policy",
                    DiagnosticKind.UNSUPPORTED,
                    "Bypassed float64 LP/QP fast path for high-precision inexact input.",
                )
            )
        return None
    numeric_digits = max(int(working_precision or 30), 30)
    system = _extract_linear_system(constraints, vars, numeric_digits)
    if system is None:
        return None
    A, b, Aeq, beq = system
    objective = sp.expand(objective)
    degree = _poly_degree(objective, vars)
    zero = {var: 0 for var in vars}

    if degree is not None and degree <= 1:
        gradient = cached_gradient(objective, tuple(vars))
        try:
            coeffs = np.asarray(
                [eval_float(component, numeric_digits) for component in gradient],
                dtype=float,
            )
            constant = eval_float(objective.subs(zero), numeric_digits)
        except (ValueError, TypeError, ArithmeticError):
            coeffs = None
        if coeffs is not None:
            result = solve_lp(
                coeffs,
                A_ub=A,
                b_ub=b,
                A_eq=Aeq,
                b_eq=beq,
                maximize=False,
                bounds=[(None, None)] * len(vars),
            )
            if result.status == "optimal":
                numeric_point = {
                    var: float(value) for var, value in zip(vars, result.x)
                }
                return _postprocess_direct(
                    "lp",
                    objective,
                    constraints,
                    vars,
                    numeric_point,
                    result.objective + constant,
                    diagnostics,
                )
            if diagnostics is not None and result.status not in {
                "infeasible",
                "unbounded",
            }:
                diagnostics.append(
                    SolverDiagnostic(
                        "lp",
                        "direct_solve",
                        DiagnosticKind.NUMERICAL_FAILURE,
                        result.message,
                    )
                )

    if degree is not None and degree <= 2:
        try:
            hessian = np.asarray(
                cached_hessian(objective, tuple(vars)).evalf(numeric_digits),
                dtype=float,
            )
        except (ValueError, TypeError, ArithmeticError):
            return None
        symmetric = 0.5 * (hessian + hessian.T)
        try:
            if np.linalg.eigvalsh(symmetric).min() < -1e-9:
                return None
        except np.linalg.LinAlgError as exc:
            if diagnostics is not None:
                diagnostics.append(
                    SolverDiagnostic.from_exception(
                        "qp",
                        "convexity_check",
                        DiagnosticKind.NUMERICAL_FAILURE,
                        exc,
                    )
                )
            return None

        gradient = cached_gradient(objective, tuple(vars))
        try:
            coeffs = np.asarray(
                [
                    eval_float(component.subs(zero), numeric_digits)
                    for component in gradient
                ],
                dtype=float,
            )
            constant = eval_float(objective.subs(zero), numeric_digits)
        except (ValueError, TypeError, ArithmeticError):
            return None
        result = solve_convex_qp(
            hessian,
            coeffs,
            A_ub=A,
            b_ub=b,
            A_eq=Aeq,
            b_eq=beq,
        )
        if result.status == "optimal":
            numeric_point = {var: float(value) for var, value in zip(vars, result.x)}
            exact_hessian = sp.Matrix(cached_hessian(objective, tuple(vars)))
            convexity_exact = exact_hessian.is_positive_semidefinite is True
            return _postprocess_direct(
                "qp",
                objective,
                constraints,
                vars,
                numeric_point,
                result.objective + constant,
                diagnostics,
                convexity_exact=convexity_exact,
            )
        if diagnostics is not None:
            diagnostics.append(
                SolverDiagnostic(
                    "qp",
                    "direct_solve",
                    DiagnosticKind.NUMERICAL_FAILURE,
                    result.message,
                )
            )
    return None


def infer_box_bounds(variables, constraints):
    """Infer finite scalar variable bounds from simple relational constraints.

    ``variables`` may be one symbol or an iterable, and ``constraints`` may be
    one relation or an iterable. Only bounds justified directly by the supplied
    relations are returned; missing or one-sided bounds are omitted.
    """
    if isinstance(variables, sp.Symbol):
        variables = (variables,)
    if isinstance(constraints, Relational):
        constraints = (constraints,)
    bounds: Dict[sp.Symbol, List[Optional[float]]] = {
        var: [None, None] for var in variables
    }
    for relation in constraints:
        lhs, rhs = relation.lhs, relation.rhs
        if isinstance(lhs, sp.Symbol) and rhs.is_number:
            value = eval_float(rhs)
            if relation.func in (sp.Ge, sp.Gt):
                current = bounds[lhs][0]
                bounds[lhs][0] = value if current is None else max(current, value)
            elif relation.func in (sp.Le, sp.Lt):
                current = bounds[lhs][1]
                bounds[lhs][1] = value if current is None else min(current, value)
            elif relation.func is sp.Eq:
                bounds[lhs] = [value, value]
        if isinstance(rhs, sp.Symbol) and lhs.is_number:
            value = eval_float(lhs)
            if relation.func in (sp.Ge, sp.Gt):
                current = bounds[rhs][1]
                bounds[rhs][1] = value if current is None else min(current, value)
            elif relation.func in (sp.Le, sp.Lt):
                current = bounds[rhs][0]
                bounds[rhs][0] = value if current is None else max(current, value)
            elif relation.func is sp.Eq:
                bounds[rhs] = [value, value]
    return {
        var: (low, high)
        for var, (low, high) in bounds.items()
        if low is not None and high is not None and low <= high
    }


def _rigorous_enclosure(expr, bounds):
    symbols = tuple(sorted(expr.free_symbols, key=sp.sstr))
    if any(symbol not in bounds for symbol in symbols):
        return None
    enclosure = enclose_expression(
        expr,
        symbols,
        tuple(bounds[symbol] for symbol in symbols),
        EvalConfig(),
    )
    if not enclosure.rigorous or enclosure.domain != "valid":
        return None
    return enclosure.lower, enclosure.upper


def complete_box_bounds(problem: LiftedProblem, bounds):
    completed = dict(bounds)
    for _ in range(8):
        changed = False
        relations = [*problem.graph_equalities, *problem.lifted_constraints]
        for relation in relations:
            if not isinstance(relation, Equality):
                continue
            lhs, rhs = relation.lhs, relation.rhs
            if isinstance(lhs, sp.Symbol) and lhs not in completed:
                enclosure = _rigorous_enclosure(rhs, completed)
                if enclosure is not None:
                    completed[lhs] = enclosure
                    changed = True
            elif isinstance(rhs, sp.Symbol) and rhs not in completed:
                enclosure = _rigorous_enclosure(lhs, completed)
                if enclosure is not None:
                    completed[rhs] = enclosure
                    changed = True
        if not changed:
            break
    return completed
