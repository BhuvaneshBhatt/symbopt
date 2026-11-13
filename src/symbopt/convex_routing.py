from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from sympy.core.relational import Equality, Relational

from .diagnostics import DiagnosticKind, SolverDiagnostic
from .lp import solve_lp
from .numeric_utils import eval_float
from .precision import float64_backends_allowed, outward_float
from .problem_models import LiftedProblem
from .qp import solve_convex_qp
from .relation_checks import normalize_relational, relation_holds
from .symbolic_cache import (
    affine_coefficients,
    cached_gradient,
    cached_hessian,
    cached_poly,
)
from .verified_intervals import EvalConfig, enclose_expression


@dataclass
class DirectSolveResult:
    """Direct LP or QP candidate with exact postprocessing and global-proof metadata."""

    kind: str
    point: dict[sp.Symbol, sp.Expr]
    objective_value: sp.Expr
    multipliers: dict[sp.Symbol, sp.Expr] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)


def _poly_degree(expr, vars) -> int | None:
    try:
        return cached_poly(sp.expand(expr), tuple(vars)).total_degree()
    except (sp.PolynomialError, ValueError, TypeError):
        return None


def _extract_linear_system(constraints, vars, digits: int = 30):
    inequalities = []
    ineq_rhs = []
    equalities = []
    eq_rhs = []
    for relation in constraints:
        kind, residual = normalize_relational(relation)
        data = affine_coefficients(residual, vars)
        if data is None:
            return None
        coefficients, constant_expr = data
        if any(value.free_symbols for value in (*coefficients, constant_expr)):
            return None
        try:
            coeffs = [eval_float(component, digits) for component in coefficients]
            constant = eval_float(constant_expr, digits)
        except (ValueError, TypeError, ArithmeticError):
            return None
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


def _recover_exact_kkt(objective, constraints, vars, numeric_point, active_constraints):
    """Recover an exact feasible KKT point using the prepared active relations."""
    equalities, active_ineq = active_constraints
    lam = tuple(sp.Dummy(f"lam{i}", real=True) for i in range(len(equalities)))
    mu = tuple(sp.Dummy(f"mu{i}", nonnegative=True) for i in range(len(active_ineq)))
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
    active_constraints = _active_constraints(constraints, numeric_point)
    metadata = {
        "numeric_point": {sp.sstr(k): float(v) for k, v in numeric_point.items()},
        "active_constraints": [
            sp.sstr(relation) for relation, _ in active_constraints[1]
        ],
        "exact_recovered": False,
        "feasibility_checks": checks,
        "convexity_exact": bool(convexity_exact),
        "rigorous_global": False,
    }

    try:
        recovered_data = _recover_exact_kkt(
            objective, constraints, vars, numeric_point, active_constraints
        )
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
    if exact_feasible and not metadata["rigorous_global"]:
        from .convex_rules import convex_kkt_certificate
        from .rule_utils import RULE_ERRORS, RuleContext

        try:
            proof = convex_kkt_certificate(
                RuleContext(objective, tuple(constraints), tuple(vars)), exact_point
            )
        except RULE_ERRORS:
            proof = None
        if proof is not None:
            metadata.update(
                {
                    "exact_recovered": True,
                    "rigorous_global": True,
                    "convexity_exact": True,
                    "stationarity_residuals": [
                        sp.sstr(v) for v in proof.metadata["stationarity"]
                    ],
                    "convex_kkt": dict(proof.metadata),
                }
            )
    return DirectSolveResult(kind, exact_point, value, multipliers, metadata)


def try_direct_solver(
    objective,
    constraints,
    vars,
    diagnostics: list[SolverDiagnostic] | None = None,
    working_precision: int | None = None,
):
    """Route affine or quadratic problems to direct numerical solvers with validation."""
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
    if not constraints and not objective.has(sp.Float):
        from .rule_utils import RULE_ERRORS, RuleContext
        from .spectral_rules import try_exact_quadratic

        try:
            exact = try_exact_quadratic(RuleContext(objective, (), tuple(vars)))
        except RULE_ERRORS:
            exact = None
        if exact is not None:
            return DirectSolveResult(
                "qp",
                dict(exact.points[0]),
                exact.value,
                metadata={
                    "exact_recovered": True,
                    "exact_feasible": True,
                    "convexity_exact": True,
                    "rigorous_global": True,
                    "stationarity_residuals": [
                        sp.sstr(v) for v in exact.metadata["stationarity_residuals"]
                    ],
                    "method": "exact_linear_algebra",
                    "hessian": exact.metadata["hessian"],
                },
            )
    numeric_digits = max(int(working_precision or 30), 30)
    system = _extract_linear_system(constraints, vars, numeric_digits)
    if system is None:
        return None
    A, b, Aeq, beq = system
    objective = sp.expand(objective)
    degree = _poly_degree(objective, vars)
    zero = {var: 0 for var in vars}

    if degree is not None and degree <= 1:
        coefficients, constant_expr = affine_coefficients(objective, vars)
        try:
            coeffs = np.asarray(
                [eval_float(component, numeric_digits) for component in coefficients],
                dtype=float,
            )
            constant = eval_float(constant_expr, numeric_digits)
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
    bounds: dict[sp.Symbol, list[float | None]] = {
        var: [None, None] for var in variables
    }
    for relation in constraints:
        lhs, rhs = relation.lhs, relation.rhs
        if isinstance(lhs, sp.Symbol) and lhs in bounds and rhs.is_number:
            variable, endpoint, kind = lhs, rhs, relation.func
        elif isinstance(rhs, sp.Symbol) and rhs in bounds and lhs.is_number:
            variable, endpoint, kind = rhs, lhs, relation.reversed.func
        else:
            continue
        if kind in (sp.Ge, sp.Gt, sp.Eq):
            low = outward_float(endpoint, -1)
            current = bounds[variable][0]
            bounds[variable][0] = low if current is None else max(current, low)
        if kind in (sp.Le, sp.Lt, sp.Eq):
            high = outward_float(endpoint, 1)
            current = bounds[variable][1]
            bounds[variable][1] = high if current is None else min(current, high)

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
    """Extend a search box with rigorous enclosures for defined auxiliary coordinates."""
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
