from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import sympy as sp

from .algebraic_numbers import algebraic_compare
from .diagnostics import DiagnosticKind, SolverDiagnostic
from .kkt import build_kkt_system
from .numeric_utils import eval_float, real_part
from .precision import finite_midpoint
from .problem_models import CandidateSolution, KKTSystem, LiftedProblem
from .relation_checks import relation_holds
from .semialgebraic import solve_exact_points
from .verified_intervals import EvalConfig, enclose_expression


def _note(
    diagnostics: list[SolverDiagnostic] | None,
    solver: str,
    stage: str,
    kind: DiagnosticKind,
    exc: BaseException,
) -> None:
    if diagnostics is not None:
        diagnostics.append(SolverDiagnostic.from_exception(solver, stage, kind, exc))


def _activity_profile(
    inequalities: Sequence[sp.Expr],
    variables: Sequence[sp.Symbol],
    box_bounds,
    tol: float,
):
    required_active = set()
    required_inactive = set()
    priorities = {}
    infeasible = False

    if not box_bounds:
        return required_active, required_inactive, priorities, infeasible

    for index, residual in enumerate(inequalities):
        symbols = tuple(sorted(residual.free_symbols, key=sp.sstr))
        if not symbols or any(symbol not in box_bounds for symbol in symbols):
            continue
        try:
            bound = enclose_expression(
                residual,
                symbols,
                tuple(box_bounds[symbol] for symbol in symbols),
                EvalConfig(use_taylor=False),
            )
        except (TypeError, ValueError, ArithmeticError):
            continue
        if not bound.rigorous or bound.domain != "valid":
            continue
        if bound.upper < -tol:
            infeasible = True
            break
        if bound.lower > tol:
            required_inactive.add(index)
            priorities[index] = float("inf")
            continue
        if abs(bound.lower) <= tol and abs(bound.upper) <= tol:
            required_active.add(index)
            priorities[index] = 0.0
            continue
        midpoint = 0.5 * (bound.lower + bound.upper)
        width = max(bound.width, tol)
        priorities[index] = abs(midpoint) / width

    return required_active, required_inactive, priorities, infeasible


@dataclass(frozen=True)
class ActiveSetPlan:
    """Shared prioritized KKT active-set enumeration plan."""

    patterns: tuple[tuple[int, ...], ...]
    complete: bool
    infeasible: bool = False


def build_active_set_plan(
    inequalities: Sequence[sp.Expr],
    variables: Sequence[sp.Symbol],
    max_active_sets: int,
    box_bounds=None,
    tol: float = 1e-7,
    equalities=(),
    diagnostics=None,
) -> ActiveSetPlan:
    """Enumerate prioritized KKT active sets and report whether the plan is exhaustive."""
    count = len(inequalities)
    if count == 0:
        return ActiveSetPlan(((),), True, False)
    if max_active_sets <= 0:
        return ActiveSetPlan((), False, False)

    required_active, required_inactive, priorities, infeasible = _activity_profile(
        inequalities, variables, box_bounds, tol
    )
    if infeasible or required_active.intersection(required_inactive):
        return ActiveSetPlan((), True, True)

    flexible = [
        index
        for index in range(count)
        if index not in required_active and index not in required_inactive
    ]
    flexible.sort(key=lambda index: priorities.get(index, 1.0))
    patterns = []
    seen = set()
    truncated = False
    for active_count in range(len(flexible) + 1):
        for chosen in combinations(flexible, active_count):
            active = set(required_active).union(chosen)
            pattern = tuple(1 if index in active else 0 for index in range(count))
            if pattern in seen:
                continue
            seen.add(pattern)
            if len(patterns) >= max_active_sets:
                truncated = True
                break
            patterns.append(pattern)
        if truncated:
            break
    return ActiveSetPlan(tuple(patterns), not truncated, False)


def _all_polynomial(expressions, symbols) -> bool:
    return all(sp.expand(expr).is_polynomial(*symbols) for expr in expressions)


def _collect_kkt_data(kkt: KKTSystem):
    stationarity = [sp.expand(eq.lhs - eq.rhs) for eq in kkt.stationarity_equations]
    equalities = [sp.expand(eq.lhs - eq.rhs) for eq in kkt.primal_equalities]
    inequalities = [sp.expand(rel.lhs - rel.rhs) for rel in kkt.primal_inequalities]
    unknowns = [
        *kkt.primal_variables,
        *kkt.equality_multipliers,
        *kkt.inequality_multipliers,
    ]
    return stationarity, equalities, inequalities, unknowns


def _solve_multiplier_system(
    equations,
    multipliers,
    diagnostics=None,
):
    if not multipliers:
        return [{}]
    try:
        solutions = sp.solve(
            list(equations),
            list(multipliers),
            dict=True,
            simplify=True,
        )
    except (NotImplementedError, ValueError, TypeError) as exc:
        _note(
            diagnostics,
            "symbolic_candidates",
            "solve_multipliers",
            DiagnosticKind.SYMBOLIC_FAILURE,
            exc,
        )
        return []

    return [
        {multiplier: sp.simplify(solution[multiplier]) for multiplier in multipliers}
        for solution in solutions
        if all(multiplier in solution for multiplier in multipliers)
    ]


def _eliminate_multipliers(
    equations,
    multipliers,
    primal_vars,
    diagnostics=None,
):
    if not equations:
        return []
    if not multipliers:
        return [sp.expand(eq) for eq in equations]
    try:
        basis = sp.groebner(
            list(equations),
            *[*multipliers, *primal_vars],
            order="lex",
        )
    except (ValueError, TypeError, NotImplementedError) as exc:
        _note(
            diagnostics,
            "symbolic_candidates",
            "groebner_elimination",
            DiagnosticKind.SYMBOLIC_FAILURE,
            exc,
        )
        return []

    reduced = []
    seen = set()
    multiplier_set = set(multipliers)
    for polynomial in basis:
        expr = sp.expand(polynomial)
        if expr != 0 and not expr.free_symbols.intersection(multiplier_set):
            key = sp.srepr(expr)
            if key not in seen:
                seen.add(key)
                reduced.append(expr)
    return reduced


def _recover_multipliers(
    equations,
    multipliers,
    primal_solution,
    diagnostics=None,
):
    if not multipliers:
        return {}
    reduced = [sp.expand(eq.subs(primal_solution)) for eq in equations]
    try:
        solutions = sp.solve(
            reduced,
            list(multipliers),
            dict=True,
            simplify=True,
        )
    except (NotImplementedError, ValueError, TypeError) as exc:
        _note(
            diagnostics,
            "symbolic_candidates",
            "recover_multipliers",
            DiagnosticKind.SYMBOLIC_FAILURE,
            exc,
        )
        solutions = []

    for solution in solutions:
        if all(multiplier in solution for multiplier in multipliers):
            return {
                multiplier: sp.simplify(solution[multiplier])
                for multiplier in multipliers
            }
    return {}


def _solve_poly_system(equations, primal_vars, diagnostics=None):
    equations = [expanded for eq in equations if (expanded := sp.expand(eq)) != 0]
    if not equations or not _all_polynomial(equations, primal_vars):
        return []

    solutions = []
    try:
        raw = sp.solve_poly_system(list(equations), *primal_vars)
        for values in raw or []:
            if len(values) == len(primal_vars):
                solutions.append(dict(zip(primal_vars, values)))
        if solutions:
            return solutions
    except (NotImplementedError, ValueError, TypeError) as exc:
        _note(
            diagnostics,
            "symbolic_candidates",
            "solve_poly_system",
            DiagnosticKind.SYMBOLIC_FAILURE,
            exc,
        )

    semialg_points = solve_exact_points(equations, primal_vars, diagnostics)
    if semialg_points:
        return semialg_points

    try:
        raw = sp.nonlinsolve(list(equations), list(primal_vars))
        for values in raw:
            if len(values) == len(primal_vars) and all(
                not value.free_symbols.intersection(primal_vars) for value in values
            ):
                solutions.append(dict(zip(primal_vars, values)))
    except (NotImplementedError, ValueError, TypeError) as exc:
        _note(
            diagnostics,
            "symbolic_candidates",
            "nonlinsolve",
            DiagnosticKind.SYMBOLIC_FAILURE,
            exc,
        )

    unique = []
    seen = set()
    for solution in solutions:
        key = tuple(sp.srepr(sp.simplify(solution[var])) for var in primal_vars)
        if key not in seen:
            seen.add(key)
            unique.append(solution)
    return unique


def _seed_values(kkt: KKTSystem, box_bounds, mode: str):
    seed = []
    for variable in kkt.primal_variables:
        if box_bounds and variable in box_bounds:
            low, high = box_bounds[variable]
            if mode == "mid":
                seed.append(finite_midpoint(low, high))
            elif mode == "low":
                seed.append(low + 0.2 * (high - low))
            else:
                seed.append(high - 0.2 * (high - low))
        else:
            seed.append(0.0 if mode == "mid" else -0.5 if mode == "low" else 0.5)
    seed.extend([0.0] * len(kkt.equality_multipliers))
    seed.extend([0.0] * len(kkt.inequality_multipliers))
    return seed


def _original_values(problem, primal):
    return {
        variable: primal[variable]
        for variable in problem.original_variables
        if variable in primal
    }


def _classify_candidate(kkt, raw_values, tol):
    """Classify only a candidate that survived feasibility filtering."""
    primal = {v: raw_values[v] for v in kkt.primal_variables if v in raw_values}
    try:
        active_rows = []
        for relation in kkt.primal_equalities:
            residual = sp.expand(relation.lhs - relation.rhs)
            active_rows.append(
                [sp.diff(residual, v).subs(primal) for v in kkt.primal_variables]
            )
        for relation in kkt.primal_inequalities:
            residual = sp.expand(relation.lhs - relation.rhs)
            value = complex(sp.N(residual.subs(primal), 30))
            if abs(value.imag) <= tol and abs(value.real) <= tol:
                active_rows.append(
                    [sp.diff(residual, v).subs(primal) for v in kkt.primal_variables]
                )
        hessian = sp.hessian(kkt.lagrangian, kkt.primal_variables).subs(raw_values)
        if active_rows:
            nullspace = sp.Matrix(active_rows).nullspace()
            if not nullspace:
                return "LocalMinimum"
            basis = sp.Matrix.hstack(*nullspace)
            hessian = sp.simplify(basis.T * hessian * basis)
        eigenvalues = list(hessian.eigenvals())
        if not eigenvalues:
            return "LocalMinimum"
        numeric = [float(sp.N(sp.re(value), 30)) for value in eigenvalues]
        if all(value > tol for value in numeric):
            return "LocalMinimum"
        if all(value < -tol for value in numeric):
            return "LocalMaximum"
        if any(value > tol for value in numeric) and any(
            value < -tol for value in numeric
        ):
            return "SaddlePoint"
        return "Degenerate"
    except (TypeError, ValueError, ArithmeticError, NotImplementedError):
        return None


def _filter_candidate(
    problem,
    kkt,
    raw_values,
    source,
    tol,
    critical_results=None,
    working_precision: int | None = None,
):
    digits = max(int(working_precision or 50), 30)
    primal = {}
    for variable in kkt.primal_variables:
        if variable not in raw_values:
            continue
        value = raw_values[variable]
        try:
            eval_float(value, digits)
        except (TypeError, ValueError, ArithmeticError, NotImplementedError):
            return None
        if getattr(value, "is_number", False) and not value.has(sp.Float):
            primal[variable] = sp.simplify(value)
        else:
            primal[variable] = sp.nsimplify(sp.re(sp.N(value, digits)))
    if len(primal) != len(kkt.primal_variables):
        return None
    if any(
        not relation_holds(relation, primal, tol)
        for relation in problem.all_constraints()
    ):
        return None

    multipliers = {
        multiplier: sp.N(raw_values[multiplier], digits)
        for multiplier in [
            *kkt.equality_multipliers,
            *kkt.inequality_multipliers,
        ]
        if multiplier in raw_values
    }
    classification = _classify_candidate(kkt, {**raw_values, **primal}, tol)
    projected = _original_values(problem, primal)
    objective = sp.simplify(problem.lifted_objective.subs(primal))
    try:
        numeric_value = eval_float(objective, digits)
    except (TypeError, ValueError, ArithmeticError, NotImplementedError):
        return None
    return CandidateSolution(
        primal_values=primal,
        multiplier_values=multipliers,
        objective_value=objective,
        objective_value_numeric=numeric_value,
        feasibility_checked=True,
        source=source,
        local_classification=classification,
        original_variable_values=projected,
    )


def solve_exact_candidates(
    problem: LiftedProblem,
    kkt_system=None,
    box_bounds=None,
    tol: float = 1e-7,
    max_active_sets: int = 64,
    cp_results=None,
    diagnostics: list[SolverDiagnostic] | None = None,
    active_set_plan: ActiveSetPlan | None = None,
    exact_search_metadata: dict | None = None,
):
    """Extract polynomial/KKT candidates using symbolic solving and elimination before numerical fallback.

    The active-set budget is ordered and pruned using available feasibility
    information so boundary solutions are not discarded merely because exhaustive
    enumeration is too large."""
    kkt = kkt_system or build_kkt_system(problem)
    if not problem.is_polynomial_problem():
        return []

    candidates = []
    seen = set()
    primal_vars = list(kkt.primal_variables)
    stationarity, equalities, inequalities, _ = _collect_kkt_data(kkt)
    plan = active_set_plan or build_active_set_plan(
        inequalities,
        primal_vars,
        max_active_sets,
        box_bounds,
        tol,
        equalities,
        diagnostics,
    )
    patterns = plan.patterns
    diagnostic_start = len(diagnostics) if diagnostics is not None else 0

    for pattern in patterns:
        active_multipliers = [
            multiplier
            for multiplier, active in zip(kkt.inequality_multipliers, pattern)
            if active
        ]
        multiplier_vars = [*kkt.equality_multipliers, *active_multipliers]
        inactive = {
            multiplier: sp.Integer(0)
            for multiplier, active in zip(kkt.inequality_multipliers, pattern)
            if not active
        }
        stationarity_eqs = [sp.expand(eq.subs(inactive)) for eq in stationarity]
        active_eqs = [
            *equalities,
            *[
                inequality
                for inequality, active in zip(inequalities, pattern)
                if active
            ],
        ]

        reduced_systems = []
        for solution in _solve_multiplier_system(
            stationarity_eqs, multiplier_vars, diagnostics
        ):
            reduced_systems.append(
                ([sp.expand(eq.subs(solution)) for eq in active_eqs], dict(solution))
            )

        eliminated = _eliminate_multipliers(
            stationarity_eqs,
            multiplier_vars,
            primal_vars,
            diagnostics,
        )
        if eliminated:
            reduced_systems.append(([*active_eqs, *eliminated], {}))

        for reduced_eqs, multiplier_solution in reduced_systems:
            for primal_solution in _solve_poly_system(
                reduced_eqs, primal_vars, diagnostics
            ):
                raw = dict(primal_solution)
                recovered = _recover_multipliers(
                    stationarity_eqs,
                    multiplier_vars,
                    primal_solution,
                    diagnostics,
                )
                if not recovered:
                    recovered = {
                        multiplier: sp.simplify(value.subs(primal_solution))
                        for multiplier, value in multiplier_solution.items()
                        if multiplier in multiplier_vars
                    }

                for multiplier in kkt.equality_multipliers:
                    if multiplier in recovered:
                        raw[multiplier] = recovered[multiplier]
                for multiplier, active in zip(kkt.inequality_multipliers, pattern):
                    raw[multiplier] = (
                        recovered[multiplier]
                        if active and multiplier in recovered
                        else sp.Integer(0)
                    )

                candidate = _filter_candidate(
                    problem,
                    kkt,
                    raw,
                    "symbolic_active_" + "".join(map(str, pattern)),
                    tol,
                    critical_results=cp_results,
                )
                if candidate is None:
                    continue
                key = tuple(
                    round(real_part(candidate.primal_values[var]), 12)
                    for var in primal_vars
                )
                if key not in seen:
                    seen.add(key)
                    candidates.append(candidate)

    if exact_search_metadata is not None:
        new_diagnostics = (
            diagnostics[diagnostic_start:] if diagnostics is not None else []
        )
        symbolic_failures = any(
            item.kind == DiagnosticKind.SYMBOLIC_FAILURE for item in new_diagnostics
        )
        exact_search_metadata.update(
            {
                "active_set_plan_complete": plan.complete,
                "patterns_planned": len(patterns),
                "exact_search_complete": bool(plan.complete and not symbolic_failures),
            }
        )
    return sorted(candidates, key=lambda candidate: candidate.objective_value_numeric)


def solve_kkt_candidates(
    problem: LiftedProblem,
    kkt_system=None,
    box_bounds=None,
    tol: float = 1e-7,
    max_active_sets: int = 64,
    cp_results=None,
    diagnostics: list[SolverDiagnostic] | None = None,
    active_set_plan: ActiveSetPlan | None = None,
    working_precision: int | None = None,
):
    """Generate KKT candidates numerically across prioritized active-set patterns.

    Returned points are filtered against the lifted constraints before entering the
    integrated optimization result."""
    kkt = kkt_system or build_kkt_system(problem)
    stationarity, equalities, inequalities, unknowns = _collect_kkt_data(kkt)
    candidates = []
    seen = set()
    temp_unknowns = [sp.Dummy(f"t{index}") for index in range(len(unknowns))]
    substitutions = dict(zip(unknowns, temp_unknowns))
    plan = active_set_plan or build_active_set_plan(
        inequalities,
        kkt.primal_variables,
        max_active_sets,
        box_bounds,
        tol,
        equalities,
        diagnostics,
    )
    patterns = plan.patterns

    for pattern in patterns:
        equations = [*stationarity, *equalities]
        for index, active in enumerate(pattern):
            equations.append(
                inequalities[index] if active else kkt.inequality_multipliers[index]
            )
        temp_equations = [sp.expand(eq.xreplace(substitutions)) for eq in equations]

        for mode in ("mid", "low", "high"):
            seed = _seed_values(kkt, box_bounds, mode)
            try:
                root = sp.nsolve(
                    temp_equations,
                    temp_unknowns,
                    seed,
                    tol=10 ** (-min(14, max(8, int((working_precision or 30) // 2)))),
                    maxsteps=100,
                    verify=False,
                    prec=max(int(working_precision or 30), 30),
                )
            except (ValueError, ZeroDivisionError, TypeError) as exc:
                _note(
                    diagnostics,
                    "kkt_candidates",
                    f"nsolve_{''.join(map(str, pattern))}_{mode}",
                    DiagnosticKind.NUMERICAL_FAILURE,
                    exc,
                )
                continue

            values = list(root) if hasattr(root, "__len__") else [root]
            if len(values) != len(unknowns):
                continue
            candidate = _filter_candidate(
                problem,
                kkt,
                dict(zip(unknowns, values)),
                "nsolve_active_" + "".join(map(str, pattern)),
                tol,
                critical_results=cp_results,
                working_precision=working_precision,
            )
            if candidate is None:
                continue
            key = tuple(
                round(real_part(candidate.primal_values[var]), 10)
                for var in kkt.primal_variables
            )
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)

    return sorted(candidates, key=lambda candidate: candidate.objective_value_numeric)


def dedupe_candidates(candidates):
    """Merge numerically coincident candidate proposals and sort them by objective value."""
    unique = {}
    for candidate in candidates:
        ordered = sorted(candidate.primal_values, key=sp.sstr)
        if candidate.objective_value_numeric is None or any(
            getattr(value, "free_symbols", ())
            for value in candidate.primal_values.values()
        ):
            key = (
                candidate.objective_value,
                tuple((v, candidate.primal_values[v]) for v in ordered),
            )
        else:
            key = tuple(
                (
                    v,
                    round(candidate.objective_value_numeric, 12),
                    round(real_part(candidate.primal_values[v]), 10),
                )
                for v in ordered
            )
        unique[key] = candidate
    return sorted(
        unique.values(),
        key=lambda candidate: (
            float("inf")
            if candidate.objective_value_numeric is None
            else candidate.objective_value_numeric
        ),
    )


def select_best_candidate(candidates):
    """Choose the least objective candidate using exact comparison when available."""
    if not candidates:
        return None
    best = candidates[0]
    for candidate in candidates[1:]:
        comparison = algebraic_compare(
            candidate.objective_value,
            best.objective_value,
        )
        if (comparison is not None and comparison < 0) or (
            comparison is None
            and candidate.objective_value_numeric is not None
            and best.objective_value_numeric is not None
            and candidate.objective_value_numeric < best.objective_value_numeric
        ):
            best = candidate
    return best
