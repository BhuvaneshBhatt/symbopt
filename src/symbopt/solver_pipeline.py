from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple
from itertools import product
from copy import copy as shallow_copy

from dataclasses import is_dataclass, replace
import sympy as sp
from sympy.core.relational import Relational

from .algebraic_numbers import exactify, exactify_dict
from .algebraize import algebraize, algebraize_cases
from .candidates import (
    build_active_set_plan,
    dedupe_candidates,
    select_best_candidate,
    solve_exact_candidates,
    solve_kkt_candidates,
)
from .certification import build_certificate
from .diagnostics import DiagnosticKind, SolverDiagnostic
from .decomposition import find_components
from .hierarchy import search_sos_hierarchy
from .intervals import interval_optimize
from .kkt import build_kkt_system
from .problem_models import (
    CandidateSolution,
    PipelineResult,
    ProofArtifact,
)
from .numeric_utils import real_part
from .precision import decimal_digits_to_bits, precision_provenance
from .convex_routing import (
    complete_box_bounds,
    infer_box_bounds,
    try_direct_solver,
)
from .structural_results import StructuralGlobalResult
from .structural_routing import try_structural_solver
from .objective_transforms import find_monotone_reduction
from .sos import sos_lower_bound as compute_sos_bound
from .semialgebraic import (
    certify_global_min,
    exact_feasible,
    function_range_exact,
    get_semialg_capabilities,
    minimize_exact,
    prove_nonnegative_exact,
    prune_redundant,
)
from .polynomial_structure import build_sparsity_graph, reduce_sign_symmetry
from .relation_checks import normalize_relational


def _skip_semialg_pruning(objective, constraints, variables) -> bool:
    """Return whether affine constraints can go directly to LP/QP routing.

    Exact semialgebraic redundancy elimination is unnecessary for linear and
    quadratic programs because the direct solvers accept redundant rows.
    Avoiding those implication queries is especially important when many affine
    constraints are present.
    """
    try:
        obj_degree = sp.Poly(sp.expand(objective), *variables).total_degree()
    except (sp.PolynomialError, TypeError, ValueError):
        return False
    if obj_degree > 2:
        return False
    for relation in constraints:
        if not isinstance(relation, Relational):
            return False
        try:
            _, residual = normalize_relational(relation)
            degree = sp.Poly(sp.expand(residual), *variables).total_degree()
        except (sp.PolynomialError, TypeError, ValueError):
            return False
        if degree > 1:
            return False
    return True


def _defer_semialg_auto(problem, variables, certification: str) -> bool:
    """Prefer symbolic candidates for small nonlinear polynomial systems.

    Complete CAD-style optimization can be disproportionately expensive for
    low-dimensional polynomial problems with high-degree nonlinear boundaries.
    In automatic mode, let the symbolic KKT path handle these cases first.
    Explicit ``candidate`` or ``complete`` policies still force the semialg
    optimizer route.
    """
    if certification != "auto" or len(variables) < 2 or len(variables) > 2:
        return False
    try:
        sp.Poly(sp.expand(problem.original_objective), *variables)
    except (sp.PolynomialError, TypeError, ValueError):
        return False

    max_degree = 0
    for relation in problem.original_constraints:
        if not isinstance(relation, Relational):
            return False
        try:
            _, residual = normalize_relational(relation)
            degree = sp.Poly(sp.expand(residual), *variables).total_degree()
        except (sp.PolynomialError, TypeError, ValueError):
            return False
        if degree is sp.S.NegativeInfinity:
            degree = 0
        max_degree = max(max_degree, int(degree))
    return max_degree >= 3


def _exact_branch_infimum(result):
    certificate = result.certificate
    if certificate is None or certificate.lower_bound is None:
        return None
    if certificate.status not in {"proven_optimal", "proven_infimum"}:
        return None
    return sp.simplify(certificate.lower_bound)


def _merge_piecewise_results(results):
    """Combine branch bounds without treating a branch proof as a global proof."""
    if not results:
        raise ValueError("At least one Piecewise branch result is required.")

    candidates = []
    for result in results:
        candidates.extend(result.candidates)
    candidates = dedupe_candidates(candidates)
    best_candidate = select_best_candidate(candidates)

    exact_infima = [_exact_branch_infimum(result) for result in results]
    exact_pairs = [
        (value, result)
        for value, result in zip(exact_infima, results)
        if value is not None
    ]

    selected = None
    if best_candidate is not None:
        for result in results:
            if result.best_candidate is best_candidate:
                selected = result
                break
        if selected is None:
            selected = min(
                (result for result in results if result.best_candidate is not None),
                key=lambda result: result.best_candidate.objective_value_numeric,
            )

    if exact_pairs:
        exact_value, exact_result = min(
            exact_pairs,
            key=lambda item: real_part(item[0]),
        )
        if (
            best_candidate is None
            or real_part(exact_value) < best_candidate.objective_value_numeric
        ):
            selected = exact_result

    if selected is None:
        selected = results[0]

    if is_dataclass(selected):
        combined = replace(
            selected,
            candidates=list(candidates),
            best_candidate=best_candidate,
            diagnostics=list(getattr(selected, "diagnostics", ())),
            critical_point_results=list(
                getattr(selected, "critical_point_results", ())
            ),
            inferred_box_bounds=dict(getattr(selected, "inferred_box_bounds", {})),
            precision_provenance=dict(getattr(selected, "precision_provenance", {})),
        )
    else:
        combined = shallow_copy(selected)
        combined.candidates = list(candidates)
        combined.best_candidate = best_candidate

    lower_bounds = [
        result.certificate.lower_bound
        for result in results
        if result.certificate is not None and result.certificate.lower_bound is not None
    ]
    all_have_lower = len(lower_bounds) == len(results)
    global_lower = (
        min(lower_bounds, key=real_part) if all_have_lower and lower_bounds else None
    )
    global_upper = (
        best_candidate.objective_value if best_candidate is not None else None
    )

    all_infima_exact = all(value is not None for value in exact_infima)
    global_infimum = (
        min(exact_infima, key=real_part) if all_infima_exact and exact_infima else None
    )
    attained = (
        global_infimum is not None
        and best_candidate is not None
        and sp.simplify(best_candidate.objective_value - global_infimum) == 0
    )

    certificate = selected.certificate
    if certificate is None:
        certificate = build_certificate(best_candidate=best_candidate)
    else:
        certificate = replace(
            certificate,
            primal_solution=dict(certificate.primal_solution),
            proof_objects=list(certificate.proof_objects),
        )
    certificate.primal_solution = (
        dict(best_candidate.primal_values) if best_candidate is not None else {}
    )
    certificate.primal_value = (
        best_candidate.objective_value if best_candidate is not None else None
    )
    certificate.lower_bound = global_infimum if all_infima_exact else global_lower
    certificate.upper_bound = global_upper

    combined.certified = bool(all_infima_exact)
    if all_infima_exact:
        combined.certification_gap = 0.0 if attained else None
        combined.status = (
            "global_piecewise_optimal" if attained else "global_piecewise_infimum"
        )
        certificate.status = "proven_optimal" if attained else "proven_infimum"
    else:
        combined.certified = False
        combined.status = "piecewise_candidate_or_bound"
        if global_lower is not None and global_upper is not None:
            combined.certification_gap = float(sp.N(global_upper - global_lower, 30))
            certificate.gap = sp.simplify(global_upper - global_lower)
            certificate.status = "bounded"
        else:
            combined.certification_gap = None
            certificate.gap = None
            certificate.status = (
                "candidate_only" if global_upper is not None else "unknown"
            )

    certificate.proof_objects.append(
        ProofArtifact(
            "piecewise",
            "Combined Piecewise branch bounds",
            {
                "branch_count": len(results),
                "exact_branch_infima": [
                    sp.sstr(value) if value is not None else None
                    for value in exact_infima
                ],
                "all_branch_infima_exact": all_infima_exact,
            },
        )
    )
    combined.certificate = certificate
    combined.optimum_value = sp.simplify(global_infimum) if all_infima_exact else None
    combined.attained = bool(attained) if all_infima_exact else None
    return combined


def _original_variables(problem):
    return list(problem.original_variables)


def _pipeline_explain(
    problem,
    direct,
    candidates,
    interval_result,
    sos_result,
    certificate,
    semialg_result=None,
    symmetry_report=None,
    sparsity_components=None,
    diagnostics=(),
):
    lines = []
    if direct is not None:
        lines.append(f"direct route chosen: {direct.kind}")
    else:
        lines.append(
            f"algebraized polynomial problem: {problem.is_polynomial_problem()}"
        )
        lines.append(f"num candidates: {len(candidates)}")
        if interval_result is not None:
            lines.append(
                "interval bounds: "
                f"[{interval_result.best_lower_bound}, "
                f"{interval_result.best_upper_bound}]"
            )
        if sos_result is not None:
            lines.append(f"sos success: {getattr(sos_result, 'success', None)}")

    if semialg_result is not None:
        lines.append(
            "semialg exact optimizer: "
            f"method={semialg_result.method}, "
            f"certified={semialg_result.certified}, "
            f"attained={semialg_result.attained}"
        )

    if symmetry_report is not None:
        lines.append(
            f"even symmetries: {[sp.sstr(v) for v in symmetry_report.even_variables]}"
        )
        lines.append(
            "permutation symmetries: "
            f"{[(sp.sstr(a), sp.sstr(b)) for a, b in symmetry_report.permutation_pairs]}"
        )

    if sparsity_components is not None:
        lines.append(
            "correlative sparsity components: "
            f"{[[sp.sstr(v) for v in component] for component in sparsity_components]}"
        )

    if certificate is not None:
        lines.append(f"certificate status: {certificate.status}")
    if diagnostics:
        lines.append(f"solver diagnostics: {len(diagnostics)}")
    return lines


def _prepare_single_problem(problem, box_bounds):
    """Apply bound inference and safe symmetry reduction once for one problem."""
    merged = complete_box_bounds(
        problem,
        infer_box_bounds(problem.all_variables, problem.all_constraints()),
    )
    if box_bounds:
        merged.update(box_bounds)

    original_vars = _original_variables(problem)
    added_constraints, merged, symmetry_report = reduce_sign_symmetry(
        problem.original_objective,
        problem.original_constraints,
        original_vars,
        merged,
    )
    if added_constraints:
        problem = algebraize(
            problem.original_objective,
            [*problem.original_constraints, *added_constraints],
            variables=original_vars,
        )
        merged = complete_box_bounds(
            problem,
            infer_box_bounds(problem.all_variables, problem.all_constraints()),
        )
        if box_bounds:
            merged.update(box_bounds)
    return problem, merged, original_vars, symmetry_report


def _run_early_routes(
    problem,
    original_vars,
    diagnostics,
    semialg_certification,
    semialg_range_cost_limit,
    working_precision,
):
    """Run global routes that can avoid candidate, SOS, and interval searches."""
    structural_result = try_structural_solver(
        problem.original_objective,
        problem.original_constraints,
        original_vars,
        working_precision=working_precision,
    )
    direct = None
    if structural_result is None:
        direct = try_direct_solver(
            problem.original_objective,
            problem.original_constraints,
            original_vars,
            diagnostics=diagnostics,
            working_precision=working_precision,
        )

    semialg_result = None
    semialg_attempted = False
    if direct is None and structural_result is None and not problem.parameter_symbols:
        deferred = _defer_semialg_auto(problem, original_vars, semialg_certification)
        semialg_attempted = deferred
        try:
            capabilities = get_semialg_capabilities()
        except ImportError:
            capabilities = None
        if not deferred:
            semialg_result = minimize_exact(
                problem.original_objective,
                problem.original_constraints,
                original_vars,
                diagnostics,
                certification=semialg_certification,
                range_cost_limit=semialg_range_cost_limit,
            )
            if capabilities is not None:
                semialg_attempted = bool(
                    capabilities.has_optimizer and capabilities.supports_cert_policy
                )
    return structural_result, direct, semialg_result, semialg_attempted


def _search_kkt_stage(
    problem,
    merged,
    max_active_sets,
    diagnostics,
    candidates,
):
    """Build KKT data lazily and enumerate exact/numerical candidates."""
    kkt = build_kkt_system(problem)
    critical_results = []
    equalities, inequalities = problem.kkt_residuals()
    active_set_plan = build_active_set_plan(
        inequalities,
        kkt.primal_variables,
        max_active_sets,
        box_bounds=merged,
        tol=1e-7,
        equalities=equalities,
        diagnostics=diagnostics,
    )
    exact_meta = {}
    try:
        candidates.extend(
            solve_exact_candidates(
                problem,
                kkt_system=kkt,
                box_bounds=merged,
                tol=1e-7,
                max_active_sets=max_active_sets,
                cp_results=critical_results,
                diagnostics=diagnostics,
                active_set_plan=active_set_plan,
                exact_search_metadata=exact_meta,
            )
        )
    except (ValueError, TypeError, NotImplementedError) as exc:
        exact_meta["exact_search_complete"] = False
        diagnostics.append(
            SolverDiagnostic.from_exception(
                "symbolic_candidates",
                "candidate_generation",
                DiagnosticKind.SYMBOLIC_FAILURE,
                exc,
            )
        )

    if not exact_meta.get("exact_search_complete", False):
        try:
            candidates.extend(
                solve_kkt_candidates(
                    problem,
                    kkt_system=kkt,
                    box_bounds=merged,
                    tol=1e-7,
                    max_active_sets=max_active_sets,
                    cp_results=critical_results,
                    diagnostics=diagnostics,
                    active_set_plan=active_set_plan,
                )
            )
        except (ValueError, TypeError, NotImplementedError) as exc:
            diagnostics.append(
                SolverDiagnostic.from_exception(
                    "kkt_candidates",
                    "candidate_generation",
                    DiagnosticKind.NUMERICAL_FAILURE,
                    exc,
                )
            )
    candidates = dedupe_candidates(candidates)
    return kkt, critical_results, candidates, select_best_candidate(candidates)


def _run_sos_stage(problem, best, diagnostics):
    """Compute polynomial sparsity metadata and a rigorous SOS lower bound."""
    symbols = problem.all_variables
    sparsity_components = None
    try:
        graph_exprs = [problem.lifted_objective, *problem.constraint_residuals]
        sparsity_components = build_sparsity_graph(
            graph_exprs, symbols
        ).connected_components()
    except (ValueError, TypeError) as exc:
        diagnostics.append(
            SolverDiagnostic.from_exception(
                "structure",
                "sparsity_analysis",
                DiagnosticKind.UNSUPPORTED,
                exc,
            )
        )

    equalities, inequalities = problem.sos_residuals()
    sos_result = None
    sos_bound = None
    sos_numeric = None
    try:
        upper_guess = float(best.objective_value_numeric) if best is not None else 100.0
        hierarchy = search_sos_hierarchy(
            problem.lifted_objective,
            symbols,
            inequalities=inequalities,
            equalities=equalities,
            search_hi=upper_guess,
            max_order=3,
        )
        if hierarchy.best_result is not None:
            sos_result = hierarchy.best_result.sos_result
            if hierarchy.certified and getattr(
                hierarchy.best_result, "certified", False
            ):
                sos_bound = hierarchy.best_result.lower_bound
                sos_numeric = hierarchy.best_result.lower_bound_numeric
    except (ValueError, TypeError, ArithmeticError) as exc:
        diagnostics.append(
            SolverDiagnostic.from_exception(
                "sos_hierarchy",
                "hierarchy_search",
                DiagnosticKind.NUMERICAL_FAILURE,
                exc,
            )
        )
        try:
            candidate_bound, candidate_sos = compute_sos_bound(
                problem.lifted_objective, symbols
            )
            sos_result = candidate_sos
            certificate_obj = getattr(candidate_sos, "certificate", None)
            if (
                candidate_sos.success
                and certificate_obj is not None
                and certificate_obj.exact_residual_zero
                and all(check is True for check in certificate_obj.exact_psd_checks)
            ):
                sos_bound = candidate_bound
                sos_numeric = float(sp.N(candidate_bound, 30))
        except (ValueError, TypeError, ArithmeticError) as fallback_exc:
            diagnostics.append(
                SolverDiagnostic.from_exception(
                    "sos",
                    "fallback_lower_bound",
                    DiagnosticKind.CERTIFICATE_FAILURE,
                    fallback_exc,
                )
            )
    return sos_result, sos_bound, sos_numeric, sparsity_components


def _run_interval_stage(
    problem,
    merged,
    interval_max_boxes,
    interval_tol,
    working_precision,
    diagnostics,
):
    """Run verified interval branch-and-bound when finite bounds are available."""
    if not merged:
        return None
    try:
        return interval_optimize(
            problem.lifted_objective,
            problem.all_constraints(),
            merged,
            max_boxes=interval_max_boxes,
            tol=interval_tol,
            precision=decimal_digits_to_bits(int(working_precision or 50)),
        )
    except (ValueError, TypeError, ArithmeticError) as exc:
        diagnostics.append(
            SolverDiagnostic.from_exception(
                "interval",
                "branch_and_bound",
                DiagnosticKind.NUMERICAL_FAILURE,
                exc,
            )
        )
        return None


def _certify_search_result(
    problem,
    best,
    interval_result,
    sos_bound,
    semialg_attempted,
    original_vars,
    candidates,
    diagnostics,
):
    """Combine lower-bound evidence and exact semialgebraic fallback proofs."""
    certified = False
    gap = None
    status = "ok"
    semialg_sign = None
    semialg_proof = None
    semialg_range = None

    if best is not None and sos_bound is not None:
        try:
            exact_gap = sp.simplify(best.objective_value - sos_bound)
            if exact_gap == 0:
                certified = True
                gap = 0.0
                status = "global_certified_by_sos"
            else:
                gap = float(sp.N(exact_gap, 30))
                status = "rigorously_bounded_by_sos"
        except (ValueError, TypeError, ArithmeticError) as exc:
            diagnostics.append(
                SolverDiagnostic.from_exception(
                    "certification",
                    "sos_gap",
                    DiagnosticKind.CERTIFICATE_FAILURE,
                    exc,
                )
            )
    elif (
        best is not None
        and interval_result is not None
        and interval_result.lower_bound_rigorous
    ):
        try:
            gap = float(best.objective_value_numeric - interval_result.best_lower_bound)
            status = "rigorously_bounded_by_interval"
        except (ValueError, TypeError, ArithmeticError) as exc:
            diagnostics.append(
                SolverDiagnostic.from_exception(
                    "certification",
                    "interval_gap",
                    DiagnosticKind.CERTIFICATE_FAILURE,
                    exc,
                )
            )

    if (
        not semialg_attempted
        and not problem.parameter_symbols
        and not certified
        and best is not None
        and problem.original_objective.is_polynomial(*original_vars)
    ):
        semialg_sign = prove_nonnegative_exact(
            sp.expand(problem.original_objective - best.objective_value),
            problem.original_constraints,
            original_vars,
            diagnostics,
        )
        if semialg_sign.proven is True:
            certified = True
            gap = 0.0
            status = "global_certified_by_semialg_sign"

    if (
        not semialg_attempted
        and not problem.parameter_symbols
        and not certified
        and best is not None
    ):
        point = best.original_variable_values or best.primal_values
        semialg_proof = certify_global_min(
            problem.original_objective,
            problem.original_constraints,
            original_vars,
            point,
            best.objective_value,
            diagnostics,
        )
        if semialg_proof.proven is True:
            certified = True
            gap = 0.0
            status = "global_certified_by_semialg"

    same_vars = set(problem.all_variables) == set(original_vars)
    use_range = (
        not semialg_attempted
        and not problem.parameter_symbols
        and same_vars
        and not certified
        and (best is None or problem.infimum_only_possible() or not candidates)
    )
    if use_range:
        semialg_range = function_range_exact(
            problem.original_objective,
            problem.original_constraints,
            original_vars,
            diagnostics,
        )
        if semialg_range is not None:
            infimum = semialg_range.infimum
            if infimum is not None and infimum not in (sp.oo, -sp.oo):
                if best is None:
                    certified = True
                    status = "global_infimum_by_semialg_range"
                else:
                    try:
                        same_value = sp.simplify(best.objective_value - infimum) == 0
                    except (TypeError, ValueError, ArithmeticError):
                        same_value = False
                    if same_value:
                        certified = True
                        gap = 0.0
                        status = "global_certified_by_semialg_range"
    return (
        certified,
        gap,
        status,
        semialg_sign,
        semialg_proof,
        semialg_range,
    )


def _solve_single_problem(
    problem,
    box_bounds,
    interval_max_boxes,
    interval_tol,
    max_active_sets,
    use_sos,
    seed_diagnostics=(),
    semialg_certification="auto",
    semialg_range_cost_limit=None,
    working_precision=None,
    precision_provenance=None,
):
    diagnostics = list(seed_diagnostics)
    problem, merged, original_vars, symmetry_report = _prepare_single_problem(
        problem, box_bounds
    )
    structural_result, direct, semialg_result, semialg_attempted = _run_early_routes(
        problem,
        original_vars,
        diagnostics,
        semialg_certification,
        semialg_range_cost_limit,
        working_precision,
    )
    semialg_proof = None
    semialg_sign = None
    semialg_range = None
    kkt = None
    critical_results = []

    candidates = []
    best = None
    sos_result = None
    sos_bound = None
    sos_bound_numeric = None
    interval_result = None
    sparsity_components = None
    status = "ok"
    certified = False
    gap = None
    direct_kind = None

    if structural_result is not None:
        direct_kind = structural_result.kind
        for point in structural_result.points:
            candidate = CandidateSolution(
                primal_values=dict(point),
                multiplier_values={},
                objective_value=sp.simplify(structural_result.value),
                objective_value_numeric=real_part(structural_result.value)
                if structural_result.value not in (sp.oo, -sp.oo)
                else float(structural_result.value),
                feasibility_checked=True,
                source=structural_result.kind,
                local_classification="GlobalMinimum",
                original_variable_values=dict(point),
            )
            candidates.append(candidate)
        candidates = dedupe_candidates(candidates)
        best = select_best_candidate(candidates)
        certified = bool(structural_result.certified)
        gap = 0.0 if certified and structural_result.attained else None
        status = (
            f"global_certified_by_{structural_result.kind}"
            if structural_result.attained
            else f"global_infimum_by_{structural_result.kind}"
        )
    elif direct is not None:
        direct_kind = direct.kind
        point = exactify_dict(direct.point)
        objective_value = exactify(direct.objective_value)
        candidate = CandidateSolution(
            primal_values=point,
            multiplier_values=dict(direct.multipliers),
            objective_value=objective_value,
            objective_value_numeric=real_part(objective_value),
            feasibility_checked=bool(direct.metadata.get("exact_feasible", False)),
            source=direct_kind,
            local_classification="LocalMinimum",
            original_variable_values=dict(point),
        )
        candidates = [candidate]
        best = candidate
        certified = bool(direct.metadata.get("rigorous_global", False))
        status = (
            f"global_certified_by_{direct_kind}"
            if certified
            else f"direct_candidate_by_{direct_kind}"
        )
    elif semialg_result is not None and semialg_result.certified is True:
        if semialg_result.attained:
            for point in semialg_result.points:
                candidate = CandidateSolution(
                    primal_values=dict(point),
                    multiplier_values={},
                    objective_value=semialg_result.value,
                    objective_value_numeric=real_part(semialg_result.value),
                    feasibility_checked=True,
                    source="semialg",
                    local_classification="GlobalMinimum",
                    original_variable_values=dict(point),
                )
                candidates.append(candidate)
            candidates = dedupe_candidates(candidates)
            best = select_best_candidate(candidates)
            gap = 0.0
            status = "global_certified_by_semialg"
        else:
            gap = None
            status = "global_infimum_by_semialg"
        certified = True
    else:
        # An uncertified candidate-policy result can still seed the local
        # pipeline, but it is not treated as a lower-bound certificate.
        if semialg_result is not None and semialg_result.attained:
            for point in semialg_result.points:
                candidates.append(
                    CandidateSolution(
                        primal_values=dict(point),
                        multiplier_values={},
                        objective_value=semialg_result.value,
                        objective_value_numeric=real_part(semialg_result.value),
                        feasibility_checked=True,
                        source="semialg_candidate",
                        local_classification=None,
                        original_variable_values=dict(point),
                    )
                )

        kkt, critical_results, candidates, best = _search_kkt_stage(
            problem, merged, max_active_sets, diagnostics, candidates
        )

        precision_allows_float64 = bool(
            (precision_provenance or {}).get("float64_backends_allowed", True)
        )
        if use_sos and problem.is_polynomial_problem() and precision_allows_float64:
            sos_result, sos_bound, sos_bound_numeric, sparsity_components = (
                _run_sos_stage(problem, best, diagnostics)
            )

        interval_result = _run_interval_stage(
            problem,
            merged,
            interval_max_boxes,
            interval_tol,
            working_precision,
            diagnostics,
        )

        (
            certified,
            gap,
            status,
            semialg_sign,
            semialg_proof,
            semialg_range,
        ) = _certify_search_result(
            problem,
            best,
            interval_result,
            sos_bound,
            semialg_attempted,
            original_vars,
            candidates,
            diagnostics,
        )

    if problem.infimum_only_possible():
        if structural_result is not None and structural_result.certified:
            status = (
                f"global_certified_by_{structural_result.kind}"
                if structural_result.attained
                else f"global_infimum_by_{structural_result.kind}"
            )
        elif semialg_result is not None and semialg_result.certified is True:
            # semialg's exact attained flag is authoritative for supported
            # semialgebraic problems; do not overwrite it with a closure guess.
            status = (
                "global_certified_by_semialg"
                if semialg_result.attained
                else "global_infimum_by_semialg"
            )
        elif certified and status.startswith("global_certified"):
            status += "_on_closure"
        elif status == "ok":
            status = "strict_constraints_present"

    sos_rigorous = (
        sos_bound is not None
        and sos_result is not None
        and getattr(
            getattr(sos_result, "certificate", None), "exact_residual_zero", False
        )
        and all(
            check is True
            for check in getattr(
                getattr(sos_result, "certificate", None),
                "exact_psd_checks",
                [],
            )
        )
    )
    certificate = build_certificate(
        best_candidate=best,
        interval_result=interval_result,
        sos_result=sos_result,
        sos_lower_bound=sos_bound,
        sos_bound_numeric=sos_bound_numeric,
        lp_qp_kind=(direct_kind if direct_kind in {"lp", "qp"} else None),
        constraints=problem.original_constraints,
        sos_rigorous=sos_rigorous,
        direct_metadata=(direct.metadata if direct is not None else None),
        semialg_result=semialg_result,
        semialg_proof=semialg_proof,
        semialg_sign=semialg_sign,
        semialg_range=semialg_range,
    )

    if structural_result is not None:
        certificate.proof_objects.append(
            ProofArtifact(
                kind=structural_result.kind,
                summary=f"Exact structural global route: {structural_result.kind}",
                payload=dict(structural_result.metadata),
            )
        )
        if structural_result.certified:
            certificate.lower_bound = sp.simplify(structural_result.value)
            certificate.upper_bound = (
                sp.simplify(structural_result.value)
                if structural_result.attained
                else certificate.upper_bound
            )
            certificate.status = (
                "proven_optimal" if structural_result.attained else "proven_infimum"
            )

    if best is not None and sos_result is not None:
        try:
            from .moment_bridge import link_sos_moments

            inequalities = problem.branch_residuals()
            bridge = link_sos_moments(
                problem.lifted_objective,
                inequalities,
                problem.all_variables,
                [best.primal_values],
                order=2,
            )
            certificate.proof_objects.append(
                ProofArtifact(
                    kind="moment_bridge",
                    summary="SOS-moment consistency report",
                    payload={
                        "message": bridge.message,
                        "num_hints": len(bridge.candidate_hints),
                        "dual_verified": (
                            getattr(
                                bridge.dual_result,
                                "verified",
                                None,
                            )
                            if bridge.dual_result is not None
                            else None
                        ),
                    },
                )
            )
        except (ValueError, TypeError, ArithmeticError) as exc:
            diagnostics.append(
                SolverDiagnostic.from_exception(
                    "moments",
                    "moment_bridge",
                    DiagnosticKind.CERTIFICATE_FAILURE,
                    exc,
                )
            )

    certificate.proof_objects.append(
        ProofArtifact(
            kind="explain",
            summary="Pipeline decision trace",
            payload={
                "diagnostics": [
                    {
                        "solver": item.solver,
                        "stage": item.stage,
                        "kind": item.kind.value,
                        "message": item.message,
                        "exception_type": item.exception_type,
                    }
                    for item in diagnostics
                ],
                "lines": _pipeline_explain(
                    problem,
                    direct,
                    candidates,
                    interval_result,
                    sos_result,
                    certificate,
                    semialg_result=semialg_result,
                    symmetry_report=symmetry_report,
                    sparsity_components=sparsity_components,
                    diagnostics=diagnostics,
                ),
            },
        )
    )

    if certified and certificate.status not in {
        "proven_optimal",
        "proven_infimum",
    }:
        certificate.status = "proven_optimal" if gap == 0 else "bounded"

    return PipelineResult(
        lifted_problem=problem,
        kkt_system=kkt,
        candidates=candidates,
        best_candidate=best,
        interval_result=interval_result,
        inferred_box_bounds=merged,
        certified=certified,
        certification_gap=gap,
        status=status,
        critical_point_results=critical_results,
        sos_result=sos_result,
        sos_lower_bound=sos_bound,
        sos_bound_numeric=sos_bound_numeric,
        certificate=certificate,
        diagnostics=diagnostics,
        semialg_result=semialg_result,
        semialg_range=semialg_range,
        optimum_value=(
            sp.simplify(structural_result.value)
            if structural_result is not None and structural_result.certified
            else (
                sp.simplify(semialg_result.value)
                if semialg_result is not None and semialg_result.certified is True
                else (
                    sp.simplify(best.objective_value)
                    if certified and best is not None
                    else None
                )
            )
        ),
        attained=(
            structural_result.attained
            if structural_result is not None and structural_result.certified
            else (
                semialg_result.attained
                if semialg_result is not None and semialg_result.certified is True
                else (True if certified and best is not None else None)
            )
        ),
        working_precision=working_precision,
        precision_provenance=dict(precision_provenance or {}),
        structural_result=structural_result,
        optimization_sense="minimize",
        kkt_objective=(problem.lifted_objective if kkt is not None else None),
    )


def _optimal_points(result):
    if (
        result.certified is not True
        or result.attained is not True
        or result.optimum_value is None
    ):
        return ()
    points = []
    for candidate in result.candidates:
        if sp.simplify(candidate.objective_value - result.optimum_value) != 0:
            continue
        point = candidate.original_variable_values or candidate.primal_values
        if point and point not in points:
            points.append(dict(point))
    structural = getattr(result, "structural_result", None)
    for point in getattr(structural, "points", ()) or ():
        point = dict(point)
        if point and point not in points:
            points.append(point)
    return tuple(points)


def _merge_separable_results(
    objective,
    constraints,
    variables,
    decomposition,
    results,
    precision_info,
):
    """Build one certified result from independently certified components."""
    if not results or not all(
        result.certified and result.optimum_value is not None for result in results
    ):
        return None
    value = sp.simplify(
        decomposition.constant + sp.Add(*[result.optimum_value for result in results])
    )
    attained = all(result.attained is True for result in results)
    component_points = [_optimal_points(result) for result in results]
    combined_points = []
    if attained and all(component_points):
        for combination in product(*component_points):
            point = {}
            for part in combination:
                point.update(part)
            combined_points.append(point)
            if len(combined_points) >= 64:
                break

    problem = algebraize(objective, constraints, variables=variables)
    candidates = []
    for point in combined_points:
        candidates.append(
            CandidateSolution(
                primal_values=dict(point),
                multiplier_values={},
                objective_value=value,
                objective_value_numeric=real_part(
                    value, precision_info.effective_precision
                ),
                feasibility_checked=True,
                source="separable_decomposition",
                local_classification="GlobalMinimum",
                original_variable_values=dict(point),
            )
        )
    best = candidates[0] if candidates else None
    structural = StructuralGlobalResult(
        kind="separable_decomposition",
        value=value,
        points=tuple(combined_points),
        attained=attained,
        certified=True,
        metadata={
            "constant": sp.sstr(decomposition.constant),
            "components": [
                {
                    "variables": [sp.sstr(var) for var in component.variables],
                    "objective": sp.sstr(component.objective),
                    "constraints": [sp.sstr(item) for item in component.constraints],
                    "status": result.status,
                    "value": sp.sstr(result.optimum_value),
                }
                for component, result in zip(decomposition.components, results)
            ],
        },
    )
    certificate = build_certificate(best_candidate=best)
    certificate.lower_bound = value
    certificate.upper_bound = value if attained else None
    certificate.gap = sp.Integer(0) if attained else None
    certificate.status = "proven_optimal" if attained else "proven_infimum"
    certificate.proof_objects.append(
        ProofArtifact(
            kind="separable_decomposition",
            summary="Independent-variable decomposition into certified subproblems",
            payload=dict(structural.metadata),
        )
    )
    diagnostics = [diag for result in results for diag in result.diagnostics]
    inferred = {}
    for result in results:
        inferred.update(result.inferred_box_bounds)
    return PipelineResult(
        lifted_problem=problem,
        kkt_system=None,
        candidates=candidates,
        best_candidate=best,
        interval_result=None,
        inferred_box_bounds=inferred,
        certified=True,
        certification_gap=0.0 if attained else None,
        status="global_certified_by_separable_decomposition"
        if attained
        else "global_infimum_by_separable_decomposition",
        certificate=certificate,
        diagnostics=diagnostics,
        optimum_value=value,
        attained=attained,
        working_precision=precision_info.effective_precision,
        precision_provenance=precision_info.as_dict(),
        structural_result=structural,
        optimization_sense="minimize",
        kkt_objective=None,
    )


def _map_monotone_result(
    subresult,
    objective,
    constraints,
    reduction,
    effective_precision,
    precision_info,
):
    """Map a certified monotone reduction back without cloning the result graph."""
    mapped_candidates = []
    for candidate in subresult.candidates:
        point = candidate.original_variable_values or candidate.primal_values
        value = sp.simplify(objective.subs(point))
        mapped_candidates.append(
            replace(
                candidate,
                primal_values=dict(candidate.primal_values),
                multiplier_values=dict(candidate.multiplier_values),
                original_variable_values=dict(candidate.original_variable_values),
                objective_value=value,
                objective_value_numeric=real_part(value, effective_precision),
                source=f"monotone_{reduction.kind}:{candidate.source}",
            )
        )
    mapped_best = select_best_candidate(mapped_candidates)
    optimum = subresult.optimum_value
    if optimum is not None:
        optimum = sp.simplify(reduction.recover_value(optimum))

    certificate = subresult.certificate
    if certificate is not None:
        proof_objects = list(certificate.proof_objects)
        proof_objects.append(
            ProofArtifact(
                kind="monotone_reduction",
                summary=f"Monotone objective reduction: {reduction.kind}",
                payload={
                    "original_objective": sp.sstr(objective),
                    "reduced_objective": sp.sstr(reduction.reduced_objective),
                    "added_constraints": [
                        sp.sstr(item) for item in reduction.added_constraints
                    ],
                    "increasing": reduction.increasing,
                },
            )
        )
        primal_solution = dict(certificate.primal_solution)
        primal_value = certificate.primal_value
        if mapped_best is not None:
            primal_solution = dict(
                mapped_best.original_variable_values or mapped_best.primal_values
            )
            primal_value = mapped_best.objective_value
        lower_bound = certificate.lower_bound
        upper_bound = certificate.upper_bound
        if optimum is not None and subresult.certified:
            lower_bound = optimum
            if subresult.attained:
                upper_bound = optimum
        certificate = replace(
            certificate,
            primal_solution=primal_solution,
            primal_value=primal_value,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            proof_objects=proof_objects,
        )

    lifted = subresult.lifted_problem
    notes = list(lifted.preprocessing_notes)
    notes.append(f"monotone reduction ({reduction.kind})")
    mapped_problem = replace(
        lifted,
        original_objective=objective,
        original_constraints=list(constraints),
        lifted_constraints=list(lifted.lifted_constraints),
        defining_equalities=list(lifted.defining_equalities),
        branch_inequalities=list(lifted.branch_inequalities),
        strict_inequalities=list(lifted.strict_inequalities),
        graph_equalities=list(lifted.graph_equalities),
        graph_inequalities=list(lifted.graph_inequalities),
        auxiliary_variables=list(lifted.auxiliary_variables),
        substitution_map=dict(lifted.substitution_map),
        all_variables=list(lifted.all_variables),
        lift_records=list(lifted.lift_records),
        preprocessing_notes=notes,
        strict_closure_warnings=list(lifted.strict_closure_warnings),
        original_variables=list(lifted.original_variables),
        parameter_symbols=list(lifted.parameter_symbols),
        structural_metadata=dict(lifted.structural_metadata),
    )
    return replace(
        subresult,
        lifted_problem=mapped_problem,
        candidates=mapped_candidates,
        best_candidate=mapped_best,
        certificate=certificate,
        diagnostics=list(subresult.diagnostics),
        critical_point_results=list(subresult.critical_point_results),
        inferred_box_bounds=dict(subresult.inferred_box_bounds),
        precision_provenance=precision_info.as_dict(),
        optimum_value=optimum,
        working_precision=effective_precision,
    )


def minimize(
    objective: sp.Expr,
    constraints: Sequence[Relational] | None = None,
    box_bounds: Optional[Dict[sp.Symbol, Tuple[float, float]]] = None,
    variables: Optional[Sequence[sp.Symbol]] = None,
    allow_transcendentals: bool = True,
    allow_rootof: bool = True,
    interval_max_boxes: int = 200,
    interval_tol: float = 1e-3,
    max_active_sets: int = 64,
    use_sos: bool = True,
    semialg_certification: str = "auto",
    semialg_range_cost_limit: Optional[int] = None,
    working_precision: Optional[int] = None,
    _skip_monotone_reduction: bool = False,
    _skip_decomposition: bool = False,
) -> PipelineResult:
    """Minimize a continuous SymPy objective with automatic structure-aware routing.

    The dispatcher recognizes linear and convex-quadratic problems, symbolic
    critical/KKT systems, polynomial SOS/moment opportunities, algebraized branch
    problems, and verified interval problems. The returned ``PipelineResult``
    separates the best candidate from rigorous certification evidence.

    Parameters
    ----------
    objective:
        SymPy expression to minimize.
    constraints:
        Optional SymPy relational constraints.
    box_bounds:
        Optional finite numerical bounds for interval reasoning.
    variables:
        Explicit optimization variables. Symbols appearing elsewhere are treated
        as parameters rather than silently becoming decision variables.
    allow_transcendentals, allow_rootof:
        Enable supported algebraization families.
    interval_max_boxes, interval_tol:
        Branch-and-bound search controls.
    max_active_sets:
        Budget for KKT active-set patterns.
    use_sos:
        Enable polynomial SOS lower-bound search.
    semialg_certification:
        Exact semialgebraic backend policy: ``"auto"``, ``"candidate"``, or
        ``"complete"`` when the installed semialg API supports it.
    semialg_range_cost_limit:
        Optional semialg CAD-range cost limit. ``None`` preserves the backend
        default.
    working_precision:
        Requested decimal working precision for symbolic/numerical fallback work.
        Exact routes remain exact; float64-only backends are reported explicitly
        in ``precision_provenance``.

    Returns
    -------
    PipelineResult
        Candidates, method diagnostics, bounds, and proof artifacts."""
    if semialg_certification not in {"auto", "candidate", "complete"}:
        raise ValueError(
            "semialg_certification must be 'auto', 'candidate', or 'complete'"
        )
    if semialg_range_cost_limit is not None and semialg_range_cost_limit < 0:
        raise ValueError("semialg_range_cost_limit must be nonnegative")

    objective = sp.sympify(objective)
    constraints = list(constraints or [])
    pre_diags = []
    symbols = set(objective.free_symbols)
    for relation in constraints:
        symbols.update(getattr(relation, "free_symbols", set()))
    if variables is None:
        original_vars = sorted(symbols, key=lambda item: item.sort_key())
    else:
        if isinstance(variables, sp.Symbol) or not isinstance(variables, Sequence):
            raise TypeError("variables must be a sequence of unique SymPy Symbols")
        original_vars = list(variables)
        if any(not isinstance(var, sp.Symbol) for var in original_vars):
            raise TypeError("every optimization variable must be a SymPy Symbol")
        if len(set(original_vars)) != len(original_vars):
            raise ValueError("optimization variables must be unique")
        unknown = set(original_vars) - symbols
        if unknown:
            raise ValueError(
                "optimization variables must occur in the objective or constraints: "
                + ", ".join(map(str, sorted(unknown, key=sp.sstr)))
            )
    parameter_symbols = sorted(
        symbols.difference(original_vars), key=lambda item: item.sort_key()
    )
    precision_info = precision_provenance(objective, constraints, working_precision)
    effective_precision = precision_info.effective_precision

    # Reduce globally monotone outer functions to a simpler equivalent objective
    # before algebraization. This exposes polynomial interiors directly to LP/QP
    # and semialg rather than introducing unnecessary transcendental lifts.
    if not _skip_monotone_reduction:
        reduction = find_monotone_reduction(objective, constraints, original_vars)
        if reduction is not None:
            subresult = minimize(
                reduction.reduced_objective,
                [*constraints, *reduction.added_constraints],
                box_bounds=box_bounds,
                variables=original_vars,
                allow_transcendentals=allow_transcendentals,
                allow_rootof=allow_rootof,
                interval_max_boxes=interval_max_boxes,
                interval_tol=interval_tol,
                max_active_sets=max_active_sets,
                use_sos=use_sos,
                semialg_certification=semialg_certification,
                semialg_range_cost_limit=semialg_range_cost_limit,
                working_precision=working_precision,
                _skip_monotone_reduction=True,
                _skip_decomposition=_skip_decomposition,
            )
            mapped = _map_monotone_result(
                subresult,
                objective,
                constraints,
                reduction,
                effective_precision,
                precision_info,
            )
            if mapped.certified:
                return mapped
            # If the simpler problem could not be globally certified, retain its
            # work only as a heuristic and let the original objective reach the
            # specialized range/asymptotic routes below.

    if not _skip_decomposition and not parameter_symbols:
        decomposition = find_components(objective, constraints, original_vars)
        if decomposition is not None:
            component_results = []
            for component in decomposition.components:
                component_box = None
                if box_bounds:
                    component_box = {
                        var: bounds
                        for var, bounds in box_bounds.items()
                        if var in component.variables
                    }
                component_results.append(
                    minimize(
                        component.objective,
                        component.constraints,
                        box_bounds=component_box,
                        variables=component.variables,
                        allow_transcendentals=allow_transcendentals,
                        allow_rootof=allow_rootof,
                        interval_max_boxes=interval_max_boxes,
                        interval_tol=interval_tol,
                        max_active_sets=max_active_sets,
                        use_sos=use_sos,
                        semialg_certification=semialg_certification,
                        semialg_range_cost_limit=semialg_range_cost_limit,
                        working_precision=working_precision,
                        _skip_decomposition=True,
                    )
                )
            merged_separable = _merge_separable_results(
                objective,
                constraints,
                original_vars,
                decomposition,
                component_results,
                precision_info,
            )
            if merged_separable is not None:
                return merged_separable

    if not parameter_symbols and not _skip_semialg_pruning(
        objective, constraints, original_vars
    ):
        constraints = prune_redundant(constraints, original_vars, pre_diags)
    has_piecewise = objective.has(sp.Piecewise) or any(
        getattr(relation, "lhs", sp.Integer(0)).has(sp.Piecewise)
        or getattr(relation, "rhs", sp.Integer(0)).has(sp.Piecewise)
        for relation in constraints
    )

    if has_piecewise:
        cases = algebraize_cases(
            objective,
            constraints,
            allow_transcendentals=allow_transcendentals,
            allow_rootof=allow_rootof,
            variables=original_vars,
        )
        results = []
        for case in cases:
            case_diags = list(pre_diags)
            feasible = (
                None
                if case.parameter_symbols
                else exact_feasible(
                    case.all_constraints(),
                    case.all_variables,
                    case_diags,
                )
            )
            if feasible is False:
                continue
            results.append(
                _solve_single_problem(
                    case,
                    box_bounds,
                    interval_max_boxes,
                    interval_tol,
                    max_active_sets,
                    use_sos,
                    seed_diagnostics=case_diags,
                    semialg_certification=semialg_certification,
                    semialg_range_cost_limit=semialg_range_cost_limit,
                    working_precision=effective_precision,
                    precision_provenance=precision_info.as_dict(),
                )
            )
        if not results:
            results.append(
                _solve_single_problem(
                    cases[0],
                    box_bounds,
                    interval_max_boxes,
                    interval_tol,
                    max_active_sets,
                    use_sos,
                    seed_diagnostics=pre_diags,
                    semialg_certification=semialg_certification,
                    semialg_range_cost_limit=semialg_range_cost_limit,
                    working_precision=effective_precision,
                    precision_provenance=precision_info.as_dict(),
                )
            )
        return _merge_piecewise_results(results)

    problem = algebraize(
        objective,
        constraints,
        allow_transcendentals=allow_transcendentals,
        allow_rootof=allow_rootof,
        variables=original_vars,
    )
    return _solve_single_problem(
        problem,
        box_bounds,
        interval_max_boxes,
        interval_tol,
        max_active_sets,
        use_sos,
        seed_diagnostics=pre_diags,
        semialg_certification=semialg_certification,
        semialg_range_cost_limit=semialg_range_cost_limit,
        working_precision=effective_precision,
        precision_provenance=precision_info.as_dict(),
    )


def _mapped_candidate(candidate: CandidateSolution) -> CandidateSolution:
    """Return a maximization-view candidate without mutating the minimum result."""
    label = candidate.local_classification
    if label:
        label = label.replace("minimum", "__MAXIMUM__").replace("maximum", "minimum")
        label = label.replace("__MAXIMUM__", "maximum")
    numeric = candidate.objective_value_numeric
    return replace(
        candidate,
        primal_values=dict(candidate.primal_values),
        multiplier_values=dict(candidate.multiplier_values),
        original_variable_values=dict(candidate.original_variable_values),
        objective_value=sp.simplify(-candidate.objective_value),
        objective_value_numeric=None if numeric is None else -numeric,
        source=f"maximize:{candidate.source}",
        local_classification=label,
    )


def _map_min_result_to_max(
    result: PipelineResult, objective: sp.Expr
) -> PipelineResult:
    """Map a minimum of ``-objective`` to a non-mutating maximization result."""
    mapped_candidates = [_mapped_candidate(item) for item in result.candidates]
    best = None
    if result.best_candidate is not None:
        try:
            best = mapped_candidates[result.candidates.index(result.best_candidate)]
        except ValueError:
            best = None

    interval = result.interval_result
    if interval is not None:
        interval = replace(
            interval,
            best_lower_bound=-interval.best_upper_bound,
            best_upper_bound=-interval.best_lower_bound,
            lower_bound_rigorous=interval.upper_bound_rigorous,
            upper_bound_rigorous=interval.lower_bound_rigorous,
            lower_bound_method=f"maximize_via_negation:{interval.lower_bound_method}",
        )

    structural = result.structural_result
    if structural is not None and is_dataclass(structural):
        structural = replace(structural, value=sp.simplify(-structural.value))

    semialg_result = result.semialg_result
    if semialg_result is not None and is_dataclass(semialg_result):
        if getattr(semialg_result, "value", None) is not None:
            try:
                semialg_result = replace(
                    semialg_result, value=sp.simplify(-semialg_result.value)
                )
            except TypeError:
                pass

    semialg_range = result.semialg_range
    if semialg_range is not None and is_dataclass(semialg_range):
        try:
            semialg_range = replace(
                semialg_range,
                formula=sp.simplify(
                    semialg_range.formula.subs(
                        semialg_range.value_symbol, -semialg_range.value_symbol
                    )
                ),
                infimum=(
                    None
                    if semialg_range.supremum is None
                    else sp.simplify(-semialg_range.supremum)
                ),
                supremum=(
                    None
                    if semialg_range.infimum is None
                    else sp.simplify(-semialg_range.infimum)
                ),
                infimum_attained=semialg_range.supremum_attained,
                supremum_attained=semialg_range.infimum_attained,
            )
        except (TypeError, AttributeError):
            pass

    certificate = result.certificate
    if certificate is not None:
        old_lower = certificate.lower_bound
        old_upper = certificate.upper_bound
        proof_objects = list(certificate.proof_objects)
        proof_objects.append(
            ProofArtifact(
                kind="maximize_via_minimize",
                summary="Maximization reduced exactly to minimization of the negated objective.",
                payload={"objective": sp.sstr(objective)},
            )
        )
        certificate = replace(
            certificate,
            primal_solution=dict(certificate.primal_solution),
            primal_value=(
                None
                if certificate.primal_value is None
                else sp.simplify(-certificate.primal_value)
            ),
            lower_bound=None if old_upper is None else sp.simplify(-old_upper),
            upper_bound=None if old_lower is None else sp.simplify(-old_lower),
            proof_objects=proof_objects,
        )

    mapped_problem = replace(
        result.lifted_problem,
        original_objective=objective,
        lifted_objective=sp.simplify(-result.lifted_problem.lifted_objective),
        original_constraints=list(result.lifted_problem.original_constraints),
        lifted_constraints=list(result.lifted_problem.lifted_constraints),
        defining_equalities=list(result.lifted_problem.defining_equalities),
        branch_inequalities=list(result.lifted_problem.branch_inequalities),
        strict_inequalities=list(result.lifted_problem.strict_inequalities),
        graph_equalities=list(result.lifted_problem.graph_equalities),
        graph_inequalities=list(result.lifted_problem.graph_inequalities),
        auxiliary_variables=list(result.lifted_problem.auxiliary_variables),
        substitution_map=dict(result.lifted_problem.substitution_map),
        all_variables=list(result.lifted_problem.all_variables),
        lift_records=list(result.lifted_problem.lift_records),
        preprocessing_notes=list(result.lifted_problem.preprocessing_notes),
        strict_closure_warnings=list(result.lifted_problem.strict_closure_warnings),
        original_variables=list(result.lifted_problem.original_variables),
        parameter_symbols=list(result.lifted_problem.parameter_symbols),
        structural_metadata=dict(result.lifted_problem.structural_metadata),
    )

    return replace(
        result,
        lifted_problem=mapped_problem,
        kkt_system=result.kkt_system,
        candidates=mapped_candidates,
        best_candidate=best,
        interval_result=interval,
        sos_lower_bound=None,
        sos_bound_numeric=None,
        certificate=certificate,
        diagnostics=list(result.diagnostics),
        critical_point_results=list(result.critical_point_results),
        inferred_box_bounds=dict(result.inferred_box_bounds),
        precision_provenance=dict(result.precision_provenance),
        semialg_result=semialg_result,
        semialg_range=semialg_range,
        structural_result=structural,
        optimum_value=(
            None if result.optimum_value is None else sp.simplify(-result.optimum_value)
        ),
        status=result.status.replace("minimum", "maximum").replace(
            "infimum", "supremum"
        ),
        optimization_sense="maximize",
        kkt_objective=result.kkt_objective,
    )


def maximize(
    objective: sp.Expr,
    constraints: Sequence[Relational] | None = None,
    **kwargs,
) -> PipelineResult:
    """Maximize an objective using the same certified pipeline as :func:`minimize`."""
    objective = sp.sympify(objective)
    return _map_min_result_to_max(
        minimize(-objective, constraints, **kwargs), objective
    )


def minval(
    objective: sp.Expr,
    constraints: Sequence[Relational] | None = None,
    **kwargs,
) -> Optional[sp.Expr]:
    """Return the certified minimum value or infimum when available."""
    result = minimize(objective, constraints, **kwargs)
    if result.optimum_value is not None:
        return sp.simplify(result.optimum_value)
    if result.best_candidate is not None and result.certified:
        return sp.simplify(result.best_candidate.objective_value)
    return None


def maxval(
    objective: sp.Expr,
    constraints: Sequence[Relational] | None = None,
    **kwargs,
) -> Optional[sp.Expr]:
    """Return the certified maximum value or supremum when available."""
    result = maximize(objective, constraints, **kwargs)
    if result.optimum_value is not None:
        return sp.simplify(result.optimum_value)
    if result.best_candidate is not None and result.certified:
        return sp.simplify(result.best_candidate.objective_value)
    return None


def _optimizer_points(result: PipelineResult) -> tuple[dict[sp.Symbol, sp.Expr], ...]:
    if not result.certified or result.attained is False:
        return ()
    points = []
    if result.structural_result is not None:
        for point in getattr(result.structural_result, "points", ()) or ():
            points.append(dict(point))
    optimum = result.optimum_value
    for candidate in result.candidates:
        point = candidate.original_variable_values or candidate.primal_values
        if not point:
            continue
        if (
            optimum is not None
            and sp.simplify(candidate.objective_value - optimum) != 0
        ):
            continue
        points.append(dict(point))
    unique = []
    seen = set()
    for point in points:
        key = tuple(
            sorted((sp.sstr(k), sp.sstr(sp.simplify(v))) for k, v in point.items())
        )
        if key not in seen:
            seen.add(key)
            unique.append(point)
    return tuple(unique)


def argmin(
    objective: sp.Expr,
    constraints: Sequence[Relational] | None = None,
    **kwargs,
) -> tuple[dict[sp.Symbol, sp.Expr], ...]:
    """Return represented minimizers; return ``()`` when an infimum is unattained."""
    return _optimizer_points(minimize(objective, constraints, **kwargs))


def argmax(
    objective: sp.Expr,
    constraints: Sequence[Relational] | None = None,
    **kwargs,
) -> tuple[dict[sp.Symbol, sp.Expr], ...]:
    """Return represented maximizers; return ``()`` when a supremum is unattained."""
    return _optimizer_points(maximize(objective, constraints, **kwargs))
