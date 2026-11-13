from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass, replace
from itertools import product

import sympy as sp
from sympy.core.relational import Relational

from .algebraic_numbers import algebraic_compare, exactify, exactify_dict
from .algebraize import Algebraizer, algebraize, algebraize_cases
from .candidates import (
    build_active_set_plan,
    dedupe_candidates,
    select_best_candidate,
    solve_exact_candidates,
    solve_kkt_candidates,
)
from .certification import build_certificate
from .convex_routing import (
    complete_box_bounds,
    infer_box_bounds,
    try_direct_solver,
)
from .decomposition import find_components
from .diagnostics import DiagnosticKind, SolverDiagnostic
from .fast_rules import try_fast_rules
from .hierarchy import search_sos_hierarchy
from .input_normalization import (
    assumption_constraints,
    normalize_bounds_input,
    normalize_constraint_branches,
    normalize_constraints_input,
    normalize_variables,
    require_continuous_variables,
)
from .intervals import interval_optimize
from .kkt import build_kkt_system
from .numeric_utils import real_part
from .objective_transforms import find_monotone_reduction
from .planner import plan_problem
from .polynomial_structure import build_sparsity_graph, reduce_sign_symmetry
from .precision import decimal_digits_to_bits, precision_provenance
from .problem_models import (
    CandidateSolution,
    IntervalBox,
    LiftedProblem,
    PipelineResult,
    ProofArtifact,
)
from .rational_optimization import (
    preserve_rational_domain,
    try_rational_range,
    try_univariate_rational,
)
from .relation_checks import normalize_relational
from .semialgebraic import (
    certify_global_min,
    exact_feasible,
    function_range_exact,
    get_semialg_capabilities,
    minimize_exact,
    prove_nonnegative_exact,
    prune_redundant,
)
from .solver_options import UNSET, SolverOptions, normalize_solver_options
from .sos import sos_lower_bound as compute_sos_bound
from .structural_results import StructuralGlobalResult
from .structural_routing import try_structural_solver
from .sublevel_bounds import restrict_polynomial_sublevel
from .symbolic_cache import simplify_scalar
from .univariate_analysis import try_univariate_range


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


def _exact_minimum(values):
    if not values:
        return None
    best = values[0]
    for value in values[1:]:
        comparison = algebraic_compare(value, best)
        if comparison is None:
            return None
        if comparison < 0:
            best = value
    return best


def _merge_piecewise_results(results):
    """Combine branch bounds without treating a branch proof as a global proof."""
    if not results:
        raise ValueError("At least one Piecewise branch result is required.")

    results = [
        _exact_value_result(r, r.lifted_problem.original_objective) for r in results
    ]
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
                key=lambda result: (
                    float("inf")
                    if result.best_candidate.objective_value_numeric is None
                    else result.best_candidate.objective_value_numeric
                ),
            )

    exact_value = _exact_minimum([value for value, _ in exact_pairs])
    if exact_value is not None:
        exact_result = next(
            result for value, result in exact_pairs if value == exact_value
        )
        comparison = (
            None
            if best_candidate is None
            else algebraic_compare(exact_value, best_candidate.objective_value)
        )
        if (
            len(exact_pairs) == len(results)
            or best_candidate is None
            or comparison == -1
        ):
            selected = exact_result

    if selected is None:
        selected = results[0]

    combined = replace(
        selected,
        candidates=list(candidates),
        best_candidate=best_candidate,
        diagnostics=list(selected.diagnostics),
        critical_point_results=list(selected.critical_point_results),
        inferred_box_bounds=dict(selected.inferred_box_bounds),
        precision_provenance=dict(selected.precision_provenance),
    )

    lower_bounds = [
        result.certificate.lower_bound
        for result in results
        if result.certificate is not None and result.certificate.lower_bound is not None
    ]
    all_have_lower = len(lower_bounds) == len(results)
    global_lower = _exact_minimum(lower_bounds) if all_have_lower else None
    upper_bounds = [
        result.certificate.upper_bound
        for result in results
        if result.certificate is not None and result.certificate.upper_bound is not None
    ]
    global_upper = _exact_minimum(upper_bounds)

    # One certified escaping component establishes unboundedness of the union.
    # Unresolved components cannot invalidate that descent witness.
    global_infimum = (
        -sp.oo
        if any(value is -sp.oo for value in exact_infima)
        else _exact_minimum(exact_infima)
        if all(value is not None for value in exact_infima)
        else None
    )
    if (
        global_infimum is None
        and global_lower is not None
        and global_upper is not None
        and algebraic_compare(global_lower, global_upper) == 0
    ):
        global_infimum = global_lower
    global_value_proved = global_infimum is not None
    attained = None
    if global_value_proved:
        minimizers = [
            result
            for value, result in zip(exact_infima, results)
            if value is not None and algebraic_compare(value, global_infimum) == 0
        ]
        if any(getattr(result, "attained", None) is True for result in minimizers):
            attained = True
        elif all(getattr(result, "attained", None) is False for result in minimizers):
            attained = False

    certificate = selected.certificate
    if certificate is None:
        certificate = build_certificate(best_candidate=best_candidate)
    else:
        certificate = replace(
            certificate,
            primal_solution=dict(certificate.primal_solution),
            proof_objects=list(certificate.proof_objects),
        )
    primal_certificate = (
        next(
            (
                result.certificate
                for result in results
                if result.certificate is not None
                and result.certificate.primal_value == global_upper
            ),
            None,
        )
        if global_upper is not None
        else None
    )
    certificate.primal_solution = (
        dict(primal_certificate.primal_solution)
        if primal_certificate is not None
        else {}
    )
    certificate.primal_value = (
        primal_certificate.primal_value if primal_certificate is not None else None
    )
    if global_value_proved:
        combined.candidates = dedupe_candidates(
            [
                candidate
                for result in results
                for candidate in result.candidates
                if _exact_source_candidate(
                    result.lifted_problem, candidate, result.structural_result
                )
            ]
        )
        combined.best_candidate = select_best_candidate(combined.candidates)
    certificate.lower_bound = global_infimum if global_value_proved else global_lower
    certificate.upper_bound = global_upper

    combined.certified = bool(global_value_proved)
    if global_value_proved:
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
                "pruned_component_count": sum(
                    r.status == "component_pruned" for r in results
                ),
                "branch_bound_proofs": tuple(r.certificate for r in results),
                "exact_branch_infima": [
                    sp.sstr(value) if value is not None else None
                    for value in exact_infima
                ],
                "all_branch_infima_exact": all(v is not None for v in exact_infima),
            },
        )
    )
    combined.certificate = certificate
    combined.optimum_value = (
        sp.simplify(global_infimum) if global_value_proved else None
    )
    combined.attained = attained if global_value_proved else None
    combined.method_label = "branch"
    combined.optimizer_domain = sp.Or(
        *(sp.And(*r.lifted_problem.original_constraints) for r in results)
    )
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
    allow_range_fallback=True,
    solver_plan=None,
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
    if (
        structural_result is None
        and len(original_vars) == 1
        and (direct is None or direct.metadata.get("rigorous_global") is not True)
    ):
        exact = try_univariate_rational(
            problem.original_objective,
            problem.original_constraints,
            original_vars,
        )
        if exact is not None:
            structural_result, direct = exact, None
    if (
        allow_range_fallback
        and structural_result is None
        and direct is None
        and not problem.parameter_symbols
    ):
        structural_result = try_univariate_range(
            problem.original_objective,
            problem.original_constraints,
            original_vars,
        )

    if (
        structural_result is None
        and direct is None
        and not problem.parameter_symbols
        and semialg_certification != "candidate"
    ):
        limit = semialg_range_cost_limit
        if limit is None and semialg_certification == "auto":
            limit = 12
        structural_result = try_rational_range(
            problem.original_objective,
            problem.original_constraints,
            original_vars,
            cost_limit=limit,
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
        if (
            solver_plan is not None
            and not solver_plan.try_semialg_early
            and semialg_certification == "auto"
        ):
            deferred = True
            semialg_attempted = True
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
    solver_plan=None,
):
    """Build KKT data lazily and enumerate exact/numerical candidates."""
    try:
        kkt = build_kkt_system(problem)
        equalities, inequalities = problem.kkt_residuals()
    except ValueError as exc:
        if "disequality" not in str(exc).lower():
            raise
        diagnostics.append(
            SolverDiagnostic.from_exception(
                "kkt", "constraint_support", DiagnosticKind.UNSUPPORTED, exc
            )
        )
        return None, [], candidates, select_best_candidate(candidates)
    critical_results = []
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
    allow_exact = solver_plan is None or solver_plan.try_exact_kkt
    allow_numeric = solver_plan is None or solver_plan.try_numeric_kkt
    if allow_exact:
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
    else:
        exact_meta["exact_search_complete"] = False

    if allow_numeric and not exact_meta.get("exact_search_complete", False):
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
                and certificate_obj.verified
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


def _box_covers_global_search(problem, box):
    """Check that interval search covers the feasible set or an improving sublevel set.

    A user-supplied search box is proposal guidance until its coverage is proved.
    Domain constraints and checked exterior polynomial bounds can justify that
    coverage; bounds local to a smaller box cannot become global lower bounds.
    """
    enclosing = complete_box_bounds(
        problem, infer_box_bounds(problem.all_variables, problem.all_constraints())
    )
    payload = problem.structural_metadata.get("sublevel_bounds")
    if payload is not None:
        from dataclasses import fields

        from .sublevel_bounds import _SublevelCertificate

        proof = _SublevelCertificate(
            **{
                field.name: payload[field.name]
                for field in fields(_SublevelCertificate)
            }
        )
        if proof.verify() and proof.objective == problem.original_objective:
            enclosing.update(proof.bounds)
    return all(
        variable in enclosing
        and variable in box
        and sp.Rational(box[variable][0]) <= enclosing[variable][0]
        and sp.Rational(box[variable][1]) >= enclosing[variable][1]
        for variable in problem.all_variables
    )


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
        result = interval_optimize(
            problem.lifted_objective,
            problem.all_constraints(),
            merged,
            max_boxes=interval_max_boxes,
            tol=interval_tol,
            precision=decimal_digits_to_bits(int(working_precision or 50)),
        )
        if result.lower_bound_rigorous and not _box_covers_global_search(
            problem, merged
        ):
            diagnostics.append(
                SolverDiagnostic(
                    "interval",
                    "search_box_coverage",
                    DiagnosticKind.UNSUPPORTED,
                    "The interval lower bound is local to the search box; global source coverage is unproved.",
                )
            )
            result = replace(result, lower_bound_rigorous=False)
        return result
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


def _exact_source_candidate(problem, candidate, structural=None):
    """Check the original value and feasibility before using a proposal as a proof.

    Candidate search permits tolerance-based feasibility and rational recovery.
    Those screens cannot establish attainment or a rigorous primal upper bound.
    """
    if candidate is None:
        return False
    point = candidate.original_variable_values or candidate.primal_values
    # Structural factories already bind their witnesses to source feasibility
    # and value. Reuse that proof instead of re-expanding trig/algebraic roots.
    if (
        structural is not None
        and structural.certified
        and candidate.objective_value == structural.value
        and any(
            all(point.get(k) == v for k, v in witness.items())
            for witness in structural.points
        )
    ):
        return True
    used = problem.original_objective.free_symbols | set().union(
        *(relation.free_symbols for relation in problem.original_constraints)
    )
    if any(
        variable in used and variable not in point
        for variable in problem.original_variables
    ):
        return False
    try:
        if any(
            sp.sympify(value).has(sp.Float)
            or sp.sympify(value).is_real is not True
            or sp.sympify(value).is_finite is not True
            for value in point.values()
        ):
            return False
        if any(
            sp.simplify(relation.subs(point, simultaneous=True)) is not sp.true
            for relation in problem.original_constraints
        ):
            return False
        value = sp.simplify(problem.original_objective.subs(point, simultaneous=True))
        return (
            value.is_real is True
            and value.is_finite is True
            and sp.simplify(value - candidate.objective_value) == 0
        )
    except (TypeError, ValueError, ArithmeticError, NotImplementedError):
        return False


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
    best = best if _exact_source_candidate(problem, best) else None
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


def _method_label_for_result(
    *,
    direct_kind=None,
    structural_result=None,
    semialg_result=None,
    sos_result=None,
    interval_result=None,
    kkt=None,
):
    """Choose the stable route label once, when the result is assembled."""
    if direct_kind in {"lp", "qp"}:
        return direct_kind
    if structural_result is not None:
        kind = str(getattr(structural_result, "kind", "structural"))
        if "univariate" in kind:
            return "univariate"
        if "piecewise" in kind:
            return "branch"
        return kind
    if semialg_result is not None:
        return "semialg"
    if sos_result is not None:
        return "sos"
    if interval_result is not None:
        return "interval"
    if kkt is not None:
        return "kkt"
    return "unknown"


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
    allow_range_fallback=True,
):
    diagnostics = list(seed_diagnostics)
    problem, merged, original_vars, symmetry_report = _prepare_single_problem(
        problem, box_bounds
    )
    solver_plan = plan_problem(
        problem.lifted_objective,
        problem.all_constraints(),
        problem.all_variables,
        merged,
    )
    structural_result, direct, semialg_result, semialg_attempted = _run_early_routes(
        problem,
        original_vars,
        diagnostics,
        semialg_certification,
        semialg_range_cost_limit,
        working_precision,
        allow_range_fallback=allow_range_fallback,
        solver_plan=solver_plan,
    )
    return _execute_single_problem_stages(
        problem=problem,
        merged=merged,
        original_vars=original_vars,
        symmetry_report=symmetry_report,
        solver_plan=solver_plan,
        structural_result=structural_result,
        direct=direct,
        semialg_result=semialg_result,
        semialg_attempted=semialg_attempted,
        diagnostics=diagnostics,
        interval_max_boxes=interval_max_boxes,
        interval_tol=interval_tol,
        max_active_sets=max_active_sets,
        use_sos=use_sos,
        working_precision=working_precision,
        precision_provenance=precision_provenance,
    )


def _execute_single_problem_stages(
    *,
    problem,
    merged,
    original_vars,
    symmetry_report,
    solver_plan,
    structural_result,
    direct,
    semialg_result,
    semialg_attempted,
    diagnostics,
    interval_max_boxes,
    interval_tol,
    max_active_sets,
    use_sos,
    working_precision,
    precision_provenance,
):
    """Run candidate, bound, certification, and result-assembly stages."""
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
                objective_value=structural_result.value,
                objective_value_numeric=(
                    None
                    if structural_result.value.free_symbols
                    else real_part(structural_result.value)
                ),
                feasibility_checked=bool(
                    structural_result.certified
                    or structural_result.metadata.get("primal_exact")
                ),
                source=structural_result.kind,
                local_classification="GlobalMinimum"
                if structural_result.certified
                else "FeasiblePoint",
                original_variable_values=dict(point),
            )
            candidates.append(candidate)
        candidates = dedupe_candidates(candidates)
        best = select_best_candidate(candidates)
        certified = bool(structural_result.certified)
        gap = 0.0 if certified and structural_result.attained else None
        if not certified:
            status = f"candidate_by_{structural_result.kind}"
        elif structural_result.attained is True:
            status = f"global_certified_by_{structural_result.kind}"
        elif structural_result.attained is False:
            status = f"global_infimum_by_{structural_result.kind}"
        else:
            status = f"global_bound_by_{structural_result.kind}_attainment_unknown"
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
            problem,
            merged,
            max_active_sets,
            diagnostics,
            candidates,
            solver_plan=solver_plan,
        )

        if best is not None:
            from .convex_rules import convex_kkt_certificate
            from .rule_utils import RULE_ERRORS, RuleContext

            try:
                convex = convex_kkt_certificate(
                    RuleContext(
                        problem.original_objective,
                        tuple(problem.original_constraints),
                        tuple(original_vars),
                    ),
                    best.original_variable_values or best.primal_values,
                )
            except RULE_ERRORS:
                convex = None
            if convex is not None:
                return _execute_single_problem_stages(
                    problem=problem,
                    merged=merged,
                    original_vars=original_vars,
                    symmetry_report=symmetry_report,
                    solver_plan=solver_plan,
                    structural_result=convex,
                    direct=None,
                    semialg_result=None,
                    semialg_attempted=semialg_attempted,
                    diagnostics=diagnostics,
                    interval_max_boxes=interval_max_boxes,
                    interval_tol=interval_tol,
                    max_active_sets=max_active_sets,
                    use_sos=use_sos,
                    working_precision=working_precision,
                    precision_provenance=precision_provenance,
                )

        merged, sublevel_proof = restrict_polynomial_sublevel(problem, merged, best)
        if sublevel_proof is not None:
            problem = replace(
                problem,
                structural_metadata={
                    **problem.structural_metadata,
                    "sublevel_bounds": sublevel_proof,
                },
            )
            solver_plan = plan_problem(
                problem.lifted_objective,
                problem.all_constraints(),
                problem.all_variables,
                merged,
            )

        precision_allows_float64 = bool(
            (precision_provenance or {}).get("float64_backends_allowed", True)
        )
        if (
            use_sos
            and solver_plan.try_sos
            and problem.is_polynomial_problem()
            and precision_allows_float64
        ):
            sos_result, sos_bound, sos_bound_numeric, sparsity_components = (
                _run_sos_stage(problem, best, diagnostics)
            )

        interval_result = _run_interval_stage(
            problem,
            merged if solver_plan.try_interval else {},
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
            if structural_result.attained is True:
                status = f"global_certified_by_{structural_result.kind}"
            elif structural_result.attained is False:
                status = (
                    f"global_infimum_by_{structural_result.kind}_strict_constraints"
                )
            else:
                status = f"global_bound_by_{structural_result.kind}_attainment_unknown"
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

    return _assemble_pipeline_result(
        problem=problem,
        merged=merged,
        solver_plan=solver_plan,
        structural_result=structural_result,
        direct=direct,
        direct_kind=direct_kind,
        semialg_result=semialg_result,
        semialg_proof=semialg_proof,
        semialg_sign=semialg_sign,
        semialg_range=semialg_range,
        kkt=kkt,
        critical_results=critical_results,
        candidates=candidates,
        best=best,
        interval_result=interval_result,
        sos_result=sos_result,
        sos_bound=sos_bound,
        sos_bound_numeric=sos_bound_numeric,
        sparsity_components=sparsity_components,
        certified=certified,
        gap=gap,
        status=status,
        diagnostics=diagnostics,
        symmetry_report=symmetry_report,
        working_precision=working_precision,
        precision_provenance=precision_provenance,
    )


def _assemble_pipeline_result(
    *,
    problem,
    merged,
    solver_plan,
    structural_result,
    direct,
    direct_kind,
    semialg_result,
    semialg_proof,
    semialg_sign,
    semialg_range,
    kkt,
    critical_results,
    candidates,
    best,
    interval_result,
    sos_result,
    sos_bound,
    sos_bound_numeric,
    sparsity_components,
    certified,
    gap,
    status,
    diagnostics,
    symmetry_report,
    working_precision,
    precision_provenance,
):
    """Build proof artifacts and the immutable-facing pipeline result."""
    source_best = (
        best if _exact_source_candidate(problem, best, structural_result) else None
    )
    sos_rigorous = bool(
        sos_bound is not None
        and sos_result is not None
        and sos_result.certificate is not None
        and sos_result.certificate.verified
    )
    certificate = build_certificate(
        best_candidate=source_best,
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
                summary=f"Structural route: {structural_result.kind}",
                payload=dict(structural_result.metadata),
            )
        )
        if structural_result.certified:
            certificate.lower_bound = structural_result.value
            certificate.upper_bound = (
                structural_result.value
                if structural_result.attained
                else certificate.upper_bound
            )
            certificate.status = (
                "proven_optimal"
                if structural_result.attained is True
                else (
                    "proven_infimum"
                    if structural_result.attained is False
                    else "proven_bound_attainment_unknown"
                )
            )

    if "sublevel_bounds" in problem.structural_metadata:
        certificate.proof_objects.append(
            ProofArtifact(
                kind="polynomial_sublevel_bounds",
                summary="Exact feasible upper bound and strict exterior polynomial lower bound",
                payload=problem.structural_metadata["sublevel_bounds"],
            )
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

    if solver_plan is not None:
        certificate.proof_objects.append(
            ProofArtifact(
                kind="planner",
                summary=solver_plan.explain(),
                payload={
                    "complexity": dict(solver_plan.complexity.__dict__),
                    "routes": {
                        "semialg": solver_plan.try_semialg_early,
                        "exact_kkt": solver_plan.try_exact_kkt,
                        "numeric_kkt": solver_plan.try_numeric_kkt,
                        "sos": solver_plan.try_sos,
                        "interval": solver_plan.try_interval,
                    },
                    "reasons": dict(solver_plan.reasons),
                },
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
            structural_result.value
            if structural_result is not None and structural_result.certified
            else (
                sp.simplify(semialg_result.value)
                if semialg_result is not None and semialg_result.certified is True
                else (
                    sp.simplify(semialg_range.infimum)
                    if certified and semialg_range is not None
                    else (
                        sp.simplify(best.objective_value)
                        if certified and source_best is not None
                        else None
                    )
                )
            )
        ),
        attained=(
            structural_result.attained
            if structural_result is not None and structural_result.certified
            else (
                semialg_result.attained
                if semialg_result is not None and semialg_result.certified is True
                else (
                    semialg_range.infimum_attained
                    if certified and semialg_range is not None
                    else (True if certified and source_best is not None else None)
                )
            )
        ),
        working_precision=working_precision,
        precision_provenance=dict(precision_provenance or {}),
        structural_result=structural_result,
        optimization_sense="minimize",
        kkt_objective=(problem.lifted_objective if kkt is not None else None),
        solver_plan=solver_plan,
        method_label=_method_label_for_result(
            direct_kind=direct_kind,
            structural_result=structural_result,
            semialg_result=semialg_result,
            sos_result=sos_result,
            interval_result=interval_result,
            kkt=kkt,
        ),
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
        method_label="separable_decomposition",
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
        value = simplify_scalar(objective.subs(point))
        mapped_candidates.append(
            replace(
                candidate,
                primal_values=dict(candidate.primal_values),
                multiplier_values=dict(candidate.multiplier_values),
                original_variable_values=dict(candidate.original_variable_values),
                objective_value=value,
                objective_value_numeric=None
                if value.free_symbols
                else real_part(value, effective_precision),
                source=f"monotone_{reduction.kind}:{candidate.source}",
            )
        )
    mapped_best = select_best_candidate(mapped_candidates)
    optimum = subresult.optimum_value
    if optimum is not None:
        optimum = simplify_scalar(reduction.recover_value(optimum))

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


def minimize_result(
    objective: sp.Expr,
    constraints: object = None,
    box_bounds: dict[sp.Symbol, tuple[float, float]] | None = None,
    variables: sp.Symbol | Sequence[sp.Symbol] | None = None,
    allow_transcendentals: bool = True,
    allow_rootof: bool = True,
    interval_max_boxes=UNSET,
    interval_tol=UNSET,
    max_active_sets=UNSET,
    use_sos=UNSET,
    semialg_certification=UNSET,
    semialg_range_cost_limit=UNSET,
    working_precision: int | None = None,
    bounds: object = None,
    options: Mapping[str, Mapping[str, object]] | None = None,
    use_symbol_assumptions: bool = False,
    region: object = None,
    _skip_monotone_reduction: bool = False,
    _skip_decomposition: bool = False,
    _skip_disjunction: bool = False,
) -> PipelineResult:
    """Minimize a continuous SymPy objective with automatic structure-aware routing.

    The dispatcher recognizes linear and convex-quadratic problems, symbolic
    critical/KKT systems, polynomial SOS/moment opportunities, algebraized branch
    problems, and verified interval problems. The returned ``PipelineResult``
    separates the best candidate from rigorous certification evidence. Integer
    decision variables are rejected, including assumptions implying integrality;
    integer parameters remain valid when the decision variables are explicit.

    Grouped ``options={...}`` values are normalized once. An explicitly supplied
    flat keyword always wins, even when its value equals the documented default.
    ``region`` adds exact geometric membership to ``constraints`` and ``bounds``.
    Coordinate-free regions require explicit variables; symbolic regions retain
    their owned coordinate order.
    """
    solver_options = normalize_solver_options(
        interval_max_boxes=interval_max_boxes,
        interval_tol=interval_tol,
        max_active_sets=max_active_sets,
        use_sos=use_sos,
        semialg_certification=semialg_certification,
        semialg_range_cost_limit=semialg_range_cost_limit,
        options=options,
    )
    interval_max_boxes = solver_options.interval_max_boxes
    interval_tol = solver_options.interval_tol
    max_active_sets = solver_options.max_active_sets
    use_sos = solver_options.use_sos
    semialg_certification = solver_options.semialg_certification
    semialg_range_cost_limit = solver_options.semialg_range_cost_limit

    prepared = None
    if region is not None:
        from .region_inputs import prepare_region
        from .region_optimization import constraint_formula, solve_cover, vertex_minimum

        prepared = prepare_region(region, variables)
        variables = list(prepared.variables)
        if solver_options.quadratic.backend != "scip":
            require_continuous_variables(variables)

        def solve_chart(reduced, domain, coordinates):
            return _minimize_result_impl(
                reduced,
                domain,
                box_bounds=None,
                variables=list(coordinates),
                allow_transcendentals=allow_transcendentals,
                allow_rootof=allow_rootof,
                working_precision=working_precision,
                bounds=None,
                use_symbol_assumptions=False,
                _skip_monotone_reduction=False,
                _skip_decomposition=False,
                _skip_disjunction=False,
                solver_options=solver_options,
            )

        if solver_options.structural_rules:
            extra = assumption_constraints(variables) if use_symbol_assumptions else []
            vertex = (
                vertex_minimum(prepared, objective)
                if constraints is None
                and bounds is None
                and not extra
                and prepared.is_geometry
                else None
            )
            if vertex is not None:
                source = prepared.source_formula
                parameters = sorted(
                    (sp.sympify(objective).free_symbols | source.free_symbols)
                    - set(variables),
                    key=sp.default_sort_key,
                )
                vertex_result = _structural_pipeline_result(
                    sp.sympify(objective),
                    [source],
                    variables,
                    parameters,
                    vertex,
                    box_bounds,
                    solver_options,
                    precision_provenance(objective, [source], working_precision),
                )
                vertex_result.region_preparation = prepared
                return vertex_result
            covered = solve_cover(
                prepared,
                objective,
                constraints,
                bounds,
                extra,
                solve_chart,
                _merge_piecewise_results,
            )
            if covered is not None:
                return covered
        constraints = [constraints, prepared.membership]
        if box_bounds is None:
            numeric_bounds = {
                v: pair
                for v, pair in prepared.outer_bounds.items()
                if all(not e.free_symbols and e.is_finite is True for e in pair)
            }
            if len(numeric_bounds) == len(variables):
                box_bounds = numeric_bounds

    result = _minimize_result_impl(
        objective,
        constraints,
        box_bounds=box_bounds,
        variables=variables,
        allow_transcendentals=allow_transcendentals,
        allow_rootof=allow_rootof,
        working_precision=working_precision,
        bounds=bounds,
        use_symbol_assumptions=use_symbol_assumptions,
        _skip_monotone_reduction=_skip_monotone_reduction,
        _skip_decomposition=_skip_decomposition,
        _skip_disjunction=_skip_disjunction,
        solver_options=solver_options,
    )
    if prepared is not None:
        extra, _ = normalize_bounds_input(bounds, variables)
        result.optimizer_domain = sp.And(
            constraint_formula(constraints),
            *extra,
            *(assumption_constraints(variables) if use_symbol_assumptions else []),
        )
        result.region_preparation = prepared
    return _exact_value_result(result, sp.sympify(objective))


def _exact_value_result(result, objective):
    """Separate rounded objective evaluations from exact global values.

    A floating literal denotes a represented input datum. Arithmetic on it can
    round, so a derived Float needs independent exact bounds before it can be
    exposed as an exact infimum. Verified interval enclosures remain available.
    """
    value = result.optimum_value
    if not result.certified or value is None or not value.has(sp.Float):
        return result
    certificate = result.certificate
    exact = None
    if (
        objective.is_Float
        and value.is_Float
        and sp.Rational(objective) == sp.Rational(value)
    ):
        exact = sp.Rational(objective)
    elif certificate is not None:
        lower, upper = certificate.lower_bound, certificate.upper_bound
        if lower is not None and upper is not None:
            lower, upper = sp.sympify(lower), sp.sympify(upper)
            if not lower.has(sp.Float) and not upper.has(sp.Float) and lower == upper:
                exact = lower
    if exact is not None:
        if certificate is not None:
            certificate = replace(certificate, lower_bound=exact, upper_bound=exact)
        return replace(result, optimum_value=exact, certificate=certificate)
    if certificate is not None:
        certificate = replace(
            certificate,
            status="candidate_only",
            lower_bound=None,
            upper_bound=None,
            gap=None,
            proof_objects=[
                *certificate.proof_objects,
                ProofArtifact(
                    "inexact_value",
                    "Rounded value is not an exact global bound",
                    {"reported_value": value},
                ),
            ],
        )
    return replace(
        result,
        certified=False,
        optimum_value=None,
        attained=None,
        certification_gap=None,
        certificate=certificate,
        status="candidate_inexact_value",
    )


def _structural_pipeline_result(
    objective,
    constraints,
    variables,
    parameters,
    result,
    box_bounds,
    options,
    precision,
):
    """Assemble an early exact proof without algebraization or complexity expansion."""
    problem = LiftedProblem(
        original_objective=objective,
        lifted_objective=objective,
        original_constraints=list(constraints),
        lifted_constraints=list(constraints),
        defining_equalities=[],
        branch_inequalities=[],
        strict_inequalities=[],
        graph_equalities=[],
        graph_inequalities=[],
        auxiliary_variables=[],
        substitution_map={},
        all_variables=list(variables),
        original_variables=list(variables),
        parameter_symbols=list(parameters),
        preprocessing_notes=["exact structural proof before algebraization"],
        structural_metadata={"early_structural_rule": result.kind},
    )
    return _execute_single_problem_stages(
        problem=problem,
        merged=box_bounds or {},
        original_vars=variables,
        symmetry_report=None,
        solver_plan=None,
        structural_result=result,
        direct=None,
        semialg_result=None,
        semialg_attempted=False,
        diagnostics=[],
        interval_max_boxes=options.interval_max_boxes,
        interval_tol=options.interval_tol,
        max_active_sets=options.max_active_sets,
        use_sos=options.use_sos,
        working_precision=precision.effective_precision,
        precision_provenance=precision.as_dict(),
    )


def _minimize_result_impl(
    objective,
    constraints,
    *,
    box_bounds,
    variables,
    allow_transcendentals,
    allow_rootof,
    working_precision,
    bounds,
    use_symbol_assumptions,
    _skip_monotone_reduction,
    _skip_decomposition,
    _skip_disjunction,
    solver_options: SolverOptions,
    _fast_depth=0,
) -> PipelineResult:
    """Normalize the mathematical problem, apply reductions, and dispatch it."""
    interval_max_boxes = solver_options.interval_max_boxes
    interval_tol = solver_options.interval_tol
    max_active_sets = solver_options.max_active_sets
    use_sos = solver_options.use_sos
    semialg_certification = solver_options.semialg_certification
    semialg_range_cost_limit = solver_options.semialg_range_cost_limit
    objective = sp.sympify(objective)
    explicit_vars = normalize_variables(variables)
    if box_bounds is not None:
        if not isinstance(box_bounds, Mapping):
            raise TypeError(
                "box_bounds must be a mapping from symbols to finite bound pairs"
            )
        if any(not isinstance(variable, sp.Symbol) for variable in box_bounds):
            raise TypeError("box_bounds keys must be SymPy symbols")
        box_bounds = IntervalBox(dict(box_bounds)).bounds

    # ``Or`` denotes a union of feasible regions. Optimize each conjunction and
    # combine the exact branch results through the same machinery used for
    # Piecewise objectives.
    if not _skip_disjunction:
        branches = normalize_constraint_branches(constraints)
        if not branches:
            branches = ([sp.Eq(sp.Integer(0), sp.Integer(1), evaluate=False)],)
        if len(branches) > 1:
            from .region_optimization import component_lower_bound

            branch_results = []
            incumbent = None
            for branch in branches:
                lower = None
                if incumbent is not None and explicit_vars is not None:
                    additional, _ = normalize_bounds_input(bounds, explicit_vars)
                    lower = component_lower_bound(
                        objective, [*branch, *additional], explicit_vars
                    )
                if (
                    lower is not None
                    and algebraic_compare(lower, incumbent.optimum_value) == 1
                ):
                    certificate = replace(
                        incumbent.certificate,
                        lower_bound=lower,
                        upper_bound=None,
                        primal_value=None,
                        primal_solution={},
                        proof_objects=[
                            ProofArtifact(
                                "component_bound",
                                "Dominated component",
                                {"constraints": tuple(branch), "lower_bound": lower},
                            )
                        ],
                    )
                    branch_results.append(
                        replace(
                            incumbent,
                            candidates=[],
                            best_candidate=None,
                            structural_result=None,
                            certified=False,
                            optimum_value=None,
                            attained=None,
                            certificate=certificate,
                            status="component_pruned",
                        )
                    )
                    continue
                result = _minimize_result_impl(
                    objective,
                    branch,
                    box_bounds=box_bounds,
                    variables=explicit_vars,
                    allow_transcendentals=allow_transcendentals,
                    allow_rootof=allow_rootof,
                    working_precision=working_precision,
                    bounds=bounds,
                    use_symbol_assumptions=use_symbol_assumptions,
                    _skip_monotone_reduction=_skip_monotone_reduction,
                    _skip_decomposition=_skip_decomposition,
                    _skip_disjunction=True,
                    solver_options=solver_options,
                )
                result = _exact_value_result(result, objective)
                branch_results.append(result)
                if (
                    result.certified
                    and result.attained is True
                    and result.optimum_value is not None
                    and result.certificate is not None
                    and (
                        incumbent is None
                        or algebraic_compare(
                            result.optimum_value, incumbent.optimum_value
                        )
                        == -1
                    )
                ):
                    incumbent = result
            return _merge_piecewise_results(branch_results)
        constraints = branches[0]
    else:
        constraints = normalize_constraints_input(constraints)

    bounds_constraints, normalized_box_bounds = normalize_bounds_input(
        bounds, explicit_vars
    )
    constraints = [*constraints, *bounds_constraints]
    if box_bounds:
        normalized_box_bounds.update(box_bounds)
    box_bounds = normalized_box_bounds or None

    pre_diags = []
    symbols = set(objective.free_symbols)
    for relation in constraints:
        symbols.update(getattr(relation, "free_symbols", set()))
    if explicit_vars is None:
        original_vars = sorted(symbols, key=lambda item: item.sort_key())
    else:
        original_vars = explicit_vars
    if solver_options.quadratic.backend != "scip":
        require_continuous_variables(original_vars)
    if box_bounds is not None:
        foreign = set(box_bounds) - set(original_vars)
        if foreign:
            names = ", ".join(
                sp.sstr(item) for item in sorted(foreign, key=sp.default_sort_key)
            )
            raise ValueError(
                f"box bound variable(s) {names} are not optimization variables"
            )
    if use_symbol_assumptions:
        constraints = [*constraints, *assumption_constraints(original_vars)]
        for relation in constraints:
            symbols.update(getattr(relation, "free_symbols", set()))
    constraints = preserve_rational_domain(objective, constraints, original_vars)
    parameter_symbols = sorted(
        symbols.difference(original_vars), key=lambda item: item.sort_key()
    )

    # Explicit decision variables that occur nowhere in the problem are free
    # coordinates. Solve only the active coordinates and retain the omitted
    # variables as a partial optimizer mapping representing the continuum.
    free_variables = [variable for variable in original_vars if variable not in symbols]
    active_variables = [variable for variable in original_vars if variable in symbols]
    if (
        free_variables
        and active_variables
        and solver_options.quadratic.backend in {"auto", "off"}
    ):
        subresult = _minimize_result_impl(
            objective,
            constraints,
            box_bounds=box_bounds,
            variables=active_variables,
            allow_transcendentals=allow_transcendentals,
            allow_rootof=allow_rootof,
            working_precision=working_precision,
            bounds=None,
            use_symbol_assumptions=False,
            _skip_monotone_reduction=_skip_monotone_reduction,
            _skip_decomposition=_skip_decomposition,
            _skip_disjunction=True,
            solver_options=solver_options,
        )
        metadata = dict(subresult.lifted_problem.structural_metadata)
        metadata["free_optimization_variables"] = tuple(free_variables)
        mapped_problem = replace(
            subresult.lifted_problem,
            original_variables=list(original_vars),
            structural_metadata=metadata,
        )
        return replace(subresult, lifted_problem=mapped_problem)

    precision_info = precision_provenance(objective, constraints, working_precision)
    effective_precision = precision_info.effective_precision

    if solver_options.quadratic.backend not in {"auto", "off"}:
        from dataclasses import asdict

        from .global_quadratic import minimize_quadratic
        from .quadratic_routing import quadratic_structural_result

        quadratic = minimize_quadratic(
            objective,
            constraints,
            original_vars,
            use_symbol_assumptions=False,
            **asdict(solver_options.quadratic),
        )
        result = _structural_pipeline_result(
            objective,
            constraints,
            original_vars,
            parameter_symbols,
            quadratic_structural_result(quadratic),
            box_bounds,
            solver_options,
            precision_info,
        )
        result.certificate.lower_bound = quadratic.lower_bound
        result.certificate.upper_bound = quadratic.upper_bound
        if not quadratic.exact or not quadratic.certified:
            result.certificate.status = quadratic.status
            result.status = f"quadratic_{quadratic.status}"
        return result

    fast_allowed = solver_options.structural_rules
    if not allow_rootof and any(e.has(sp.CRootOf) for e in (objective, *constraints)):
        fast_allowed = False
    if not allow_transcendentals and any(
        node.is_Function and node.func is not sp.Abs
        for e in (objective, *constraints)
        for node in sp.preorder_traversal(e)
    ):
        fast_allowed = False
    if fast_allowed:

        def solve_reduced(reduced_objective, reduced_constraints, reduced_variables):
            return _minimize_result_impl(
                reduced_objective,
                reduced_constraints,
                box_bounds=None,
                variables=list(reduced_variables),
                allow_transcendentals=allow_transcendentals,
                allow_rootof=allow_rootof,
                working_precision=working_precision,
                bounds=None,
                use_symbol_assumptions=False,
                _skip_monotone_reduction=False,
                _skip_decomposition=False,
                _skip_disjunction=False,
                solver_options=solver_options,
                _fast_depth=_fast_depth + 1,
            )

        fast_result = try_fast_rules(
            objective,
            constraints,
            original_vars,
            solve_reduced,
            depth=_fast_depth,
        )
        if fast_result is not None:
            return _structural_pipeline_result(
                objective,
                constraints,
                original_vars,
                parameter_symbols,
                fast_result,
                box_bounds,
                solver_options,
                precision_info,
            )

    # Reduce globally monotone outer functions to a simpler equivalent objective
    # before algebraization. This exposes polynomial interiors directly to LP/QP
    # and semialg rather than introducing unnecessary transcendental lifts.
    if not _skip_monotone_reduction:
        reduction = find_monotone_reduction(objective, constraints, original_vars)
        if reduction is not None:
            subresult = _minimize_result_impl(
                reduction.reduced_objective,
                [*constraints, *reduction.added_constraints],
                box_bounds=box_bounds,
                variables=original_vars,
                allow_transcendentals=allow_transcendentals,
                allow_rootof=allow_rootof,
                working_precision=working_precision,
                bounds=None,
                use_symbol_assumptions=False,
                _skip_monotone_reduction=True,
                _skip_decomposition=_skip_decomposition,
                _skip_disjunction=False,
                solver_options=solver_options,
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
                    _minimize_result_impl(
                        component.objective,
                        component.constraints,
                        box_bounds=component_box,
                        variables=component.variables,
                        allow_transcendentals=allow_transcendentals,
                        allow_rootof=allow_rootof,
                        working_precision=working_precision,
                        bounds=None,
                        use_symbol_assumptions=False,
                        _skip_monotone_reduction=False,
                        _skip_decomposition=True,
                        _skip_disjunction=False,
                        solver_options=solver_options,
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

    # Function-property analysis accepts more function heads than algebraization.
    # Keep the rational and affine paths free of optional analysis work.
    if (
        fast_allowed
        and allow_transcendentals
        and len(original_vars) == 1
        and not parameter_symbols
        and not objective.is_rational_function(original_vars[0])
        and not objective.has(sp.Float, sp.Piecewise)
        and not any(relation.has(sp.Float) for relation in constraints)
        and not objective.atoms(sp.core.function.AppliedUndef)
        and any(
            node.is_Function
            and node.func not in Algebraizer.TRANSCENDENTAL_FUNCS
            and node.func is not sp.Abs
            for expression in (objective, *constraints)
            for node in sp.preorder_traversal(expression)
        )
    ):
        analyzed = try_structural_solver(
            objective, constraints, original_vars, working_precision=effective_precision
        )
        if analyzed is None:
            analyzed = try_univariate_range(objective, constraints, original_vars)
        if analyzed is not None and analyzed.certified:
            return _structural_pipeline_result(
                objective,
                constraints,
                original_vars,
                parameter_symbols,
                analyzed,
                box_bounds,
                solver_options,
                precision_info,
            )

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
                    allow_range_fallback=not _skip_monotone_reduction,
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
                    allow_range_fallback=not _skip_monotone_reduction,
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
        allow_range_fallback=not _skip_monotone_reduction,
    )


def _swap_min_max_label(label: str | None) -> str | None:
    if not label:
        return label
    mapped = label.replace("Minimum", "__MAXIMUM__").replace("Maximum", "Minimum")
    mapped = mapped.replace("__MAXIMUM__", "Maximum")
    mapped = mapped.replace("minimum", "__maximum__").replace("maximum", "minimum")
    return mapped.replace("__maximum__", "maximum")


def _mapped_candidate(
    candidate: CandidateSolution,
    equality_multipliers=(),
) -> CandidateSolution:
    """Return a maximization-view candidate without mutating the minimum result."""
    multipliers = dict(candidate.multiplier_values)
    for multiplier in equality_multipliers:
        if multiplier in multipliers:
            multipliers[multiplier] = sp.simplify(-multipliers[multiplier])
    numeric = candidate.objective_value_numeric
    return replace(
        candidate,
        primal_values=dict(candidate.primal_values),
        multiplier_values=multipliers,
        original_variable_values=dict(candidate.original_variable_values),
        objective_value=simplify_scalar(-candidate.objective_value),
        objective_value_numeric=None if numeric is None else -numeric,
        source=f"maximize:{candidate.source}",
        local_classification=_swap_min_max_label(candidate.local_classification),
    )


def _mapped_kkt_system(kkt):
    if kkt is None:
        return None
    stationarity = [
        sp.Eq(sp.expand(-equation.lhs), 0, evaluate=False)
        for equation in kkt.stationarity_equations
    ]
    return replace(
        kkt,
        lagrangian=sp.expand(-kkt.lagrangian),
        stationarity_equations=stationarity,
        primal_equalities=list(kkt.primal_equalities),
        primal_inequalities=list(kkt.primal_inequalities),
        dual_inequalities=list(kkt.dual_inequalities),
        complementarity_eqs=list(kkt.complementarity_eqs),
    )


def _mapped_critical_result(item, equality_multipliers=()):
    if not is_dataclass(item):
        return item
    changes = {}
    if hasattr(item, "classification"):
        changes["classification"] = _swap_min_max_label(item.classification)
    if getattr(item, "lagrangian", None) is not None:
        changes["lagrangian"] = sp.simplify(-item.lagrangian)
    if getattr(item, "hessian_at_point", None) is not None:
        changes["hessian_at_point"] = -item.hessian_at_point
    if getattr(item, "eigenvalues", None) is not None:
        changes["eigenvalues"] = [sp.simplify(-value) for value in item.eigenvalues]
    if hasattr(item, "solution"):
        solution = dict(item.solution)
        for multiplier in equality_multipliers:
            if multiplier in solution:
                solution[multiplier] = sp.simplify(-solution[multiplier])
        changes["solution"] = solution
    try:
        return replace(item, **changes)
    except TypeError:
        return item


def _map_min_result_to_max(
    result: PipelineResult, objective: sp.Expr
) -> PipelineResult:
    """Map a minimum of ``-objective`` to a non-mutating maximization result."""
    equality_multipliers = (
        tuple(result.kkt_system.equality_multipliers)
        if result.kkt_system is not None
        else ()
    )
    mapped_candidates = [
        _mapped_candidate(item, equality_multipliers) for item in result.candidates
    ]
    mapped_kkt = _mapped_kkt_system(result.kkt_system)
    mapped_critical = [
        _mapped_critical_result(item, equality_multipliers)
        for item in result.critical_point_results
    ]
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
        structural = replace(structural, value=simplify_scalar(-structural.value))

    semialg_result = result.semialg_result
    if (
        semialg_result is not None
        and is_dataclass(semialg_result)
        and getattr(semialg_result, "value", None) is not None
    ):
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
                else simplify_scalar(-certificate.primal_value)
            ),
            lower_bound=None if old_upper is None else simplify_scalar(-old_upper),
            upper_bound=None if old_lower is None else simplify_scalar(-old_lower),
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
        kkt_system=mapped_kkt,
        candidates=mapped_candidates,
        best_candidate=best,
        interval_result=interval,
        sos_lower_bound=None,
        sos_bound_numeric=None,
        certificate=certificate,
        diagnostics=list(result.diagnostics),
        critical_point_results=mapped_critical,
        inferred_box_bounds=dict(result.inferred_box_bounds),
        precision_provenance=dict(result.precision_provenance),
        semialg_result=semialg_result,
        semialg_range=semialg_range,
        structural_result=structural,
        optimum_value=(
            None
            if result.optimum_value is None
            else simplify_scalar(-result.optimum_value)
        ),
        status=result.status.replace("minimum", "maximum").replace(
            "infimum", "supremum"
        ),
        optimization_sense="maximize",
        kkt_objective=(objective if mapped_kkt is not None else None),
    )


def _univariate_limit_optimizer(result: PipelineResult) -> dict[sp.Symbol, sp.Expr]:
    variables = list(result.lifted_problem.original_variables)
    if len(variables) != 1 or result.optimum_value is None:
        return {}
    variable = variables[0]
    objective = result.lifted_problem.original_objective
    constraints = result.lifted_problem.original_constraints
    try:
        from sympy.calculus.util import continuous_domain

        domain = continuous_domain(objective, variable, sp.S.Reals)
        for relation in constraints:
            if not relation.free_symbols.issubset({variable}):
                return {}
            domain = sp.Intersection(domain, relation.as_set())
    except (AttributeError, NotImplementedError, TypeError, ValueError):
        return {}
    if domain is sp.S.EmptySet:
        return {}

    target = sp.simplify(result.optimum_value)
    candidates = []
    try:
        infimum = domain.inf
        supremum = domain.sup
        if infimum is -sp.oo:
            candidates.append((-sp.oo, -sp.oo, "+"))
        elif domain.contains(infimum) is not sp.S.true:
            candidates.append((infimum, infimum, "+"))
        if supremum is sp.oo:
            candidates.append((sp.oo, sp.oo, "-"))
        elif domain.contains(supremum) is not sp.S.true:
            candidates.append((supremum, supremum, "-"))
    except (AttributeError, NotImplementedError, TypeError, ValueError):
        return {}

    for point, limit_point, direction in candidates:
        try:
            if point in (sp.oo, -sp.oo):
                value = sp.limit(objective, variable, point)
            else:
                value = sp.limit(objective, variable, limit_point, dir=direction)
            if value == target or sp.simplify(value - target) == 0:
                return {variable: point}
        except (NotImplementedError, TypeError, ValueError, RecursionError):
            continue
    return {}


def _compact_extremum(result: PipelineResult):
    optimum = result.optimum_value
    points = _optimizer_points(result)
    if len(points) == 1:
        optimizer = points[0]
    elif len(points) > 1:
        optimizer = points
    elif result.certified and result.attained is False:
        optimizer = _univariate_limit_optimizer(result)
    else:
        optimizer = {}
    return [optimum, optimizer]


def minimize(
    objective: sp.Expr,
    constraints: object = None,
    *,
    full_output: bool = False,
    **kwargs,
):
    """Minimize and return ``[optimum, optimizer]`` by default.

    The optimizer is a dictionary from optimization variables to exact values
    whenever possible. For certified unattained one-variable extrema, limiting
    values such as ``-oo`` or ``oo`` are used when the asymptotic direction is
    established. Set ``full_output=True`` to return the complete
    :class:`PipelineResult` with candidates, diagnostics, and certificates.
    """
    result = minimize_result(objective, constraints, **kwargs)
    return result if full_output else _compact_extremum(result)


def maximize_result(
    objective: sp.Expr,
    constraints: object = None,
    **kwargs,
) -> PipelineResult:
    """Return the complete certified-pipeline result for maximization."""
    objective = sp.sympify(objective)
    return _map_min_result_to_max(
        minimize_result(-objective, constraints, **kwargs), objective
    )


def maximize(
    objective: sp.Expr,
    constraints: object = None,
    *,
    full_output: bool = False,
    **kwargs,
):
    """Maximize and return ``[optimum, optimizer]`` by default."""
    result = maximize_result(objective, constraints, **kwargs)
    return result if full_output else _compact_extremum(result)


def minval(
    objective: sp.Expr,
    constraints: object = None,
    **kwargs,
) -> sp.Expr | None:
    """Return the certified minimum value or infimum when available."""
    result = minimize_result(objective, constraints, **kwargs)
    if result.optimum_value is not None:
        return simplify_scalar(result.optimum_value)
    if result.best_candidate is not None and result.certified:
        return simplify_scalar(result.best_candidate.objective_value)
    return None


def maxval(
    objective: sp.Expr,
    constraints: object = None,
    **kwargs,
) -> sp.Expr | None:
    """Return the certified maximum value or supremum when available."""
    result = maximize_result(objective, constraints, **kwargs)
    if result.optimum_value is not None:
        return simplify_scalar(result.optimum_value)
    if result.best_candidate is not None and result.certified:
        return simplify_scalar(result.best_candidate.objective_value)
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
        if result.method_label != "branch" and not _exact_source_candidate(
            result.lifted_problem, candidate, result.structural_result
        ):
            continue
        point = candidate.original_variable_values or candidate.primal_values
        if not point:
            continue
        if (
            optimum is not None
            and candidate.objective_value != optimum
            and sp.simplify(candidate.objective_value - optimum) != 0
        ):
            continue
        points.append(dict(point))
    # For certified one-dimensional extrema, recover all exact isolated
    # minimizers/maximizers from the level equation when SymPy can represent
    # them finitely. This prevents the compact API from arbitrarily hiding
    # equivalent optima such as x = -1 and x = 1.
    original_variables = list(result.lifted_problem.original_variables)
    structural_kind = getattr(result.structural_result, "kind", None)
    can_solve_level_set = (
        not result.lifted_problem.parameter_symbols
        and structural_kind
        not in {
            "periodic_trig",
            "rational_frequency_trig",
            "mixed_periodic",
            "parametric_quadratic",
            "univariate_polynomial",
            "univariate_rational",
            "power_substitution",
            "affine_ridge",
            "radial_reduction",
            "affine_elimination",
        }
    )
    if structural_kind == "nonnegative_sum" and result.structural_result.metadata.get(
        "affine_zero_conditions"
    ):
        can_solve_level_set = False
    if can_solve_level_set and len(original_variables) == 1 and optimum is not None:
        variable = original_variables[0]
        objective = result.lifted_problem.original_objective
        try:
            domain = sp.S.Reals
            from sympy.calculus.util import continuous_domain

            domain = continuous_domain(objective, variable, domain)
            for relation in result.lifted_problem.original_constraints:
                if relation.free_symbols.issubset({variable}):
                    domain = sp.Intersection(domain, relation.as_set())
            level_set = sp.solveset(sp.Eq(objective, optimum), variable, domain=domain)
            if isinstance(level_set, sp.FiniteSet):
                points.extend({variable: point} for point in level_set)
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            pass

    unique = []
    seen = set()
    for point in points:
        key = tuple(
            sorted((sp.sstr(k), sp.sstr(simplify_scalar(v))) for k, v in point.items())
        )
        if key not in seen:
            seen.add(key)
            unique.append(point)

    def point_sort_key(point):
        return tuple(
            (sp.default_sort_key(variable), sp.default_sort_key(simplify_scalar(value)))
            for variable, value in sorted(
                point.items(), key=lambda item: item[0].sort_key()
            )
        )

    unique.sort(key=point_sort_key)
    return tuple(unique)


def argmin(
    objective: sp.Expr,
    constraints: object = None,
    **kwargs,
) -> tuple[dict[sp.Symbol, sp.Expr], ...]:
    """Return represented attained minimizers as variable-to-value mappings."""
    return _optimizer_points(minimize_result(objective, constraints, **kwargs))


def argmax(
    objective: sp.Expr,
    constraints: object = None,
    **kwargs,
) -> tuple[dict[sp.Symbol, sp.Expr], ...]:
    """Return represented attained maximizers as variable-to-value mappings."""
    return _optimizer_points(maximize_result(objective, constraints, **kwargs))
