from __future__ import annotations

from .algebraize import (
    Algebraizer,
    algebraize,
    algebraize_cases,
)
from .candidates import (
    solve_exact_candidates,
    solve_kkt_candidates,
)
from .constrained_local import (
    ConstrainedLocalResult,
    maximize_local_constrained,
    minimize_local_constrained,
)
from .convex_routing import infer_box_bounds
from .critical_points import (
    CriticalConeResult,
    CriticalPointResult,
    analyze_critical_cone,
    critical_points,
)
from .diagnostics import (
    DiagnosticKind,
    SolverDiagnostic,
)
from .exact_refinement import (
    ExactRefinementResult,
    project_affine_exact,
    reconstruct_exact_vector,
    refine_exact_system,
)
from .extrema_bounds import (
    ExtremaBoundsProof,
    certify_extrema_bounds,
    check_extrema_bounds,
)
from .global_quadratic import (
    minimize_quadratic,
    solve_global_qp,
    verify_quadratic_certificate,
)
from .hierarchy import search_sos_hierarchy
from .intervals import interval_optimize
from .kkt import build_kkt_system
from .local_optimization import (
    LocalModeResult,
    maximize_local,
    minimize_local,
)
from .lp import (
    revised_simplex,
    simplex,
    solve_lp,
)
from .moment import (
    AtomExtractionResult,
    LocalizingMatrix,
    MomentMatrix,
    MomentRelaxation,
    build_localizing_matrix,
    build_moment_matrix,
    build_moment_relaxation,
    extract_moment_atoms,
)
from .moment_bridge import link_sos_moments
from .moments_dual import solve_moment_dual
from .planner import ProblemComplexity, SolverPlan, analyze_problem, plan_problem
from .polynomial_structure import (
    build_sparsity_graph,
    build_symmetry_report,
)
from .positivstellensatz import (
    PositivstellensatzCertificate,
    PositivstellensatzResult,
    PreorderingGenerator,
    preordering_generators,
    search_infeasibility_certificate,
    verify_infeasibility_certificate,
)
from .problem_models import (
    BranchCase,
    CandidateSolution,
    IntervalResult,
    KKTSystem,
    LiftedProblem,
    LiftRecord,
    OptimalityCertificate,
    PipelineResult,
    ProofArtifact,
)
from .qp import solve_convex_qp
from .quadratic_models import GlobalQuadraticResult
from .quadratic_scip import scip_available
from .rational_optimization import rational_graph
from .region_inputs import PreparedRegion, RegionConversionError, prepare_region
from .region_queries import OptimizationFamily, optimization_family, optimizer_locus
from .sdp import (
    SDPProblem,
    SDPResult,
    available_external_sdp_solvers,
    cvxpy_available,
    solve_sdp_cvxpy,
    solve_sdp_dense,
    solve_sdp_sparse_chordal,
)
from .semialgebraic import (
    SemialgCapabilities,
    SemialgOptResult,
    SemialgProof,
    SemialgRangeResult,
    function_range_exact,
    get_semialg_capabilities,
    prove_nonnegative_exact,
    semialg_available,
)
from .semidefinite import (
    SDPSolverConfig,
    solve_sdp,
)
from .solver_pipeline import (
    argmax,
    argmin,
    maximize,
    maximize_result,
    maxval,
    minimize,
    minimize_result,
    minval,
)
from .sos import (
    SOSCertificate,
    SOSRelaxationResult,
    SOSResult,
    sos_decompose,
    sos_lower_bound,
    sos_poly_relaxation,
    sos_scalar_relaxation,
)
from .sos_search import (
    SOSCertificateSearchResult,
    SOSCostEstimate,
    SOSSearchPlan,
    estimate_sos_cost,
    plan_sos_search,
    search_sos_certificate,
)
from .sublevel_bounds import polynomial_sublevel_bounds
from .symbolic_cache import (
    cache_info,
    clear_symbolic_caches,
)
from .univariate_certificates import GlobalMinimumCertificate, certify_global_minimum
from .univariate_optimization import ComponentMinimumResult, finite_component_minimum
from .univariate_verification import verify_univariate_certificate
from .verified_intervals import (
    EvalConfig,
    VerifiedBound,
    arb_available,
    enclose_expression,
)

__all__ = [
    "Algebraizer",
    "AtomExtractionResult",
    "BranchCase",
    "CandidateSolution",
    "ComponentMinimumResult",
    "ConstrainedLocalResult",
    "CriticalConeResult",
    "CriticalPointResult",
    "DiagnosticKind",
    "EvalConfig",
    "ExactRefinementResult",
    "ExtremaBoundsProof",
    "GlobalMinimumCertificate",
    "GlobalQuadraticResult",
    "IntervalResult",
    "KKTSystem",
    "LiftRecord",
    "LiftedProblem",
    "LocalModeResult",
    "LocalizingMatrix",
    "MomentMatrix",
    "MomentRelaxation",
    "OptimalityCertificate",
    "OptimizationFamily",
    "PipelineResult",
    "PositivstellensatzCertificate",
    "PositivstellensatzResult",
    "PreorderingGenerator",
    "PreparedRegion",
    "ProblemComplexity",
    "ProofArtifact",
    "RegionConversionError",
    "SDPProblem",
    "SDPResult",
    "SDPSolverConfig",
    "SOSCertificate",
    "SOSCertificateSearchResult",
    "SOSCostEstimate",
    "SOSRelaxationResult",
    "SOSResult",
    "SOSSearchPlan",
    "SemialgCapabilities",
    "SemialgOptResult",
    "SemialgProof",
    "SemialgRangeResult",
    "SolverDiagnostic",
    "SolverPlan",
    "VerifiedBound",
    "algebraize",
    "algebraize_cases",
    "analyze_critical_cone",
    "analyze_problem",
    "arb_available",
    "argmax",
    "argmin",
    "available_external_sdp_solvers",
    "build_kkt_system",
    "build_localizing_matrix",
    "build_moment_matrix",
    "build_moment_relaxation",
    "build_sparsity_graph",
    "build_symmetry_report",
    "cache_info",
    "certify_extrema_bounds",
    "certify_global_minimum",
    "check_extrema_bounds",
    "clear_symbolic_caches",
    "critical_points",
    "cvxpy_available",
    "enclose_expression",
    "estimate_sos_cost",
    "extract_moment_atoms",
    "finite_component_minimum",
    "function_range_exact",
    "get_semialg_capabilities",
    "infer_box_bounds",
    "interval_optimize",
    "link_sos_moments",
    "maximize",
    "maximize_local",
    "maximize_local_constrained",
    "maximize_result",
    "maxval",
    "minimize",
    "minimize_local",
    "minimize_local_constrained",
    "minimize_quadratic",
    "minimize_result",
    "minval",
    "optimization_family",
    "optimizer_locus",
    "plan_problem",
    "plan_sos_search",
    "polynomial_sublevel_bounds",
    "preordering_generators",
    "prepare_region",
    "project_affine_exact",
    "prove_nonnegative_exact",
    "rational_graph",
    "reconstruct_exact_vector",
    "refine_exact_system",
    "revised_simplex",
    "scip_available",
    "search_infeasibility_certificate",
    "search_sos_certificate",
    "search_sos_hierarchy",
    "semialg_available",
    "simplex",
    "solve_convex_qp",
    "solve_exact_candidates",
    "solve_global_qp",
    "solve_kkt_candidates",
    "solve_lp",
    "solve_moment_dual",
    "solve_sdp",
    "solve_sdp_cvxpy",
    "solve_sdp_dense",
    "solve_sdp_sparse_chordal",
    "sos_decompose",
    "sos_lower_bound",
    "sos_poly_relaxation",
    "sos_scalar_relaxation",
    "verify_infeasibility_certificate",
    "verify_quadratic_certificate",
    "verify_univariate_certificate",
]
