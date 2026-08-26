from __future__ import annotations

from .algebraize import (
    Algebraizer,
    algebraize,
    algebraize_cases,
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
from .polynomial_structure import (
    build_sparsity_graph,
    build_symmetry_report,
)
from .problem_models import (
    BranchCase,
    CandidateSolution,
    IntervalResult,
    KKTSystem,
    LiftRecord,
    LiftedProblem,
    OptimalityCertificate,
    PipelineResult,
    ProofArtifact,
)
from .critical_points import (
    CriticalPointResult,
    critical_points,
)
from .diagnostics import (
    DiagnosticKind,
    SolverDiagnostic,
)
from .verified_intervals import (
    EvalConfig,
    VerifiedBound,
    arb_available,
    enclose_expression,
)
from .sdp import (
    SDPProblem,
    SDPResult,
    solve_sdp_dense,
    solve_sdp_sparse_chordal,
)
from .semidefinite import (
    SDPSolverConfig,
    solve_sdp,
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
from .solver_pipeline import (
    argmax,
    argmin,
    maximize,
    maxval,
    minimize,
    minval,
)
from .kkt import build_kkt_system
from .symbolic_cache import (
    cache_info,
    clear_symbolic_caches,
)
from .convex_routing import infer_box_bounds
from .intervals import interval_optimize
from .moment_bridge import link_sos_moments
from .lp import (
    revised_simplex,
    simplex,
    solve_lp,
)
from .hierarchy import search_sos_hierarchy
from .qp import solve_convex_qp
from .candidates import (
    solve_exact_candidates,
    solve_kkt_candidates,
)
from .moments_dual import solve_moment_dual

__all__ = [
    "Algebraizer",
    "AtomExtractionResult",
    "BranchCase",
    "CandidateSolution",
    "CriticalPointResult",
    "DiagnosticKind",
    "EvalConfig",
    "IntervalResult",
    "KKTSystem",
    "LiftRecord",
    "LiftedProblem",
    "LocalizingMatrix",
    "MomentMatrix",
    "MomentRelaxation",
    "OptimalityCertificate",
    "PipelineResult",
    "ProofArtifact",
    "SDPProblem",
    "SDPResult",
    "SDPSolverConfig",
    "SOSCertificate",
    "SOSRelaxationResult",
    "SOSResult",
    "SemialgCapabilities",
    "SemialgOptResult",
    "SemialgProof",
    "SemialgRangeResult",
    "SolverDiagnostic",
    "VerifiedBound",
    "algebraize",
    "algebraize_cases",
    "arb_available",
    "argmax",
    "argmin",
    "build_kkt_system",
    "build_localizing_matrix",
    "build_moment_matrix",
    "build_moment_relaxation",
    "build_sparsity_graph",
    "build_symmetry_report",
    "cache_info",
    "clear_symbolic_caches",
    "critical_points",
    "enclose_expression",
    "extract_moment_atoms",
    "function_range_exact",
    "get_semialg_capabilities",
    "infer_box_bounds",
    "interval_optimize",
    "link_sos_moments",
    "maximize",
    "maxval",
    "minimize",
    "minval",
    "prove_nonnegative_exact",
    "revised_simplex",
    "search_sos_hierarchy",
    "semialg_available",
    "simplex",
    "solve_convex_qp",
    "solve_exact_candidates",
    "solve_kkt_candidates",
    "solve_lp",
    "solve_moment_dual",
    "solve_sdp",
    "solve_sdp_dense",
    "solve_sdp_sparse_chordal",
    "sos_decompose",
    "sos_lower_bound",
    "sos_poly_relaxation",
    "sos_scalar_relaxation",
]
