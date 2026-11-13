from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .diagnostics import DiagnosticKind, SolverDiagnostic
from .sdp import (
    SDPProblem,
    SDPResult,
    available_external_sdp_solvers,
    cvxpy_available,
    solve_sdp_cvxpy,
    solve_sdp_dense,
    solve_sdp_sparse_chordal,
)


@dataclass
class SDPSolverConfig:
    """Configuration for the internal semidefinite solver.

    The controls govern iteration budget, feasibility tolerance, regularization,
    verbosity, and conservative sparse-block routing."""

    max_iter: int = 100
    tol: float = 1e-7
    reg: float = 1e-8
    verbose: bool = False
    prefer_sparse: bool = True
    sparse_size_threshold: int = 4
    backend: str = "auto"
    external_solver: str | None = None
    external_size_threshold: int = 8


def _result_score(result: SDPResult):
    return (
        result.primal_residual_norm,
        result.dual_residual_norm,
        result.gap,
    )


def _best_result(
    first: SDPResult | None,
    second: SDPResult | None,
) -> SDPResult | None:
    if first is None:
        return second
    if second is None:
        return first
    return first if _result_score(first) <= _result_score(second) else second


def solve_sdp(
    problem: SDPProblem,
    config: SDPSolverConfig | None = None,
    warm_start: dict | None = None,
) -> SDPResult:
    """Solve an ``SDPProblem`` with the package SDP backend.

    The wrapper may compare a conservative sparse block route with the dense route
    and returns the numerically stronger result. Numerical SDP success is not by
    itself an exact symbolic certificate."""
    cfg = config or SDPSolverConfig()
    if cfg.backend not in {"auto", "internal", "cvxpy", "external"}:
        raise ValueError(
            "SDP backend must be 'auto', 'internal', 'cvxpy', or 'external'."
        )

    largest_block = max(problem.block_sizes)
    wants_external = cfg.backend in {"cvxpy", "external"} or (
        cfg.backend == "auto"
        and largest_block >= cfg.external_size_threshold
        and cvxpy_available()
        and bool(available_external_sdp_solvers())
    )
    if wants_external:
        try:
            return solve_sdp_cvxpy(
                problem,
                solver=cfg.external_solver,
                max_iter=max(1000, cfg.max_iter),
                tol=cfg.tol,
                verbose=cfg.verbose,
            )
        except (ImportError, RuntimeError, ValueError) as exc:
            if cfg.backend in {"cvxpy", "external"}:
                raise
            external_diag = SolverDiagnostic.from_exception(
                "sdp_external", "cvxpy", DiagnosticKind.NUMERICAL_FAILURE, exc
            )
    else:
        external_diag = None

    best = None

    if (
        cfg.prefer_sparse
        and problem.num_blocks == 1
        and problem.C_blocks[0].shape[0] >= cfg.sparse_size_threshold
    ):
        try:
            best = solve_sdp_sparse_chordal(
                problem,
                max_iter=cfg.max_iter,
                tol=cfg.tol,
                reg=cfg.reg,
                verbose=cfg.verbose,
            )
        except (
            ValueError,
            RuntimeError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as exc:
            sparse_diag = SolverDiagnostic.from_exception(
                "sdp_sparse",
                "sparse_route",
                DiagnosticKind.NUMERICAL_FAILURE,
                exc,
            )
            best = None
        else:
            sparse_diag = None
    else:
        sparse_diag = None

    dense = solve_sdp_dense(
        problem,
        max_iter=cfg.max_iter,
        tol=cfg.tol,
        reg=cfg.reg,
        verbose=cfg.verbose,
        warm_start=warm_start,
    )
    result = _best_result(best, dense)
    if result is not None and sparse_diag is not None:
        result.diagnostics.append(sparse_diag)
    if result is not None and external_diag is not None:
        result.diagnostics.append(external_diag)
    if result is None:
        raise RuntimeError("No SDP solve route produced a result.")
    return result
