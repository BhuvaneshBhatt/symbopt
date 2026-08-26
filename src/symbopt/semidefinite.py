from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .diagnostics import DiagnosticKind, SolverDiagnostic
from .sdp import SDPProblem, SDPResult, solve_sdp_dense, solve_sdp_sparse_chordal


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


def _result_score(result: SDPResult):
    return (
        result.primal_residual_norm,
        result.dual_residual_norm,
        result.gap,
    )


def _best_result(
    first: Optional[SDPResult],
    second: Optional[SDPResult],
) -> Optional[SDPResult]:
    if first is None:
        return second
    if second is None:
        return first
    return first if _result_score(first) <= _result_score(second) else second


def solve_sdp(
    problem: SDPProblem,
    config: Optional[SDPSolverConfig] = None,
    warm_start: Optional[dict] = None,
) -> SDPResult:
    """Solve an ``SDPProblem`` with the package SDP backend.

    The wrapper may compare a conservative sparse block route with the dense route
    and returns the numerically stronger result. Numerical SDP success is not by
    itself an exact symbolic certificate."""
    cfg = config or SDPSolverConfig()
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
    if result is None:
        raise RuntimeError("No SDP solve route produced a result.")
    return result
