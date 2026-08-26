import numpy as np
from symbopt import (
    SDPProblem,
    SDPSolverConfig,
    solve_sdp,
    solve_sdp_dense,
)


def diag_problem():
    C = np.eye(2)
    A = [np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [0.0, 1.0]])]
    return SDPProblem.single_block(C=C, A=A, b=np.array([1.0, 2.0]))


def test_sdp_dense_known_objective():
    r = solve_sdp_dense(diag_problem(), max_iter=80, tol=1e-8)
    assert r.status in {"optimal", "approximate", "max_iter"}
    assert abs(r.primal_objective - 3) < 0.2


def test_sdp_unified_config_api():
    r = solve_sdp(
        diag_problem(), SDPSolverConfig(max_iter=80, tol=1e-8, prefer_sparse=False)
    )
    assert abs(r.primal_objective - 3) < 0.2


def test_sdp_default_config():
    r = solve_sdp(diag_problem(), SDPSolverConfig(prefer_sparse=False))
    assert abs(r.primal_objective - 3) < 0.2


def test_sparse_decomposition_uses_constraint_sparsity():
    from symbopt.sdp import chordalize_sdp_block

    C = np.eye(2)
    A = [np.array([[0.0, 0.5], [0.5, 0.0]])]
    p = SDPProblem.single_block(C=C, A=A, b=np.array([0.0]))
    q = chordalize_sdp_block(p)
    assert q.num_blocks == 1
    assert np.allclose(q.A_blocks[0][0], A[0])


def test_sparse_decomposition_preserves_scalar_coupling():
    from symbopt.sdp import chordalize_sdp_block

    C = np.diag([1.0, 2.0])
    A = [np.diag([1.0, 0.0]), np.diag([0.0, 1.0])]
    B = np.array([[1.0], [2.0]])
    c = np.array([3.0])
    p = SDPProblem.single_block(C=C, A=A, b=np.array([1.0, 2.0]), B=B, c=c)
    q = chordalize_sdp_block(p)
    assert q.num_blocks == 2
    assert np.allclose(q.B, B)
    assert np.allclose(q.c, c)


def test_sparse_solver_reconstructs_original_block_shape():
    from symbopt import solve_sdp_sparse_chordal

    p = diag_problem()
    r = solve_sdp_sparse_chordal(p, max_iter=80, tol=1e-6)
    assert len(r.X_blocks) == 1
    assert r.X_blocks[0].shape == (2, 2)


def test_sdp_dual_residual_is_for_original_scaling():
    p = diag_problem()
    r = solve_sdp_dense(p, max_iter=120, tol=1e-6)
    combo = sum(r.y[i] * p.A_blocks[i][0] for i in range(p.constraint_count))
    residual = p.C_blocks[0] - combo - r.S_blocks[0]
    assert np.linalg.norm(residual) <= max(1e-4, 10 * r.dual_residual_norm)


def test_overlapping_chordal_sdp_adds_separator_equalities():
    from symbopt.sdp import chordalize_sdp_block

    C = np.diag([1.0, 2.0, 3.0])
    A = [
        np.array([[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0]]),
    ]
    p = SDPProblem.single_block(C=C, A=A, b=np.zeros(2))
    q = chordalize_sdp_block(p)
    assert q.num_blocks == 2
    assert q.block_sizes == [2, 2]
    assert q.metadata["chordal_cliques"] == [[0, 1], [1, 2]]
    assert q.constraint_count == p.constraint_count + 1
    # The added row enforces equality of the duplicated X[1,1] entry.
    consistency = q.A_blocks[-1]
    assert abs(np.trace(consistency[0])) == 1.0
    assert abs(np.trace(consistency[1])) == 1.0
    assert np.isclose(np.trace(consistency[0]) + np.trace(consistency[1]), 0.0)


def test_overlapping_chordal_solver_reconstructs_original_shape_and_objective():
    from symbopt import solve_sdp_sparse_chordal

    C = np.eye(3)
    A = [
        np.diag([1.0, 0.0, 0.0]),
        np.diag([0.0, 1.0, 0.0]),
        np.diag([0.0, 0.0, 1.0]),
        np.array([[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0]]),
    ]
    p = SDPProblem.single_block(C=C, A=A, b=np.array([1.0, 2.0, 3.0, 0.0, 0.0]))
    result = solve_sdp_sparse_chordal(p, max_iter=120, tol=1e-6)
    assert len(result.X_blocks) == 1
    assert result.X_blocks[0].shape == (3, 3)
    assert abs(result.primal_objective - 6.0) < 0.25
    assert result.primal_residual_norm < 0.1
    assert "overlapping chordal" in result.message
