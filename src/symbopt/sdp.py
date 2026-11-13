from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import find_spec

import numpy as np
from scipy.optimize import minimize

from .chordal import SparsityGraph as IndexedSparsityGraph
from .chordal import chordal_extension
from .diagnostics import DiagnosticKind, SolverDiagnostic


@dataclass
class SDPProblem:
    """Block-structured primal semidefinite program with optional nonnegative scalar variables."""

    C_blocks: list[np.ndarray]
    A_blocks: list[list[np.ndarray]]
    b: np.ndarray
    B: np.ndarray | None = None
    c: np.ndarray | None = None
    name: str = "dense_sdp"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.C_blocks = [
            _symmetrize(np.asarray(block, dtype=float)) for block in self.C_blocks
        ]
        self.A_blocks = [
            [_symmetrize(np.asarray(block, dtype=float)) for block in row]
            for row in self.A_blocks
        ]
        self.b = np.asarray(self.b, dtype=float).reshape(-1)
        if self.B is not None:
            self.B = np.asarray(self.B, dtype=float)
        if self.c is not None:
            self.c = np.asarray(self.c, dtype=float).reshape(-1)

        if not self.C_blocks:
            raise ValueError("At least one PSD block is required.")
        if len(self.A_blocks) != len(self.b):
            raise ValueError("Constraint rows must match b.")

        shapes = [block.shape for block in self.C_blocks]
        for row in self.A_blocks:
            if len(row) != len(self.C_blocks):
                raise ValueError("Each constraint row needs one matrix per PSD block.")
            for block, shape in zip(row, shapes):
                if block.shape != shape:
                    raise ValueError("Constraint block shape mismatch.")

        if self.B is not None and self.B.shape[0] != len(self.b):
            raise ValueError("B row count must match b.")
        if self.B is None and self.c is not None:
            raise ValueError("c requires B.")
        if self.B is not None and self.c is None:
            self.c = np.zeros(self.B.shape[1], dtype=float)

    @classmethod
    def single_block(
        cls,
        C,
        A,
        b,
        B=None,
        c=None,
        name="dense_sdp",
    ):
        """Build a one-block SDP from objective and affine constraint matrices."""
        return cls(
            C_blocks=[C],
            A_blocks=[[block] for block in A],
            b=b,
            B=B,
            c=c,
            name=name,
        )

    @property
    def num_blocks(self) -> int:
        """Number of PSD matrix blocks in the numerical SDP."""
        return len(self.C_blocks)

    @property
    def block_sizes(self) -> list[int]:
        """Dimensions of the PSD matrix blocks in the stored order."""
        return [block.shape[0] for block in self.C_blocks]

    @property
    def constraint_count(self) -> int:
        """Number of affine constraints in the numerical SDP."""
        return len(self.b)

    @property
    def scalar_count(self) -> int:
        """Number of additional nonnegative scalar decision variables."""
        return 0 if self.B is None else self.B.shape[1]


@dataclass
class SDPResult:
    """Numerical semidefinite-solver result in the original problem scaling."""

    status: str
    X_blocks: list[np.ndarray]
    y: np.ndarray
    S_blocks: list[np.ndarray]
    z: np.ndarray | None
    t: np.ndarray | None
    primal_objective: float
    dual_objective: float
    primal_residual_norm: float
    dual_residual_norm: float
    gap: float
    iterations: int
    message: str
    history: list[dict[str, float]] = field(default_factory=list)
    diagnostics: list[SolverDiagnostic] = field(default_factory=list)


@dataclass
class SDPWorkspace:
    """Cached block tensors and numerical indexing for one SDP solve."""

    problem: SDPProblem
    original_problem: SDPProblem
    objective_scale: float
    row_scale: np.ndarray
    constraint_op: np.ndarray
    objective_vec: np.ndarray
    block_tensors: list[np.ndarray]
    dual_matrix: np.ndarray
    dual_rhs: np.ndarray
    dual_cholesky: np.ndarray | None


def svec(matrix: np.ndarray) -> np.ndarray:
    """Vectorize a symmetric matrix with square-root-two scaling on off-diagonal entries."""
    matrix = np.asarray(matrix, dtype=float)
    size = matrix.shape[0]
    values = []
    for col in range(size):
        for row in range(col + 1):
            value = matrix[row, col]
            if row != col:
                value *= np.sqrt(2.0)
            values.append(value)
    return np.asarray(values, dtype=float)


def smat(values: np.ndarray, size: int) -> np.ndarray:
    """Invert symmetric vectorization with the matching off-diagonal scaling."""
    values = np.asarray(values, dtype=float).reshape(-1)
    matrix = np.zeros((size, size), dtype=float)
    index = 0
    for col in range(size):
        for row in range(col + 1):
            value = values[index]
            if row != col:
                value /= np.sqrt(2.0)
            matrix[row, col] = value
            matrix[col, row] = value
            index += 1
    return matrix


def fro_inner(first: np.ndarray, second: np.ndarray) -> float:
    """Compute the Frobenius inner product of the supplied numerical matrices."""
    return float(np.sum(first * second))


def _symmetrize(matrix):
    return 0.5 * (matrix + matrix.T)


def _project_psd(matrix, eps=1e-10):
    matrix = _symmetrize(matrix)
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, eps)
    projected = (vectors * values) @ vectors.T
    return _symmetrize(projected)


def _problem_scale(problem: SDPProblem):
    row_scale = np.ones(problem.constraint_count, dtype=float)
    for index, row in enumerate(problem.A_blocks):
        norm = np.sqrt(sum(np.linalg.norm(block, ord="fro") ** 2 for block in row))
        if problem.B is not None:
            norm += np.linalg.norm(problem.B[index, :])
        norm += abs(problem.b[index])
        row_scale[index] = 1.0 / max(1.0, norm)

    objective_scale = max(
        1.0,
        np.sqrt(
            sum(np.linalg.norm(block, ord="fro") ** 2 for block in problem.C_blocks)
        ),
        0.0 if problem.c is None else np.linalg.norm(problem.c),
    )
    scaled = SDPProblem(
        C_blocks=[block / objective_scale for block in problem.C_blocks],
        A_blocks=[
            [row_scale[index] * block for block in row]
            for index, row in enumerate(problem.A_blocks)
        ],
        b=row_scale * problem.b,
        B=(None if problem.B is None else row_scale[:, None] * problem.B),
        c=(None if problem.c is None else problem.c / objective_scale),
        name=f"{problem.name}_scaled",
    )
    return scaled, objective_scale, row_scale


def _build_workspace(problem: SDPProblem, reg: float = 1e-8) -> SDPWorkspace:
    scaled, objective_scale, row_scale = _problem_scale(problem)

    block_tensors = [
        np.stack(
            [scaled.A_blocks[row][block] for row in range(scaled.constraint_count)],
            axis=0,
        )
        for block in range(scaled.num_blocks)
    ]
    constraint_op = np.concatenate(
        [tensor.reshape(scaled.constraint_count, -1) for tensor in block_tensors],
        axis=1,
    )
    objective_vec = np.concatenate([block.reshape(-1) for block in scaled.C_blocks])

    dual_matrix = constraint_op @ constraint_op.T
    dual_rhs = constraint_op @ objective_vec
    if scaled.B is not None and scaled.c is not None:
        dual_matrix = dual_matrix + scaled.B @ scaled.B.T
        dual_rhs = dual_rhs + scaled.B @ scaled.c

    dual_matrix = _symmetrize(dual_matrix)
    dual_matrix += reg * np.eye(scaled.constraint_count)
    try:
        factor = np.linalg.cholesky(dual_matrix)
    except np.linalg.LinAlgError:
        factor = None

    return SDPWorkspace(
        problem=scaled,
        original_problem=problem,
        objective_scale=objective_scale,
        row_scale=row_scale,
        constraint_op=constraint_op,
        objective_vec=objective_vec,
        block_tensors=block_tensors,
        dual_matrix=dual_matrix,
        dual_rhs=dual_rhs,
        dual_cholesky=factor,
    )


def _stack_blocks(blocks: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([block.reshape(-1) for block in blocks])


def _primal_residual(
    workspace: SDPWorkspace,
    blocks: list[np.ndarray],
    scalars: np.ndarray | None,
) -> np.ndarray:
    residual = workspace.constraint_op @ _stack_blocks(blocks)
    problem = workspace.problem
    if problem.B is not None and scalars is not None:
        residual = residual + problem.B @ scalars
    return residual - problem.b


def _primal_objective(
    workspace: SDPWorkspace,
    blocks: list[np.ndarray],
    scalars: np.ndarray | None,
) -> float:
    value = workspace.objective_vec @ _stack_blocks(blocks)
    problem = workspace.problem
    if problem.c is not None and scalars is not None:
        value += float(problem.c @ scalars)
    return float(value)


def _solve_dual_system(workspace: SDPWorkspace) -> np.ndarray:
    if workspace.dual_cholesky is not None:
        factor = workspace.dual_cholesky
        temp = np.linalg.solve(factor, workspace.dual_rhs)
        return np.linalg.solve(factor.T, temp)
    return np.linalg.lstsq(
        workspace.dual_matrix,
        workspace.dual_rhs,
        rcond=None,
    )[0]


def _estimate_dual(
    workspace: SDPWorkspace,
):
    problem = workspace.problem
    dual = _solve_dual_system(workspace)
    slack_blocks = []
    for index, objective in enumerate(problem.C_blocks):
        combination = np.tensordot(
            dual,
            workspace.block_tensors[index],
            axes=(0, 0),
        )
        slack_blocks.append(_project_psd(objective - combination, 0.0))

    scalar_slack = None
    if problem.scalar_count > 0 and problem.B is not None and problem.c is not None:
        scalar_slack = np.maximum(
            problem.c - problem.B.T @ dual,
            0.0,
        )
    return dual, slack_blocks, scalar_slack


def _dual_residuals(
    workspace: SDPWorkspace,
    dual: np.ndarray,
    slack_blocks: list[np.ndarray],
    scalar_slack: np.ndarray | None,
):
    problem = workspace.problem
    residuals = []
    for index, objective in enumerate(problem.C_blocks):
        combination = np.tensordot(
            dual,
            workspace.block_tensors[index],
            axes=(0, 0),
        )
        residuals.append(objective - slack_blocks[index] - combination)

    scalar_residual = np.zeros(problem.scalar_count, dtype=float)
    if problem.scalar_count > 0:
        scalar_residual = problem.c - problem.B.T @ dual - scalar_slack
    return residuals, scalar_residual


def _initial_params(problem: SDPProblem):
    values = []
    for size in problem.block_sizes:
        for col in range(size):
            for row in range(col, size):
                values.append(1.0 if row == col else 0.0)
    if problem.scalar_count > 0:
        values.extend([1.0] * problem.scalar_count)
    return np.asarray(values, dtype=float)


def _pack_warm_start(
    problem: SDPProblem,
    warm_start: dict | None,
):
    if not warm_start:
        return _initial_params(problem)

    blocks = warm_start.get("X_blocks")
    if blocks is None:
        return _initial_params(problem)

    values = []
    for block, size in zip(blocks, problem.block_sizes):
        block = _project_psd(np.asarray(block, dtype=float), 1e-8)
        try:
            factor = np.linalg.cholesky(block + 1e-8 * np.eye(size))
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(block)
            eigvals = np.maximum(eigvals, 1e-8)
            factor = eigvecs @ np.diag(np.sqrt(eigvals))

        for col in range(size):
            for row in range(col, size):
                values.append(float(factor[row, col]))

    if problem.scalar_count > 0:
        scalars = warm_start.get(
            "z",
            np.ones(problem.scalar_count),
        )
        values.extend(
            np.sqrt(
                np.maximum(
                    np.asarray(scalars, dtype=float),
                    1e-12,
                )
            )
        )
    return np.asarray(values, dtype=float)


def _unpack_factors(
    problem: SDPProblem,
    theta: np.ndarray,
):
    theta = np.asarray(theta, dtype=float)
    offset = 0
    factors = []
    for size in problem.block_sizes:
        count = size * (size + 1) // 2
        values = theta[offset : offset + count]
        offset += count
        factor = np.zeros((size, size), dtype=float)
        # theta uses column-major lower-triangular storage.
        index = 0
        for col in range(size):
            for row in range(col, size):
                factor[row, col] = values[index]
                index += 1
        factors.append(factor)
    scalar_roots = None
    if problem.scalar_count > 0:
        scalar_roots = theta[offset : offset + problem.scalar_count]
    return factors, scalar_roots


def _unpack_params(
    problem: SDPProblem,
    theta: np.ndarray,
):
    factors, scalar_roots = _unpack_factors(problem, theta)
    blocks = [factor @ factor.T for factor in factors]
    scalars = None if scalar_roots is None else scalar_roots * scalar_roots
    return blocks, scalars


def _penalty_value_gradient(
    workspace: SDPWorkspace,
    params: np.ndarray,
    penalty: float,
    reg: float,
):
    """Analytic objective/gradient for the Cholesky-factor penalty model."""
    problem = workspace.problem
    params = np.asarray(params, dtype=float)
    factors, scalar_roots = _unpack_factors(problem, params)
    blocks = [factor @ factor.T for factor in factors]
    scalars = None if scalar_roots is None else scalar_roots * scalar_roots
    residual = _primal_residual(workspace, blocks, scalars)
    objective = _primal_objective(workspace, blocks, scalars)
    value = (
        objective + penalty * float(residual @ residual) + reg * float(params @ params)
    )

    gradient_parts = []
    for factor, objective_block, tensor in zip(
        factors, problem.C_blocks, workspace.block_tensors
    ):
        matrix_gradient = objective_block + 2.0 * penalty * np.tensordot(
            residual, tensor, axes=(0, 0)
        )
        factor_gradient = 2.0 * _symmetrize(matrix_gradient) @ factor
        values = []
        for col in range(factor.shape[0]):
            for row in range(col, factor.shape[0]):
                values.append(factor_gradient[row, col])
        gradient_parts.append(np.asarray(values, dtype=float))

    if scalar_roots is not None:
        scalar_gradient = (
            problem.c.copy() if problem.c is not None else np.zeros_like(scalar_roots)
        )
        if problem.B is not None:
            scalar_gradient += 2.0 * penalty * (problem.B.T @ residual)
        gradient_parts.append(2.0 * scalar_roots * scalar_gradient)

    gradient = (
        np.concatenate(gradient_parts) if gradient_parts else np.zeros_like(params)
    )
    gradient += 2.0 * reg * params
    return float(value), gradient


def _solve_penalty_once(
    workspace: SDPWorkspace,
    max_outer: int = 8,
    max_iter: int = 100,
    tol: float = 1e-7,
    reg: float = 1e-8,
    verbose: bool = False,
    warm_start: dict | None = None,
):
    """Solve one penalty sequence from a prepared SDP workspace."""
    problem = workspace.problem
    theta = _pack_warm_start(problem, warm_start)
    penalties = [
        1.0,
        10.0,
        100.0,
        1000.0,
        10000.0,
        100000.0,
        1e6,
        1e7,
    ][:max_outer]
    history = []
    best = None
    rng = np.random.default_rng(0)

    for outer, penalty in enumerate(penalties):

        def penalty_objective(params, _penalty=penalty):
            return _penalty_value_gradient(workspace, params, _penalty, reg)

        starts = [theta]
        if outer == 0:
            starts.extend(
                theta + 0.1 * rng.normal(size=theta.shape[0]) for _ in range(2)
            )

        best_result = None
        best_value = float("inf")
        for start in starts:
            result = minimize(
                penalty_objective,
                start,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": max_iter, "ftol": max(tol * 0.1, 1e-14)},
            )
            if float(result.fun) < best_value:
                best_value = float(result.fun)
                best_result = result

        theta = np.asarray(best_result.x, dtype=float)
        blocks, scalars = _unpack_params(problem, theta)
        residual = _primal_residual(
            workspace,
            blocks,
            scalars,
        )
        objective = _primal_objective(
            workspace,
            blocks,
            scalars,
        )
        residual_norm = float(np.linalg.norm(residual))
        history.append(
            {
                "outer": float(outer),
                "penalty": penalty,
                "primal_residual": residual_norm,
                "objective_scaled": objective,
            }
        )
        if verbose:
            print(
                "outer",
                outer,
                "penalty",
                penalty,
                "residual",
                residual_norm,
                "objective",
                objective,
            )

        best = blocks, scalars, residual
        if residual_norm <= tol:
            break

    blocks, scalars, residual = best
    dual, slack_blocks, scalar_slack = _estimate_dual(workspace)
    dual_blocks, scalar_residual = _dual_residuals(
        workspace,
        dual,
        slack_blocks,
        scalar_slack,
    )

    scale = workspace.objective_scale
    original = workspace.original_problem
    dual_original = scale * workspace.row_scale * dual
    slack_original = [scale * block for block in slack_blocks]
    scalar_slack_original = None if scalar_slack is None else scale * scalar_slack

    # The workspace residual is row-scaled; undo that scaling vectorially.
    primal_vec = residual / workspace.row_scale
    primal_norm = float(np.linalg.norm(primal_vec))

    # Scaled dual residuals map back by the single objective scale factor.
    dual_residuals = [scale * block for block in dual_blocks]
    dual_norm = max(
        (float(np.linalg.norm(block, ord="fro")) for block in dual_residuals),
        default=0.0,
    )
    if original.scalar_count > 0:
        scalar_residual_original = scale * scalar_residual
        dual_norm = max(dual_norm, float(np.linalg.norm(scalar_residual_original)))

    primal_obj = sum(
        fro_inner(objective, block)
        for objective, block in zip(original.C_blocks, blocks)
    )
    if original.c is not None and scalars is not None:
        primal_obj += float(original.c @ scalars)
    dual_obj = float(original.b @ dual_original)
    gap = abs(primal_obj - dual_obj)
    status = (
        "optimal"
        if primal_norm <= tol and dual_norm <= max(10 * tol, 1e-7)
        else "approximate"
        if primal_norm <= max(50 * tol, 5e-3)
        else "failed"
    )

    return SDPResult(
        status,
        blocks,
        dual_original,
        slack_original,
        scalars,
        scalar_slack_original,
        float(primal_obj),
        float(dual_obj),
        primal_norm,
        dual_norm,
        float(gap),
        len(history),
        "Solved with factorized penalty method.",
        history,
    )


def _sparsity_components(problem: SDPProblem, tol: float = 1e-12):
    if problem.num_blocks != 1:
        return None
    size = problem.C_blocks[0].shape[0]
    pattern = np.abs(problem.C_blocks[0]) > tol
    for row in problem.A_blocks:
        pattern |= np.abs(row[0]) > tol
    pattern |= pattern.T

    unseen = set(range(size))
    components = []
    while unseen:
        seed = unseen.pop()
        stack = [seed]
        component = {seed}
        while stack:
            current = stack.pop()
            neighbors = set(np.flatnonzero(pattern[current]))
            for neighbor in neighbors:
                if neighbor not in component:
                    component.add(neighbor)
                    unseen.discard(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda comp: comp[0])


def _sparsity_pattern(problem: SDPProblem, tol: float = 1e-12) -> np.ndarray:
    if problem.num_blocks != 1:
        raise ValueError("Chordal decomposition currently accepts one PSD block.")
    pattern = np.abs(problem.C_blocks[0]) > tol
    for row in problem.A_blocks:
        pattern |= np.abs(row[0]) > tol
    pattern |= pattern.T
    np.fill_diagonal(pattern, True)
    return pattern


def _maximal_chordal_cliques(problem: SDPProblem, tol: float = 1e-12):
    pattern = _sparsity_pattern(problem, tol)
    n = pattern.shape[0]
    edges = {(i, j) for i in range(n) for j in range(i + 1, n) if pattern[i, j]}
    decomposition = chordal_extension(IndexedSparsityGraph(n, edges))
    # The low-level heuristic can emit an early clique later subsumed by a
    # larger one; clean that up deterministically.
    raw = [tuple(sorted(clique)) for clique in decomposition.cliques]
    maximal = []
    for clique in sorted(set(raw), key=lambda c: (-len(c), c)):
        if not any(set(clique).issubset(set(other)) for other in maximal):
            maximal.append(clique)
    maximal.sort(key=lambda c: (c[0], len(c), c))
    return maximal


def _allocate_clique_entries(matrix: np.ndarray, cliques, tol: float = 0.0):
    """Allocate every symmetric coefficient entry to one containing clique."""
    n = matrix.shape[0]
    locals_ = [np.zeros((len(clique), len(clique)), dtype=float) for clique in cliques]
    positions = [
        {global_i: local_i for local_i, global_i in enumerate(clique)}
        for clique in cliques
    ]
    for i in range(n):
        for j in range(i, n):
            value = float(matrix[i, j])
            if abs(value) <= tol:
                continue
            owner = next(
                (
                    k
                    for k, clique in enumerate(cliques)
                    if i in positions[k] and j in positions[k]
                ),
                None,
            )
            if owner is None:
                raise ValueError(f"No chordal clique covers matrix entry ({i}, {j}).")
            li = positions[owner][i]
            lj = positions[owner][j]
            locals_[owner][li, lj] = value
            locals_[owner][lj, li] = value
    return locals_


def _separator_equalities(cliques):
    """Rows enforcing equality of all duplicated clique entries on overlaps."""
    occurrences = {}
    for block_index, clique in enumerate(cliques):
        for local_i, global_i in enumerate(clique):
            for local_j in range(local_i, len(clique)):
                global_j = clique[local_j]
                key = (min(global_i, global_j), max(global_i, global_j))
                occurrences.setdefault(key, []).append((block_index, local_i, local_j))
    rows = []
    keys = []
    for key, copies in sorted(occurrences.items()):
        if len(copies) <= 1:
            continue
        base = copies[0]
        for copy in copies[1:]:
            mats = [
                np.zeros((len(clique), len(clique)), dtype=float) for clique in cliques
            ]
            for sign, (block, i, j) in ((1.0, copy), (-1.0, base)):
                coefficient = sign if i == j else 0.5 * sign
                mats[block][i, j] += coefficient
                if i != j:
                    mats[block][j, i] += coefficient
            rows.append(mats)
            keys.append(key)
    return rows, keys


def chordalize_sdp_block(
    problem: SDPProblem,
    tol: float = 1e-12,
) -> SDPProblem:
    """Decompose a sparse PSD block into overlapping chordal clique blocks.

    Data coefficients are allocated exactly once among maximal cliques of a
    chordal extension.  Entries duplicated on clique overlaps receive explicit
    equality rows, so the transformed block problem is an exact sparse-PSD
    completion formulation rather than a heuristic block split.
    """
    if problem.num_blocks != 1:
        return problem
    cliques = _maximal_chordal_cliques(problem, tol)
    n = problem.C_blocks[0].shape[0]
    if len(cliques) <= 1 and cliques and len(cliques[0]) == n:
        return problem

    C_blocks = _allocate_clique_entries(problem.C_blocks[0], cliques, tol=tol)
    A_rows = [
        _allocate_clique_entries(row[0], cliques, tol=tol) for row in problem.A_blocks
    ]
    consistency_rows, overlap_keys = _separator_equalities(cliques)
    A_rows.extend(consistency_rows)
    b = np.concatenate([problem.b, np.zeros(len(consistency_rows), dtype=float)])
    if problem.B is not None:
        B = np.vstack(
            [
                problem.B,
                np.zeros((len(consistency_rows), problem.B.shape[1]), dtype=float),
            ]
        )
    else:
        B = None
    metadata = dict(problem.metadata)
    metadata.update(
        {
            "chordal_original_size": n,
            "chordal_cliques": [list(clique) for clique in cliques],
            "chordal_overlap_keys": [list(key) for key in overlap_keys],
            "chordal_original_constraint_count": problem.constraint_count,
        }
    )
    return SDPProblem(
        C_blocks=C_blocks,
        A_blocks=A_rows,
        b=b,
        B=B,
        c=None if problem.c is None else problem.c.copy(),
        name=f"{problem.name}_chordal_cliques",
        metadata=metadata,
    )


def _rebuild_chordal_matrix(cliques, blocks, size):
    values = np.zeros((size, size), dtype=float)
    counts = np.zeros((size, size), dtype=float)
    for clique, block in zip(cliques, blocks):
        for local_i, global_i in enumerate(clique):
            for local_j, global_j in enumerate(clique):
                values[global_i, global_j] += block[local_i, local_j]
                counts[global_i, global_j] += 1.0
    mask = counts > 0
    values[mask] /= counts[mask]
    return _symmetrize(values)


def _rebuild_chordal_result(
    original: SDPProblem, transformed: SDPProblem, result: SDPResult
) -> SDPResult:
    metadata = transformed.metadata
    cliques = [list(map(int, clique)) for clique in metadata.get("chordal_cliques", [])]
    if not cliques:
        return result
    size = int(metadata["chordal_original_size"])
    original_rows = int(metadata["chordal_original_constraint_count"])
    primal = _rebuild_chordal_matrix(cliques, result.X_blocks, size)
    y = np.asarray(result.y[:original_rows], dtype=float)
    slack = original.C_blocks[0].copy()
    for multiplier, row in zip(y, original.A_blocks):
        slack -= multiplier * row[0]
    primal_residuals = []
    for row, rhs in zip(original.A_blocks, original.b):
        primal_residuals.append(fro_inner(row[0], primal) - rhs)
    if original.B is not None and result.z is not None:
        primal_residuals = np.asarray(primal_residuals) + original.B @ result.z
    primal_norm = float(np.linalg.norm(primal_residuals))
    primal_obj = fro_inner(original.C_blocks[0], primal)
    if original.c is not None and result.z is not None:
        primal_obj += float(original.c @ result.z)
    dual_obj = float(original.b @ y)
    return SDPResult(
        result.status,
        [primal],
        y,
        [slack],
        result.z,
        result.t,
        float(primal_obj),
        dual_obj,
        primal_norm,
        result.dual_residual_norm,
        abs(float(primal_obj) - dual_obj),
        result.iterations,
        result.message + " Used overlapping chordal PSD-completion decomposition.",
        result.history,
        result.diagnostics,
    )


def _warm_start_options(
    problem: SDPProblem,
    warm_start: dict | None,
):
    starts = [warm_start]
    if warm_start is None:
        starts.append(None)
    if problem.num_blocks >= 1:
        starts.append(
            {
                "X_blocks": [np.eye(size) for size in problem.block_sizes],
                "z": (
                    np.ones(problem.scalar_count) if problem.scalar_count > 0 else None
                ),
            }
        )
    return starts


def _best_result(
    result: SDPResult,
    best: SDPResult | None,
):
    if best is None:
        return result
    result_score = (
        result.primal_residual_norm,
        result.gap,
    )
    best_score = (
        best.primal_residual_norm,
        best.gap,
    )
    return result if result_score < best_score else best


def _solve_penalty_sdp(
    problem: SDPProblem,
    max_outer: int = 8,
    max_iter: int = 100,
    tol: float = 1e-7,
    reg: float = 1e-8,
    verbose: bool = False,
    warm_start: dict | None = None,
):
    workspace = _build_workspace(problem, reg=reg)
    best = None
    diagnostics = []

    for index, start in enumerate(_warm_start_options(workspace.problem, warm_start)):
        try:
            result = _solve_penalty_once(
                workspace,
                max_outer=max_outer,
                max_iter=max_iter,
                tol=tol,
                reg=reg,
                verbose=verbose,
                warm_start=start,
            )
            best = _best_result(result, best)
            if (
                result.status in {"optimal", "approximate"}
                and result.primal_residual_norm < 1e-6
            ):
                break
        except (
            ValueError,
            RuntimeError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as exc:
            diagnostics.append(
                SolverDiagnostic.from_exception(
                    "sdp",
                    f"restart_{index}",
                    DiagnosticKind.NUMERICAL_FAILURE,
                    exc,
                )
            )

    if best is not None:
        best.diagnostics.extend(diagnostics)
        return best
    return _solve_penalty_once(
        workspace,
        max_outer=max_outer,
        max_iter=max_iter,
        tol=tol,
        reg=reg,
        verbose=verbose,
        warm_start=warm_start,
    )


def solve_sdp_sparse_chordal(
    problem: SDPProblem,
    max_iter: int = 100,
    tol: float = 1e-7,
    reg: float = 1e-8,
    verbose: bool = False,
):
    """Solve an SDP after chordally decomposing a sparse single PSD block.

    Overlapping clique blocks are linked by separator equalities, and the
    resulting sparse solution is reconstructed in the coordinates of the
    original semidefinite program before it is returned.
    """
    sparse_problem = chordalize_sdp_block(problem)
    if sparse_problem is problem:
        return _solve_penalty_sdp(
            problem,
            max_outer=max(1, min(8, max_iter // 20 or 1)),
            max_iter=max_iter,
            tol=tol,
            reg=reg,
            verbose=verbose,
            warm_start=None,
        )
    result = _solve_penalty_sdp(
        sparse_problem,
        max_outer=max(1, min(8, max_iter // 20 or 1)),
        max_iter=max_iter,
        tol=tol,
        reg=reg,
        verbose=verbose,
        warm_start=None,
    )
    return _rebuild_chordal_result(problem, sparse_problem, result)


def solve_sdp_dense(
    problem: SDPProblem,
    max_iter: int = 100,
    tol: float = 1e-7,
    reg: float = 1e-8,
    verbose: bool = False,
    warm_start: dict | None = None,
    sparse_chordal: bool = False,
) -> SDPResult:
    """Solve an SDP with the package's dense penalty backend.

    When requested for a sufficiently large single sparse block, the routine
    delegates to chordal decomposition; otherwise it solves the original block
    representation directly and can reuse a supplied warm start.
    """
    if sparse_chordal and problem.num_blocks == 1 and problem.C_blocks[0].shape[0] >= 4:
        return solve_sdp_sparse_chordal(
            problem,
            max_iter=max_iter,
            tol=tol,
            reg=reg,
            verbose=verbose,
        )
    return _solve_penalty_sdp(
        problem,
        max_outer=max(1, min(8, max_iter // 20 or 1)),
        max_iter=max_iter,
        tol=tol,
        reg=reg,
        verbose=verbose,
        warm_start=warm_start,
    )


def cvxpy_available() -> bool:
    """Return whether the optional CVXPY SDP adapter is importable."""
    return find_spec("cvxpy") is not None


def available_external_sdp_solvers() -> list[str]:
    """Return installed CVXPY solvers capable of accepting conic problems."""
    try:
        import cvxpy as cp
    except ImportError:
        return []
    preferred = ("MOSEK", "CLARABEL", "CVXOPT", "SCS")
    installed = set(cp.installed_solvers())
    return [name for name in preferred if name in installed]


def solve_sdp_cvxpy(
    problem: SDPProblem,
    *,
    solver: str | None = None,
    max_iter: int = 10000,
    tol: float = 1e-7,
    verbose: bool = False,
) -> SDPResult:
    """Solve ``problem`` through an optional mature CVXPY conic backend.

    The returned matrices remain numerical candidates.  SOS callers must still
    perform their existing exact affine reconstruction, polynomial-identity
    check, and exact PSD verification before exposing a certificate.
    """
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError("CVXPY is required for the external SDP backend") from exc

    choices = available_external_sdp_solvers()
    chosen = solver.upper() if solver else (choices[0] if choices else None)
    if chosen is None or chosen not in set(cp.installed_solvers()):
        raise RuntimeError("No supported CVXPY conic solver is installed")

    x_vars = [cp.Variable(block.shape, symmetric=True) for block in problem.C_blocks]
    psd_constraints = [variable >> 0 for variable in x_vars]
    constraints = list(psd_constraints)
    t_var = cp.Variable(problem.scalar_count) if problem.scalar_count else None
    scalar_constraint = None
    if t_var is not None:
        scalar_constraint = t_var >= 0
        constraints.append(scalar_constraint)

    equalities = []
    for row_index, row in enumerate(problem.A_blocks):
        expression = 0
        for matrix, variable in zip(row, x_vars):
            expression += cp.sum(cp.multiply(matrix, variable))
        if problem.B is not None and t_var is not None:
            expression += problem.B[row_index, :] @ t_var
        equality = expression == problem.b[row_index]
        constraints.append(equality)
        equalities.append(equality)

    objective_expr = 0
    for matrix, variable in zip(problem.C_blocks, x_vars):
        objective_expr += cp.sum(cp.multiply(matrix, variable))
    if problem.c is not None and t_var is not None:
        objective_expr += problem.c @ t_var
    model = cp.Problem(cp.Minimize(objective_expr), constraints)

    kwargs = {"verbose": verbose}
    if chosen == "SCS":
        kwargs.update({"max_iters": int(max_iter), "eps": float(tol)})
    elif chosen == "CLARABEL":
        kwargs.update(
            {
                "max_iter": int(max_iter),
                "tol_gap_abs": float(tol),
                "tol_feas": float(tol),
            }
        )
    model.solve(solver=chosen, **kwargs)

    if model.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"CVXPY/{chosen} returned status {model.status}")
    if any(variable.value is None for variable in x_vars):
        raise RuntimeError("CVXPY did not return primal PSD blocks")

    blocks = [
        _symmetrize(np.asarray(variable.value, dtype=float)) for variable in x_vars
    ]
    scalars = (
        None if t_var is None else np.asarray(t_var.value, dtype=float).reshape(-1)
    )
    y = np.asarray(
        [constraint.dual_value for constraint in equalities], dtype=float
    ).reshape(-1)
    slacks = [
        _symmetrize(np.asarray(constraint.dual_value, dtype=float))
        for constraint in psd_constraints
    ]
    z = None
    if scalar_constraint is not None and scalar_constraint.dual_value is not None:
        z = np.asarray(scalar_constraint.dual_value, dtype=float).reshape(-1)

    primal_residual = []
    for row_index, row in enumerate(problem.A_blocks):
        value = sum(fro_inner(matrix, block) for matrix, block in zip(row, blocks))
        if problem.B is not None and scalars is not None:
            value += float(problem.B[row_index, :] @ scalars)
        primal_residual.append(value - problem.b[row_index])
    primal_residual_norm = float(np.linalg.norm(primal_residual))

    dual_parts = []
    for block_index, objective in enumerate(problem.C_blocks):
        stationarity = objective.copy()
        for row_index in range(problem.constraint_count):
            stationarity += y[row_index] * problem.A_blocks[row_index][block_index]
        stationarity -= slacks[block_index]
        dual_parts.append(np.linalg.norm(stationarity, ord="fro"))
    if problem.c is not None and problem.B is not None and z is not None:
        dual_parts.append(np.linalg.norm(problem.c + problem.B.T @ y - z))
    dual_residual_norm = float(np.linalg.norm(dual_parts))

    primal_objective = float(model.value)
    dual_objective = float(-problem.b @ y)
    gap = abs(primal_objective - dual_objective)
    iterations = int(getattr(model.solver_stats, "num_iters", 0) or 0)
    return SDPResult(
        status="optimal" if model.status == cp.OPTIMAL else "optimal_inaccurate",
        X_blocks=blocks,
        y=y,
        S_blocks=slacks,
        z=z,
        t=scalars,
        primal_objective=primal_objective,
        dual_objective=dual_objective,
        primal_residual_norm=primal_residual_norm,
        dual_residual_norm=dual_residual_norm,
        gap=gap,
        iterations=iterations,
        message=f"Solved by CVXPY/{chosen}; numerical candidate requires exact post-verification.",
        diagnostics=[],
    )
