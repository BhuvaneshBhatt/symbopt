from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .array_validation import (
    checked_bounds as _checked_bounds,
    finite_array as _finite_array,
    matrix_rhs as _matrix_rhs,
)
from scipy.optimize import linprog


@dataclass
class LPResult:
    status: str
    x: np.ndarray
    objective: float
    basis: List[int]
    iterations: int
    message: str


def _validate_standard_lp(c, A_ub, b_ub):
    c = _finite_array(c, "c").reshape(-1)
    if len(c) == 0:
        raise ValueError("c must contain at least one objective coefficient.")
    A, b = _matrix_rhs(A_ub, b_ub, len(c), "A_ub", "b_ub")
    if A is None:
        raise ValueError("A_ub and b_ub are required.")
    return c, A, b


def simplex(c, A_ub=None, b_ub=None, max_iter: int = 200):
    """Primal tableau simplex for Ax <= b, x >= 0 with a feasible slack basis."""
    c, A, b = _validate_standard_lp(c, A_ub, b_ub)
    if np.any(b < -1e-12):
        return LPResult(
            "needs_phase_one",
            np.zeros(len(c)),
            float("nan"),
            [],
            0,
            "The slack basis is infeasible; use solve_lp for feasibility initialization.",
        )

    rows, cols = A.shape
    tableau = np.zeros((rows + 1, cols + rows + 1))
    tableau[:rows, :cols] = A
    tableau[:rows, cols : cols + rows] = np.eye(rows)
    tableau[:rows, -1] = b
    tableau[-1, :cols] = c
    basis = list(range(cols, cols + rows))

    for iteration in range(1, max_iter + 1):
        entering = int(np.argmin(tableau[-1, :-1]))
        if tableau[-1, entering] >= -1e-10:
            solution = np.zeros(cols + rows)
            for row, basic in enumerate(basis):
                solution[basic] = tableau[row, -1]
            return LPResult(
                "optimal",
                solution[:cols],
                float(c @ solution[:cols]),
                basis,
                iteration,
                "Optimal basis found.",
            )

        ratios = np.array(
            [
                tableau[row, -1] / tableau[row, entering]
                if tableau[row, entering] > 1e-12
                else np.inf
                for row in range(rows)
            ]
        )
        leaving = int(np.argmin(ratios))
        if not np.isfinite(ratios[leaving]):
            return LPResult(
                "unbounded",
                np.zeros(cols),
                np.inf,
                basis,
                iteration,
                "LP is unbounded in the selected entering direction.",
            )

        pivot = tableau[leaving, entering]
        tableau[leaving, :] /= pivot
        for row in range(rows + 1):
            if row != leaving:
                tableau[row, :] -= tableau[row, entering] * tableau[leaving, :]
        basis[leaving] = entering

    solution = np.zeros(cols + rows)
    for row, basic in enumerate(basis):
        solution[basic] = tableau[row, -1]
    return LPResult(
        "max_iter",
        solution[:cols],
        float(c @ solution[:cols]),
        basis,
        max_iter,
        "Maximum iterations reached.",
    )


def revised_simplex(c, A_ub=None, b_ub=None, max_iter: int = 200):
    """Revised simplex for Ax <= b, x >= 0 with a feasible slack basis."""
    c, A, b = _validate_standard_lp(c, A_ub, b_ub)
    if np.any(b < -1e-12):
        return LPResult(
            "needs_phase_one",
            np.zeros(len(c)),
            float("nan"),
            [],
            0,
            "The slack basis is infeasible; use solve_lp for feasibility initialization.",
        )

    rows, cols = A.shape
    augmented = np.hstack([A, np.eye(rows)])
    costs = np.concatenate([c, np.zeros(rows)])
    basis = list(range(cols, cols + rows))

    for iteration in range(1, max_iter + 1):
        basis_matrix = augmented[:, basis]
        try:
            basic_values = np.linalg.solve(basis_matrix, b)
            dual = np.linalg.solve(basis_matrix.T, costs[basis])
        except np.linalg.LinAlgError:
            return LPResult(
                "failed",
                np.zeros(cols),
                float("nan"),
                basis,
                iteration,
                "The current basis is singular.",
            )

        reduced = costs - dual @ augmented
        nonbasic = [index for index in range(cols) if index not in basis]
        entering = min(nonbasic, key=lambda index: reduced[index]) if nonbasic else None

        if entering is None or reduced[entering] >= -1e-10:
            solution = np.zeros(cols + rows)
            solution[np.asarray(basis)] = basic_values
            return LPResult(
                "optimal",
                solution[:cols],
                float(c @ solution[:cols]),
                basis,
                iteration,
                "Optimal basis found.",
            )

        direction = np.linalg.solve(basis_matrix, augmented[:, entering])
        ratios = np.array(
            [
                basic_values[row] / direction[row] if direction[row] > 1e-12 else np.inf
                for row in range(rows)
            ]
        )
        leaving = int(np.argmin(ratios))
        if not np.isfinite(ratios[leaving]):
            return LPResult(
                "unbounded",
                np.zeros(cols),
                np.inf,
                basis,
                iteration,
                "LP is unbounded in the selected entering direction.",
            )
        basis[leaving] = entering

    basis_matrix = augmented[:, basis]
    solution = np.zeros(cols + rows)
    solution[np.asarray(basis)] = np.linalg.solve(basis_matrix, b)
    return LPResult(
        "max_iter",
        solution[:cols],
        float(c @ solution[:cols]),
        basis,
        max_iter,
        "Maximum iterations reached.",
    )


def solve_lp(
    c,
    A_ub=None,
    b_ub=None,
    A_eq=None,
    b_eq=None,
    maximize: bool = False,
    bounds: Optional[Sequence[Tuple[Optional[float], Optional[float]]]] = None,
):
    """Solve a general continuous linear program.

    Supports inequalities, equalities, variable bounds, and free variables. The
    general driver uses SciPy HiGHS for robust feasibility initialization; the
    package also exposes separate educational simplex routines for feasible
    standard form."""
    c = _finite_array(c, "c").reshape(-1)
    size = len(c)
    if size == 0:
        raise ValueError("c must contain at least one objective coefficient.")

    A_ub, b_ub = _matrix_rhs(A_ub, b_ub, size, "A_ub", "b_ub")
    A_eq, b_eq = _matrix_rhs(A_eq, b_eq, size, "A_eq", "b_eq")
    checked_bounds = _checked_bounds(bounds, size)
    scipy_bounds = [(0.0, None)] * size if checked_bounds is None else checked_bounds

    objective = -c if maximize else c
    result = linprog(
        objective,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=scipy_bounds,
        method="highs",
    )

    status_map = {
        0: "optimal",
        1: "max_iter",
        2: "infeasible",
        3: "unbounded",
    }
    status = status_map.get(int(result.status), "failed")
    value = float("nan")
    if result.fun is not None and np.isfinite(result.fun):
        value = float(-result.fun if maximize else result.fun)

    point = (
        np.asarray(result.x, dtype=float) if result.x is not None else np.zeros(size)
    )
    return LPResult(
        status,
        point,
        value,
        [],
        int(getattr(result, "nit", 0)),
        str(result.message),
    )
