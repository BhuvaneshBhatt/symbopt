from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from .array_validation import (
    checked_bounds as _checked_bounds,
)
from .array_validation import (
    finite_array as _finite_array,
)
from .array_validation import (
    matrix_rhs as _matrix_rhs,
)


@dataclass
class QPResult:
    """Numerical quadratic-program status, candidate point, objective and iteration diagnostics."""

    status: str
    x: np.ndarray
    objective: float
    iterations: int
    message: str


def solve_convex_qp(
    Q,
    c,
    A_ub=None,
    b_ub=None,
    A_eq=None,
    b_eq=None,
    x0=None,
    bounds: Sequence[tuple[float | None, float | None]] | None = None,
    convexity_tol: float = 1e-10,
):
    """Solve a convex quadratic program with linear constraints.

    The quadratic matrix is symmetrized and checked for positive semidefiniteness.
    Variables are free by default unless explicit bounds are supplied."""
    Q = _finite_array(Q, "Q", ndim=2)
    c = _finite_array(c, "c").reshape(-1)
    size = len(c)
    if size == 0:
        raise ValueError("c must contain at least one objective coefficient.")

    if Q.shape != (size, size):
        raise ValueError("Q must be square with dimension len(c).")

    if not np.isfinite(convexity_tol) or convexity_tol < 0:
        raise ValueError("convexity_tol must be finite and nonnegative.")

    Q = 0.5 * (Q + Q.T)
    eig_min = float(np.linalg.eigvalsh(Q).min()) if size else 0.0
    if eig_min < -convexity_tol:
        return QPResult(
            "nonconvex",
            np.zeros(size),
            float("nan"),
            0,
            "Q is not positive semidefinite.",
        )

    if x0 is None:
        x0 = np.zeros(size, dtype=float)
    else:
        x0 = _finite_array(x0, "x0").reshape(-1)
        if len(x0) != size:
            raise ValueError("x0 must have length len(c).")

    A_ub, b_ub = _matrix_rhs(A_ub, b_ub, size, "A_ub", "b_ub")
    A_eq, b_eq = _matrix_rhs(A_eq, b_eq, size, "A_eq", "b_eq")

    constraints = []
    if A_ub is not None:
        for index in range(A_ub.shape[0]):
            row = A_ub[index].copy()
            bound = float(b_ub[index])
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x, row=row, bound=bound: bound - row @ x,
                }
            )

    if A_eq is not None:
        for index in range(A_eq.shape[0]):
            row = A_eq[index].copy()
            bound = float(b_eq[index])
            constraints.append(
                {
                    "type": "eq",
                    "fun": lambda x, row=row, bound=bound: row @ x - bound,
                }
            )

    checked_bounds = _checked_bounds(bounds, size)
    scipy_bounds = [(None, None)] * size if checked_bounds is None else checked_bounds

    def objective_fn(x):
        with np.errstate(over="ignore", invalid="ignore"):
            return 0.5 * float(x @ Q @ x) + float(c @ x)

    def gradient_fn(x):
        with np.errstate(over="ignore", invalid="ignore"):
            return Q @ x + c

    result = scipy_minimize(
        objective_fn,
        x0,
        jac=gradient_fn,
        method="SLSQP",
        bounds=scipy_bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    status = "optimal" if result.success else "failed"
    return QPResult(
        status=status,
        x=np.asarray(result.x, dtype=float),
        objective=float(objective_fn(result.x)),
        iterations=int(getattr(result, "nit", 0)),
        message=str(result.message),
    )
