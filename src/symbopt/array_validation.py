from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def finite_array(value, name: str, ndim: int | None = None) -> np.ndarray:
    """Return a finite float ndarray with an optional dimensionality check."""
    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def matrix_rhs(matrix, rhs, size: int, matrix_name: str, rhs_name: str):
    """Validate an optional matrix/right-hand-side pair against a variable size."""
    if matrix is None and rhs is None:
        return None, None
    if matrix is None or rhs is None:
        raise ValueError(f"{matrix_name} and {rhs_name} must be supplied together.")
    checked_matrix = finite_array(matrix, matrix_name)
    if checked_matrix.ndim == 1:
        checked_matrix = checked_matrix.reshape(1, -1)
    elif checked_matrix.ndim != 2:
        raise ValueError(f"{matrix_name} must be a one- or two-dimensional array.")
    checked_rhs = finite_array(rhs, rhs_name).reshape(-1)
    if checked_matrix.shape[1] != size:
        raise ValueError(f"{matrix_name} has incompatible column count.")
    if checked_matrix.shape[0] != len(checked_rhs):
        raise ValueError(f"{rhs_name} has incompatible length.")
    return checked_matrix, checked_rhs


def checked_bounds(
    bounds: Optional[Sequence[Tuple[Optional[float], Optional[float]]]],
    size: int,
):
    """Validate SciPy-style scalar bounds."""
    if bounds is None:
        return None
    if size == 1 and len(bounds) == 2 and not isinstance(bounds[0], (tuple, list)):
        bounds = [bounds]
    if len(bounds) != size:
        raise ValueError("bounds must have one entry per variable.")
    checked = []
    for pair in bounds:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("Each bounds entry must be a (lower, upper) pair.")
        lower, upper = pair
        if lower is not None:
            lower = float(lower)
            if not np.isfinite(lower):
                raise ValueError("Finite lower bounds are required when specified.")
        if upper is not None:
            upper = float(upper)
            if not np.isfinite(upper):
                raise ValueError("Finite upper bounds are required when specified.")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("A lower bound cannot exceed its upper bound.")
        checked.append((lower, upper))
    return checked
