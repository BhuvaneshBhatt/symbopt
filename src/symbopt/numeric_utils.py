from __future__ import annotations

import math
from functools import lru_cache

import sympy as sp


@lru_cache(maxsize=8192)
def eval_complex(expr: sp.Expr, digits: int = 50) -> complex:
    """Evaluate an expression numerically and return a Python complex value."""
    return complex(sp.N(expr, digits))


def eval_float(expr: sp.Expr, digits: int = 50) -> float:
    """Evaluate a finite real float, rejecting complex or nonfinite numerical values."""
    value = eval_complex(expr, digits)
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise ValueError("Expression has no finite numerical value.")
    if abs(value.imag) > 10 ** (-(digits // 3)):
        raise ValueError("Expression is not numerically real.")
    return float(value.real)


def real_part(expr: sp.Expr, digits: int = 50) -> float:
    """Return the numerical real part without asserting that the imaginary part vanishes."""
    return float(eval_complex(expr, digits).real)


def is_numerically_real(
    expr: sp.Expr,
    tol: float = 1e-9,
    digits: int = 50,
) -> bool:
    """Whether a finite numerical value has imaginary part within tolerance."""
    value = eval_complex(expr, digits)
    return (
        math.isfinite(value.real)
        and math.isfinite(value.imag)
        and abs(value.imag) <= tol
    )
