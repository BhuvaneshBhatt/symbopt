from __future__ import annotations

from functools import lru_cache

import sympy as sp


@lru_cache(maxsize=8192)
def eval_complex(expr: sp.Expr, digits: int = 50) -> complex:
    return complex(sp.N(expr, digits))


def eval_float(expr: sp.Expr, digits: int = 50) -> float:
    value = eval_complex(expr, digits)
    if abs(value.imag) > 10 ** (-(digits // 3)):
        raise ValueError("Expression is not numerically real.")
    return float(value.real)


def real_part(expr: sp.Expr, digits: int = 50) -> float:
    return float(eval_complex(expr, digits).real)


def is_numerically_real(
    expr: sp.Expr,
    tol: float = 1e-9,
    digits: int = 50,
) -> bool:
    return abs(eval_complex(expr, digits).imag) <= tol
