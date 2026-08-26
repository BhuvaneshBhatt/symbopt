from __future__ import annotations

from typing import Mapping, Tuple

import sympy as sp
from sympy.core.relational import (
    Equality,
    GreaterThan,
    LessThan,
    Relational,
    StrictGreaterThan,
    StrictLessThan,
)

from .numeric_utils import real_part


def normalize_relational(rel: Relational) -> Tuple[str, sp.Expr]:
    """Return a relation kind and a residual normalized around zero."""
    residual = sp.expand(rel.lhs - rel.rhs)
    if isinstance(rel, Equality):
        return "eq", residual
    if isinstance(rel, GreaterThan):
        return "ge", residual
    if isinstance(rel, StrictGreaterThan):
        return "gt", residual
    if isinstance(rel, LessThan):
        return "ge", -residual
    if isinstance(rel, StrictLessThan):
        return "gt", -residual
    if isinstance(rel, sp.Unequality):
        return "neq", residual
    raise TypeError(f"Unsupported relational type: {type(rel)}")


def relation_holds(
    rel: Relational,
    values: Mapping[sp.Symbol, object],
    tol: float = 1e-7,
) -> bool:
    kind, residual = normalize_relational(rel)
    try:
        value = real_part(residual.subs(values))
    except (TypeError, ValueError, ArithmeticError, NotImplementedError):
        return False

    if kind == "eq":
        return abs(value) <= tol
    if kind == "ge":
        return value >= -tol
    if kind == "gt":
        return value > tol
    if kind == "neq":
        return abs(value) > tol
    return False
