from __future__ import annotations

from collections.abc import Mapping

import sympy as sp
from sympy.core.relational import (
    Equality,
    GreaterThan,
    LessThan,
    Relational,
    StrictGreaterThan,
    StrictLessThan,
)

from .numeric_utils import eval_float


def normalize_relational(rel: Relational) -> tuple[str, sp.Expr]:
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
    """Screen a finite real residual with the supplied feasibility tolerance."""
    kind, residual = normalize_relational(rel)
    try:
        value = eval_float(residual.subs(values))
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
