from __future__ import annotations
from typing import Dict, Sequence
import sympy as sp


def nullspace_directions(hessian_at_point: sp.Matrix):
    try:
        return hessian_at_point.nullspace()
    except (TypeError, ValueError, ArithmeticError, NotImplementedError):
        return []


def directional_taylor_test(
    f: sp.Expr,
    vars: Sequence[sp.Symbol],
    point: Dict[sp.Symbol, sp.Expr],
    direction: sp.Matrix,
    max_order: int = 6,
):
    t = sp.Symbol("_t", real=True)
    subs = {v: sp.expand(point[v] + t * direction[i]) for i, v in enumerate(vars)}
    try:
        series_expr = sp.expand(f.subs(subs) - f.subs(point))
        s = sp.series(series_expr, t, 0, max_order + 1).removeO()
    except (TypeError, ValueError, ArithmeticError, NotImplementedError):
        try:
            s = sp.expand((f.subs(subs) - f.subs(point)))
        except (TypeError, ValueError, ArithmeticError, NotImplementedError):
            return "unknown", None, sp.Integer(0)
    poly = sp.Poly(sp.expand(s), t)
    if poly.is_zero:
        return "flat", None, sp.Integer(0)
    for k in range(1, poly.degree() + 1):
        c = sp.simplify(poly.coeff_monomial(t**k))
        if c != 0:
            if c.is_positive:
                return "up", k, c
            if c.is_negative:
                return "down", k, c
            return "unknown", k, c
    return "flat", None, sp.Integer(0)


def classify_degenerate(
    f: sp.Expr,
    vars: Sequence[sp.Symbol],
    point: Dict[sp.Symbol, sp.Expr],
    hessian_at_point: sp.Matrix,
):
    dirs = nullspace_directions(hessian_at_point)
    if not dirs:
        return None, []
    results = [directional_taylor_test(f, vars, point, d) for d in dirs]
    if any(r[0] == "down" for r in results):
        return "DegenerateSaddleCandidate", results
    if all(r[0] in {"up", "flat"} for r in results):
        if any(r[0] == "up" for r in results):
            return "FlatMinimumCandidate", results
        return "FlatDirectionDetected", results
    return "Degenerate", results
