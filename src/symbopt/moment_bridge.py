from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import sympy as sp

from .moment import build_moment_matrix
from .moments_dual import solve_moment_dual


@dataclass
class MomentCandidateHint:
    """A point, its objective value and its candidate Dirac moment sequence."""

    point: dict[sp.Symbol, sp.Expr]
    objective_value: sp.Expr
    moment_values: dict[tuple[int, ...], sp.Expr]


@dataclass
class SOSMomentReport:
    """Candidate moment hints and an optional dual-moment consistency result."""

    candidate_hints: list[MomentCandidateHint]
    dual_result: object | None
    message: str


def dirac_moments_from_point(
    point: dict[sp.Symbol, sp.Expr], vars: Sequence[sp.Symbol], order: int
) -> dict[tuple[int, ...], sp.Expr]:
    """Evaluate the truncated Dirac moment sequence at a specified coordinate mapping."""
    vars = list(vars)
    mm = build_moment_matrix(vars, order)
    out: dict[tuple[int, ...], sp.Expr] = {}
    for sym in mm.matrix.free_symbols:
        alpha = tuple(int(x) for x in sym.name.split("_")[1:])
        val = sp.Integer(1)
        for v, e in zip(vars, alpha):
            val *= sp.expand(point[v] ** e)
        out[alpha] = sp.nsimplify(sp.expand(val))
    out[(0,) * len(vars)] = sp.Integer(1)
    return out


def moment_hint_from_point(
    objective: sp.Expr,
    point: dict[sp.Symbol, sp.Expr],
    vars: Sequence[sp.Symbol],
    order: int = 2,
) -> MomentCandidateHint:
    """Construct an objective-value and Dirac-moment hint from a proposed point."""
    mv = dirac_moments_from_point(point, vars, order)
    obj = sp.nsimplify(sp.expand(objective.subs(point)))
    return MomentCandidateHint(point=point, objective_value=obj, moment_values=mv)


def link_sos_moments(
    objective: sp.Expr,
    inequalities: Sequence[sp.Expr],
    vars: Sequence[sp.Symbol],
    candidate_points: Sequence[dict[sp.Symbol, sp.Expr]],
    order: int = 2,
) -> SOSMomentReport:
    """Build consistency information linking candidate points, SOS data, and moment relaxations."""
    hints = [
        moment_hint_from_point(objective, p, vars, order=order)
        for p in candidate_points
    ]
    dual = None
    msg = (
        "No candidate points available for moment consistency."
        if not hints
        else "Constructed Dirac-moment candidate hints."
    )
    if hints:
        try:
            dual = solve_moment_dual(
                objective,
                inequalities,
                vars,
                order,
                moment_values=hints[0].moment_values,
            )
            msg += " Built dual-moment object from first hint."
        except (TypeError, ValueError, ArithmeticError, NotImplementedError) as exc:
            msg += f" Dual-moment construction was unavailable: {type(exc).__name__}: {exc}"
    return SOSMomentReport(candidate_hints=hints, dual_result=dual, message=msg)
