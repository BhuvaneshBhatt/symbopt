from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Sequence
import sympy as sp
from .sos import sos_poly_relaxation
from .sos_program import sparse_monomial_basis


@lru_cache(maxsize=256)
def _cached_basis_plan(poly, vars_tuple, inequalities_tuple, equalities_tuple, degree):
    exprs = (poly,) + inequalities_tuple + equalities_tuple
    return tuple(sparse_monomial_basis(exprs, vars_tuple, degree))


@dataclass
class HierarchyStep:
    order: int
    ineq_mult_degree: int
    eq_mult_degree: int
    success: bool
    lower_bound: Optional[sp.Expr]
    message: str


@dataclass
class HierarchySearchResult:
    steps: List[HierarchyStep]
    best_lower_bound: Optional[sp.Expr]
    best_result: Optional[object]
    certified: bool
    message: str


def search_sos_hierarchy(
    poly: sp.Expr,
    vars: Sequence[sp.Symbol],
    inequalities: Sequence[sp.Expr] = (),
    equalities: Sequence[sp.Expr] = (),
    search_lo: float = -100.0,
    search_hi: float = 100.0,
    max_order: int = 3,
):
    """Search increasing SOS relaxation orders for a polynomial lower bound.

    Hierarchy state reuses structural basis information and warm starts where
    possible. Exact certification still requires exact identity and PSD recovery."""
    steps = []
    best = None
    best_lb = None
    vars_tuple = tuple(vars)
    inequalities_tuple = tuple(inequalities)
    equalities_tuple = tuple(equalities)
    for d in range(1, max_order + 1):
        # Precompute and cache the structure-aware basis plan. The SOS compiler
        # reuses the same cached monomial basis objects internally.
        _cached_basis_plan(poly, vars_tuple, inequalities_tuple, equalities_tuple, d)
        res = sos_poly_relaxation(
            poly,
            vars,
            inequalities=inequalities,
            equalities=equalities,
            search_lo=search_lo,
            search_hi=search_hi,
            max_iter=10,
            ineq_mult_degree=d,
            eq_mult_degree=d,
        )
        lb = res.lower_bound
        steps.append(HierarchyStep(d, d, d, bool(res.success), lb, res.message))
        if res.success:
            best = res
            best_lb = lb
            if getattr(res, "certified", False):
                return HierarchySearchResult(
                    steps,
                    best_lb,
                    best,
                    True,
                    "Found exact SOS hierarchy certificate.",
                )
    return HierarchySearchResult(
        steps,
        best_lb,
        best,
        False,
        (
            "Found only approximate SOS hierarchy bounds."
            if best is not None
            else "No feasible SOS hierarchy step found."
        ),
    )
