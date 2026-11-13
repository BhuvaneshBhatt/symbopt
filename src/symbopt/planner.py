from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import comb

import sympy as sp
from sympy.core.relational import Relational

from .relation_checks import normalize_relational
from .semialgebraic import semialg_available
from .sos_search import estimate_sos_cost


@dataclass(frozen=True)
class ProblemComplexity:
    """Cheap structural metrics used to route an optimization problem."""

    variable_count: int
    constraint_count: int
    equality_count: int
    inequality_count: int
    polynomial: bool
    max_degree: int | None
    monomial_count: int | None
    density: float | None
    estimated_active_sets: int
    estimated_kkt_degree: int | None
    estimated_groebner_score: int | None
    estimated_sos_gram_dimension: int | None
    estimated_sos_scalar_variables: int | None
    semialg_available: bool
    finite_box: bool


@dataclass(frozen=True)
class SolverPlan:
    """Automatic route plan computed before expensive symbolic/numeric work."""

    complexity: ProblemComplexity
    try_semialg_early: bool
    try_exact_kkt: bool
    try_numeric_kkt: bool
    try_sos: bool
    try_interval: bool
    reasons: Mapping[str, str]

    def explain(self) -> str:
        """Describe the selected proposal routes and the budgets that govern them."""
        selected = [
            name
            for name, enabled in (
                ("semialg", self.try_semialg_early),
                ("exact-kkt", self.try_exact_kkt),
                ("numeric-kkt", self.try_numeric_kkt),
                ("sos", self.try_sos),
                ("interval", self.try_interval),
            )
            if enabled
        ]
        return "auto planner: " + ", ".join(selected or ["structural/direct only"])


def _poly_metrics(expressions, variables):
    polys = []
    try:
        for expr in expressions:
            polys.append(sp.Poly(sp.expand(expr), *variables))
    except (sp.PolynomialError, TypeError, ValueError):
        return False, None, None, None
    degrees = [0 if poly.is_zero else int(poly.total_degree()) for poly in polys]
    monomials = sum(len(poly.terms()) for poly in polys)
    max_degree = max(degrees, default=0)
    if not variables:
        density = 0.0
    else:
        possible = sum(comb(len(variables) + degree, degree) for degree in degrees)
        density = float(monomials / max(1, possible))
    return True, max_degree, monomials, density


def analyze_problem(
    objective, constraints, variables, box_bounds=None
) -> ProblemComplexity:
    """Return inexpensive complexity estimates without invoking a solver."""
    variables = tuple(variables)
    residuals = [sp.sympify(objective)]
    sos_equalities = []
    sos_inequalities = []
    equality_count = 0
    inequality_count = 0
    for constraint in constraints:
        if not isinstance(constraint, Relational):
            residuals.append(sp.sympify(constraint))
            continue
        kind, residual = normalize_relational(constraint)
        residuals.append(residual)
        if kind == "eq":
            equality_count += 1
            sos_equalities.append(residual)
        elif kind == "ge":
            inequality_count += 1
            sos_inequalities.append(residual)
        else:
            inequality_count += 1
    polynomial, max_degree, monomial_count, density = _poly_metrics(
        residuals, variables
    )

    active_sets = 2 ** min(inequality_count, 30)
    kkt_degree = None
    groebner_score = None
    sos_gram = None
    sos_scalars = None
    if polynomial and max_degree is not None:
        stationarity_degree = max(1, max_degree - 1)
        kkt_degree = stationarity_degree ** max(1, len(variables))
        groebner_score = max(1, len(variables) + len(constraints)) * kkt_degree
        try:
            estimate = estimate_sos_cost(
                objective,
                variables=variables,
                inequalities=sos_inequalities,
                equalities=sos_equalities,
            )
            if estimate.applicable:
                sos_gram = estimate.main_gram_dimension
                sos_scalars = estimate.scalar_decision_vars
        except (TypeError, ValueError, ArithmeticError):
            pass

    finite_box = (
        bool(variables)
        and box_bounds is not None
        and all(
            var in box_bounds
            and box_bounds[var][0] not in (None, -sp.oo)
            and box_bounds[var][1] not in (None, sp.oo)
            for var in variables
        )
    )
    return ProblemComplexity(
        len(variables),
        len(constraints),
        equality_count,
        inequality_count,
        polynomial,
        max_degree,
        monomial_count,
        density,
        active_sets,
        kkt_degree,
        groebner_score,
        sos_gram,
        sos_scalars,
        semialg_available(),
        finite_box,
    )


def plan_problem(objective, constraints, variables, box_bounds=None) -> SolverPlan:
    """Construct a conservative automatic solver plan from cheap metrics."""
    info = analyze_problem(objective, constraints, variables, box_bounds)
    reasons = {}

    semialg = bool(
        info.semialg_available
        and info.polynomial
        and info.variable_count <= 4
        and (info.max_degree or 0) <= 8
        and (info.estimated_groebner_score or 0) <= 25000
    )
    reasons["semialg"] = (
        "small exact semialgebraic problem"
        if semialg
        else "unavailable, nonpolynomial, or estimated elimination cost is high"
    )

    sparsity_discount = 2 if info.density is not None and info.density <= 0.25 else 1
    exact_kkt = bool(
        info.polynomial
        and info.variable_count <= 6
        and info.estimated_active_sets <= 256
        and (info.estimated_groebner_score or 0) <= 12000 * sparsity_discount
    )
    reasons["exact_kkt"] = (
        "active-set and algebraic degree estimates are within budget"
        if exact_kkt
        else "estimated active-set/algebraic search is too large"
    )

    numeric_kkt = bool(info.variable_count <= 30 and info.estimated_active_sets <= 4096)
    reasons["numeric_kkt"] = (
        "candidate search is within numerical active-set budget"
        if numeric_kkt
        else "too many variables or active sets for automatic enumeration"
    )

    gram_budget = 65 if info.density is not None and info.density <= 0.20 else 45
    scalar_budget = 3000 if info.density is not None and info.density <= 0.20 else 1800
    sos = bool(
        info.polynomial
        and info.estimated_sos_gram_dimension is not None
        and info.estimated_sos_gram_dimension <= gram_budget
        and (info.estimated_sos_scalar_variables or 0) <= scalar_budget
    )
    reasons["sos"] = (
        "estimated Gram/SDP size is within automatic budget"
        if sos
        else "SOS Gram estimate is unavailable or too large"
    )

    interval = bool(info.finite_box and info.variable_count <= 10)
    reasons["interval"] = (
        "finite box available"
        if interval
        else "verified interval search needs a finite low-dimensional box"
    )
    return SolverPlan(info, semialg, exact_kkt, numeric_kkt, sos, interval, reasons)
