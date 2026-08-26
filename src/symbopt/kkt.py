from __future__ import annotations

from typing import Optional

import sympy as sp

from .problem_models import KKTSystem, LiftedProblem


def build_kkt_system(
    problem: LiftedProblem,
    objective: Optional[sp.Expr] = None,
    close_strict_ineq: bool = True,
    include_graph_equalities: bool = True,
    include_graph_ineq: bool = True,
) -> KKTSystem:
    """Construct KKT equations for a lifted minimization problem.

    Inequalities use the package convention ``h(x) >= 0`` and therefore enter the
    Lagrangian as ``-mu*h`` with ``mu >= 0``. The returned object records
    stationarity, feasibility, and complementarity equations separately."""
    f = sp.expand(problem.lifted_objective if objective is None else objective)

    eq_residuals, ineq_residuals = problem.kkt_residuals(
        include_graph_eq=include_graph_equalities,
        include_graph_ineq=include_graph_ineq,
        close_strict=close_strict_ineq,
    )

    lam = [sp.Symbol(f"lam_{i + 1}", real=True) for i in range(len(eq_residuals))]
    mu = [sp.Symbol(f"mu_{j + 1}", real=True) for j in range(len(ineq_residuals))]

    L = f
    for mult, residual in zip(lam, eq_residuals):
        L += mult * residual
    for m, h in zip(mu, ineq_residuals):
        L -= m * h

    stationarity = [
        sp.Eq(sp.expand(sp.diff(L, x)), 0, evaluate=False)
        for x in problem.all_variables
    ]
    primal_equalities = [sp.Eq(g, 0, evaluate=False) for g in eq_residuals]
    primal_inequalities = [sp.Ge(h, 0) for h in ineq_residuals]
    dual_inequalities = [sp.Ge(m, 0) for m in mu]
    complementarity = [
        sp.Eq(sp.expand(m * h), 0, evaluate=False) for m, h in zip(mu, ineq_residuals)
    ]

    return KKTSystem(
        primal_variables=list(problem.all_variables),
        equality_multipliers=lam,
        inequality_multipliers=mu,
        lagrangian=sp.expand(L),
        stationarity_equations=stationarity,
        primal_equalities=primal_equalities,
        primal_inequalities=primal_inequalities,
        dual_inequalities=dual_inequalities,
        complementarity_eqs=complementarity,
    )
