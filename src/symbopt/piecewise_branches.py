from __future__ import annotations

from collections.abc import Sequence

import sympy as sp
from sympy.core.relational import Relational

from .problem_models import BranchCase


def _expand_abs(expr: sp.Expr) -> list[tuple[list[Relational], sp.Expr]]:
    atoms = sorted(expr.atoms(sp.Abs), key=sp.sstr)
    branches: list[tuple[list[Relational], sp.Expr]] = [([], expr)]

    for abs_atom in atoms:
        inner = abs_atom.args[0]
        new_branches: list[tuple[list[Relational], sp.Expr]] = []
        for assumptions, current_expr in branches:
            pos_expr = sp.simplify(current_expr.xreplace({abs_atom: inner}))
            neg_expr = sp.simplify(current_expr.xreplace({abs_atom: -inner}))
            new_branches.append((assumptions + [sp.Ge(inner, 0)], pos_expr))
            new_branches.append((assumptions + [sp.Lt(inner, 0)], neg_expr))
        branches = new_branches
    return branches


def abs_to_piecewise_cases(
    objective: sp.Expr,
    constraints: Sequence[Relational] | None = None,
) -> list[BranchCase]:
    """Expand absolute values into owned sign branches for algebraization."""
    constraints = list(constraints or [])
    current_cases = [
        BranchCase(assumptions=a, objective=o, constraints=[])
        for a, o in _expand_abs(objective)
    ]

    for rel in constraints:
        lhs_branches = _expand_abs(rel.lhs)
        rhs_branches = _expand_abs(rel.rhs)
        next_cases: list[BranchCase] = []
        for case in current_cases:
            for lhs_assumptions, lhs_expr in lhs_branches:
                for rhs_assumptions, rhs_expr in rhs_branches:
                    next_cases.append(
                        BranchCase(
                            assumptions=case.assumptions
                            + lhs_assumptions
                            + rhs_assumptions,
                            objective=case.objective,
                            constraints=case.constraints
                            + [rel.func(lhs_expr, rhs_expr)],
                        )
                    )
        current_cases = next_cases

    return current_cases
