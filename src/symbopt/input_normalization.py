"""Normalization helpers for user-facing optimization inputs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import product

import sympy as sp
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean, BooleanFalse, BooleanTrue

from .precision import outward_float


def _is_true(item) -> bool:
    return (
        item is None
        or item is True
        or item is sp.S.true
        or isinstance(item, BooleanTrue)
    )


def _is_false(item) -> bool:
    return item is False or item is sp.S.false or isinstance(item, BooleanFalse)


def normalize_constraint_branches(
    constraints=None,
) -> tuple[list[Relational], ...]:
    """Normalize constraints into disjunctive branches of flat relations.

    ``And`` combines constraints within a branch and ``Or`` creates alternative
    feasible branches. Lists, tuples, sets, and other finite iterables are
    interpreted conjunctively. ``True`` is ignored. ``False`` is rejected; use an explicit
    contradictory relation to represent an infeasible branch.
    """

    def visit(item) -> tuple[list[Relational], ...]:
        if _is_true(item):
            return ([],)
        if _is_false(item):
            raise TypeError(
                "False is not a relational constraint; use an explicitly contradictory relation if needed"
            )
        if isinstance(item, Relational):
            return ([item],)
        if isinstance(item, sp.Or):
            out: list[list[Relational]] = []
            for argument in item.args:
                out.extend(visit(argument))
            return tuple(out)
        if isinstance(item, sp.And):
            parts = [visit(argument) for argument in item.args]
            if any(not branch_set for branch_set in parts):
                return ()
            return tuple(
                [relation for branch in choice for relation in branch]
                for choice in product(*parts)
            )
        if isinstance(item, Boolean):
            raise TypeError(
                "constraints must use SymPy relations combined with And/Or; "
                f"unsupported Boolean expression: {item!s}"
            )
        if isinstance(item, Mapping):
            raise TypeError(
                "constraint mappings are ambiguous; use bounds={...} for variable bounds"
            )
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes, sp.Basic)):
            branches: tuple[list[Relational], ...] = ([],)
            for element in item:
                right = visit(element)
                if not right:
                    return ()
                branches = tuple(
                    [*left_branch, *right_branch]
                    for left_branch in branches
                    for right_branch in right
                )
            return branches
        raise TypeError(
            "constraints must be None, a SymPy relation, an And/Or expression, "
            "or an iterable of constraints"
        )

    return visit(constraints)


def normalize_constraints_input(constraints=None) -> list[Relational]:
    """Return one flat conjunction; reject genuinely disjunctive input."""
    branches = normalize_constraint_branches(constraints)
    if not branches:
        return [sp.Eq(sp.Integer(0), sp.Integer(1), evaluate=False)]
    if len(branches) != 1:
        raise TypeError("disjunctive constraints require branch-aware optimization")
    return branches[0]


def normalize_variables(variables) -> list[sp.Symbol] | None:
    """Normalize a symbol or iterable of symbols to a validated list."""
    if variables is None:
        return None
    if isinstance(variables, sp.Symbol):
        result = [variables]
    elif isinstance(variables, Iterable) and not isinstance(
        variables, (str, bytes, sp.Basic)
    ):
        result = list(variables)
    else:
        raise TypeError(
            "variables must be a SymPy Symbol or an iterable of SymPy Symbols"
        )
    if not result:
        raise ValueError("variables must contain at least one optimization variable")
    if any(not isinstance(variable, sp.Symbol) for variable in result):
        raise TypeError("every optimization variable must be a SymPy Symbol")
    if len(set(result)) != len(result):
        raise ValueError("optimization variables must be unique")
    return result


def normalize_bounds_input(bounds, variables: Sequence[sp.Symbol] | None = None):
    """Return ``(constraints, finite_box_bounds)`` from common bounds syntax."""
    if bounds is None:
        return [], {}
    if isinstance(bounds, Mapping):
        items = list(bounds.items())
    else:
        if variables is None:
            raise TypeError("sequence bounds require explicit variables")
        try:
            pairs = list(bounds)
        except TypeError as exc:
            raise TypeError(
                "bounds must be a mapping or a sequence of (lower, upper) pairs"
            ) from exc
        if len(pairs) != len(variables):
            raise ValueError(
                "sequence bounds must have one pair per optimization variable"
            )
        items = list(zip(variables, pairs))

    constraints = []
    finite = {}
    allowed_variables = None if variables is None else set(variables)
    for variable, pair in items:
        if not isinstance(variable, sp.Symbol):
            raise TypeError("bounds keys must be SymPy symbols")
        if allowed_variables is not None and variable not in allowed_variables:
            raise ValueError(
                f"bound variable {variable} is not an optimization variable"
            )
        if pair is None:
            continue
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("each bound must be a (lower, upper) pair")
        lower, upper = pair
        if lower is not None:
            lower = sp.sympify(lower)
            if lower is not -sp.oo:
                constraints.append(sp.Ge(variable, lower))
        if upper is not None:
            upper = sp.sympify(upper)
            if upper is not sp.oo:
                constraints.append(sp.Le(variable, upper))
        if lower is not None and upper is not None:
            lower_s, upper_s = sp.sympify(lower), sp.sympify(upper)
            if (
                lower_s not in (-sp.oo, sp.oo)
                and upper_s not in (-sp.oo, sp.oo)
                and lower_s.is_number
                and upper_s.is_number
            ):
                finite[variable] = (
                    outward_float(lower_s, -1),
                    outward_float(upper_s, 1),
                )
    return constraints, finite


def assumption_constraints(variables: Sequence[sp.Symbol]) -> list[Relational]:
    """Translate explicit sign assumptions on symbols into domain relations."""
    relations = []
    for variable in variables:
        if variable.is_positive is True:
            relations.append(sp.Gt(variable, 0, evaluate=False))
        elif variable.is_nonnegative is True:
            relations.append(sp.Ge(variable, 0, evaluate=False))
        elif variable.is_negative is True:
            relations.append(sp.Lt(variable, 0, evaluate=False))
        elif variable.is_nonpositive is True:
            relations.append(sp.Le(variable, 0, evaluate=False))
    return relations


def require_continuous_variables(variables):
    """Reject discrete decision domains before any continuous solver can run."""
    if any(variable.is_integer is True for variable in variables):
        raise NotImplementedError(
            "integer optimization is not supported; decision variables must be continuous"
        )
