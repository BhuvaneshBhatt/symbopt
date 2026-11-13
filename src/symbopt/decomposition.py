from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import sympy as sp
from sympy.core.relational import Relational


@dataclass(frozen=True)
class SeparableComponent:
    """One objective component with the decision variables and constraints that belong to it."""

    variables: tuple[sp.Symbol, ...]
    objective: sp.Expr
    constraints: tuple[Relational, ...]


@dataclass(frozen=True)
class SeparableDecomposition:
    """Independent objective components whose certified extrema can be combined."""

    components: tuple[SeparableComponent, ...]
    constant: sp.Expr = field(default_factory=lambda: sp.Integer(0))


def find_components(
    objective: sp.Expr,
    constraints: Sequence[Relational],
    variables: Sequence[sp.Symbol],
):
    """Detect additive independent-variable components.

    Two decision variables are connected whenever they occur together in one
    additive objective term or in one constraint.  A decomposition is returned
    only when at least two nonempty components remain and every constraint
    belongs to exactly one decision-variable component.
    """
    variables = tuple(variables)
    if len(variables) < 2:
        return None
    varset = set(variables)
    adjacency = {var: set() for var in variables}
    terms = sp.Add.make_args(sp.expand(objective))

    def connect(symbols):
        symbols = tuple(symbols)
        for left in symbols:
            adjacency[left].update(sym for sym in symbols if sym != left)

    for term in terms:
        connect(term.free_symbols.intersection(varset))
    for relation in constraints:
        used = relation.free_symbols.intersection(varset)
        if not used:
            # Parameter-only/global logical conditions need separate semantics;
            # leave them to the ordinary pipeline.
            return None
        connect(used)

    seen = set()
    components = []
    for start in variables:
        if start in seen:
            continue
        stack = [start]
        group = set()
        while stack:
            current = stack.pop()
            if current in group:
                continue
            group.add(current)
            seen.add(current)
            stack.extend(adjacency[current] - group)
        components.append(tuple(var for var in variables if var in group))
    if len(components) < 2:
        return None

    constant_terms = []
    objectives = {component: [] for component in components}
    for term in terms:
        used = term.free_symbols.intersection(varset)
        if not used:
            constant_terms.append(term)
            continue
        matches = [component for component in components if used.issubset(component)]
        if len(matches) != 1:
            return None
        objectives[matches[0]].append(term)

    constraint_groups = {component: [] for component in components}
    for relation in constraints:
        used = relation.free_symbols.intersection(varset)
        matches = [component for component in components if used.issubset(component)]
        if len(matches) != 1:
            return None
        constraint_groups[matches[0]].append(relation)

    result = []
    for component in components:
        component_objective = (
            sp.Add(*objectives[component]) if objectives[component] else sp.Integer(0)
        )
        result.append(
            SeparableComponent(
                variables=component,
                objective=sp.simplify(component_objective),
                constraints=tuple(constraint_groups[component]),
            )
        )
    return SeparableDecomposition(
        components=tuple(result),
        constant=sp.simplify(sp.Add(*constant_terms)),
    )
