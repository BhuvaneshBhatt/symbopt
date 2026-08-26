from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import sympy as sp
from sympy.core.relational import Relational


@dataclass(frozen=True)
class MonotoneReduction:
    reduced_objective: sp.Expr
    added_constraints: Tuple[Relational, ...]
    kind: str
    increasing: bool = True

    def recover_value(self, reduced_value: sp.Expr) -> sp.Expr:
        inner_value = reduced_value if self.increasing else -reduced_value
        if self.kind == "exp":
            if inner_value is -sp.oo:
                return sp.Integer(0)
            return sp.exp(inner_value)
        if self.kind == "log":
            if inner_value == 0:
                return -sp.oo
            return sp.log(inner_value)
        if self.kind == "sqrt":
            return sp.sqrt(inner_value)
        if self.kind.startswith("positive_power:"):
            exponent = sp.sympify(self.kind.split(":", 1)[1])
            return sp.Pow(inner_value, exponent)
        if self.kind.startswith("positive_base_exp:"):
            base = sp.sympify(self.kind.split(":", 1)[1])
            return sp.Pow(base, inner_value)
        raise ValueError(f"Unknown monotone reduction kind: {self.kind}")


def find_monotone_reduction(
    objective: sp.Expr,
    constraints: Sequence[Relational],
    variables: Sequence[sp.Symbol],
):
    """Recognize exact monotone outer compositions around a simpler objective.

    Only globally monotone real branches with explicit domain constraints are
    reduced. The returned problem is equivalent for minimization.
    """
    objective = sp.sympify(objective)
    variables = tuple(variables)

    if objective.func is sp.exp and objective.args[0].free_symbols.intersection(
        variables
    ):
        return MonotoneReduction(objective.args[0], (), "exp", True)

    if objective.func is sp.log and objective.args[0].free_symbols.intersection(
        variables
    ):
        inner = objective.args[0]
        return MonotoneReduction(inner, (sp.Gt(inner, 0),), "log", True)

    if isinstance(objective, sp.Pow):
        base, exponent = objective.as_base_exp()
        # sqrt and other positive rational powers of a variable-dependent base.
        if (
            base.free_symbols.intersection(variables)
            and exponent.is_Rational
            and exponent > 0
        ):
            if exponent.q % 2 == 0:
                kind = (
                    "sqrt"
                    if exponent == sp.Rational(1, 2)
                    else f"positive_power:{sp.sstr(exponent)}"
                )
                return MonotoneReduction(base, (sp.Ge(base, 0),), kind, True)
            # Odd positive integer powers are globally increasing on R. Even
            # integer powers are not monotone and must not be reduced.
            if exponent.q == 1 and int(exponent) % 2 == 1:
                kind = f"positive_power:{sp.sstr(exponent)}"
                return MonotoneReduction(base, (), kind, True)

        # a**g(x), a>0.  For 0<a<1 the map is decreasing, so minimize -g.
        if exponent.free_symbols.intersection(variables) and not base.free_symbols:
            positive = sp.ask(sp.Q.positive(base))
            if positive is True and base != 1:
                increasing = sp.ask(sp.Q.gt(base, 1)) is True
                decreasing = sp.ask(sp.Q.lt(base, 1)) is True
                if increasing or decreasing:
                    reduced = exponent if increasing else -exponent
                    return MonotoneReduction(
                        reduced,
                        (),
                        f"positive_base_exp:{sp.sstr(base)}",
                        increasing,
                    )
    return None
