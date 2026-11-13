from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import sympy as sp
from sympy.core.relational import Relational

from .univariate_analysis import certified_monotone_direction


@dataclass(frozen=True)
class MonotoneReduction:
    """An equivalent minimization problem under a proved monotone outer transformation."""

    reduced_objective: sp.Expr
    added_constraints: tuple[Relational, ...]
    kind: str
    increasing: bool = True

    def recover_value(self, reduced_value: sp.Expr) -> sp.Expr:
        """Map the reduced infimum through the original outer function, including infinite limits."""
        inner_value = reduced_value if self.increasing else -reduced_value
        if self.kind == "exp":
            if inner_value is -sp.oo:
                return sp.Integer(0)
            return sp.exp(inner_value)
        if self.kind.startswith("log_scale:"):
            scale = sp.Rational(self.kind.split(":", 1)[1])
            return scale * sp.log(inner_value)
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
        z = sp.Dummy("_outer", real=True)
        increasing = certified_monotone_direction(sp.exp(z), z, sp.S.Reals)
        return MonotoneReduction(
            objective.args[0], (), "exp", True if increasing is None else increasing
        )

    if objective.func is sp.log and objective.args[0].free_symbols.intersection(
        variables
    ):
        inner = objective.args[0]
        z = sp.Dummy("_outer", positive=True)
        increasing = certified_monotone_direction(
            sp.log(z), z, sp.Interval.open(0, sp.oo)
        )
        return MonotoneReduction(
            inner,
            (sp.Gt(inner, 0),),
            "log",
            True if increasing is None else increasing,
        )

    terms = sp.Add.make_args(objective)
    if 1 < len(terms) <= 8:
        weighted = [term.as_coeff_Mul() for term in terms]
        if all(
            c.is_Rational and c != 0 and f.func is sp.log and f.args[0].is_real is True
            for c, f in weighted
        ):
            signs = {sp.sign(c) for c, _ in weighted}
            if len(signs) == 1:
                sign = signs.pop()
                scale = sp.Rational(sign, sp.ilcm(*(c.q for c, _ in weighted)))
                powers = [int(c / scale) for c, _ in weighted]
                if sum(powers) <= 16:
                    arguments = [f.args[0] for _, f in weighted]
                    product = sp.prod(
                        arg**power for arg, power in zip(arguments, powers)
                    )
                    # Each logarithm must have a positive real argument; a
                    # positive product alone would also admit two negatives.
                    return MonotoneReduction(
                        product if sign > 0 else -product,
                        tuple(sp.Gt(arg, 0) for arg in arguments),
                        f"log_scale:{scale}",
                        sign > 0,
                    )

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
                z = sp.Dummy("_outer", real=True)
                direction = certified_monotone_direction(base**z, z, sp.S.Reals)
                increasing = (
                    direction
                    if direction is not None
                    else sp.ask(sp.Q.gt(base, 1)) is True
                )
                decreasing = (
                    not direction
                    if direction is not None
                    else sp.ask(sp.Q.lt(base, 1)) is True
                )
                if increasing or decreasing:
                    reduced = exponent if increasing else -exponent
                    return MonotoneReduction(
                        reduced,
                        (),
                        f"positive_base_exp:{sp.sstr(base)}",
                        increasing,
                    )
    return None
