from __future__ import annotations
from typing import Dict, Optional, Tuple
import sympy as sp
from sympy.polys.rootoftools import CRootOf, RootOf


def is_fixed_real_algebraic(expr: object) -> bool:
    """Return whether *expr* is an exact real algebraic constant."""
    value = sp.sympify(expr)
    if value.free_symbols:
        return False
    if value.is_real is False:
        return False
    return bool(
        value.is_algebraic is True
        or isinstance(value, (RootOf, CRootOf, sp.AlgebraicNumber))
    )


def isolating_interval(expr: object) -> Optional[Tuple[sp.Rational, sp.Rational]]:
    """Return an exact rational isolating interval for a fixed real RootOf.

    Rational/algebraic constants that SymPy can radicalize are returned as a
    degenerate interval when they are rational.  No floating approximation is
    used to identify a root branch.
    """
    value = sp.sympify(expr)
    if isinstance(value, sp.AlgebraicNumber):
        value = value.as_expr()
    if isinstance(value, (RootOf, CRootOf)):
        if value.free_symbols or value.is_real is False:
            return None
        try:
            interval = value._get_interval()
            return sp.Rational(interval.a), sp.Rational(interval.b)
        except (
            TypeError,
            ValueError,
            ArithmeticError,
            NotImplementedError,
            AttributeError,
        ):
            return None
    if isinstance(value, sp.Rational):
        return value, value
    return None


def exactify(expr):
    value = sp.sympify(expr)
    # Preserve exact algebraic branch objects; nsimplify is intended for
    # approximate input and can obscure RootOf identity/provenance.
    if isinstance(value, sp.AlgebraicNumber):
        return value.as_expr()
    if isinstance(value, (RootOf, CRootOf)) or (
        not value.free_symbols
        and value.is_algebraic is True
        and not value.has(sp.Float)
    ):
        return value
    try:
        return sp.nsimplify(value, rational=True)
    except (TypeError, ValueError, ArithmeticError, NotImplementedError):
        try:
            return sp.AlgebraicNumber(value).as_expr()
        except (TypeError, ValueError, ArithmeticError, NotImplementedError):
            return value


def exactify_dict(d: Dict[sp.Symbol, object]) -> Dict[sp.Symbol, sp.Expr]:
    return {k: exactify(v) for k, v in d.items()}


def _truth_value(relation) -> Optional[bool]:
    if relation is sp.S.true or relation is True:
        return True
    if relation is sp.S.false or relation is False:
        return False
    try:
        simplified = sp.simplify(relation)
        if simplified is sp.S.true:
            return True
        if simplified is sp.S.false:
            return False
    except (TypeError, ValueError, ArithmeticError, NotImplementedError):
        pass
    return None


def algebraic_compare(a, b) -> Optional[int]:
    """Compare exact real algebraic values without tolerance-based equality.

    Returns ``-1``, ``0`` or ``1`` when the ordering is established exactly,
    otherwise ``None``.  In particular, a tiny numerical difference is never
    rounded to zero.
    """
    try:
        left = exactify(a)
        right = exactify(b)
        d = sp.simplify(left - right)
        if d.is_zero is True or _truth_value(sp.Eq(left, right)) is True:
            return 0
        if d.is_positive is True or _truth_value(sp.Gt(left, right)) is True:
            return 1
        if d.is_negative is True or _truth_value(sp.Lt(left, right)) is True:
            return -1
        sign = sp.sign(d)
        if sign == 1:
            return 1
        if sign == -1:
            return -1
        if sign == 0:
            return 0
    except (TypeError, ValueError, ArithmeticError, NotImplementedError):
        return None
    return None
