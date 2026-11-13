from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from itertools import product

import sympy as sp

Exponent = tuple[int, ...]


@lru_cache(maxsize=4096)
def exponent_tuples(size: int, degree: int) -> tuple[Exponent, ...]:
    # Filtering a tiny Cartesian product is cheaper than generator recursion.
    """Return exponent tuples ordered by total degree and then lexicographically."""
    if size <= 1 or degree <= 0 or (degree + 1) ** size <= 1024:
        powers = [
            values
            for values in product(range(degree + 1), repeat=size)
            if sum(values) <= degree
        ]
        powers.sort(key=lambda values: (sum(values), values))
        return tuple(powers)
    return tuple(
        powers
        for total in range(degree + 1)
        for powers in _fixed_degree(size, total, ())
    )


def _fixed_degree(size, total, prefix):
    if size == 1:
        yield (*prefix, total)
    else:
        for first in range(total + 1):
            yield from _fixed_degree(size - 1, total - first, (*prefix, first))


@lru_cache(maxsize=4096)
def monomial_basis(
    variables: tuple[sp.Symbol, ...],
    degree: int,
) -> tuple[sp.Expr, ...]:
    """Return monomials of bounded total degree in the canonical exponent order."""
    basis = []
    for powers in exponent_tuples(len(variables), degree):
        term = sp.Integer(1)
        for variable, power in zip(variables, powers):
            term *= variable**power
        basis.append(term)
    return tuple(basis)


@lru_cache(maxsize=8192)
def monomial_data(
    term: sp.Expr,
    variables: tuple[sp.Symbol, ...],
) -> tuple[sp.Expr, Exponent]:
    """Return a single monomial's coefficient and exponent tuple, rejecting sums."""
    poly = sp.Poly(term, *variables)
    terms = poly.terms()
    if len(terms) != 1:
        raise ValueError("Expected a single monomial term.")
    powers, coeff = terms[0]
    return coeff, tuple(int(power) for power in powers)


@lru_cache(maxsize=4096)
def basis_data(
    basis: tuple[sp.Expr, ...],
    variables: tuple[sp.Symbol, ...],
) -> tuple[tuple[sp.Expr, Exponent], ...]:
    """Return coefficient and exponent data in the supplied monomial-basis order."""
    return tuple(monomial_data(term, variables) for term in basis)


@lru_cache(maxsize=4096)
def polynomial_terms(
    expr: sp.Expr,
    variables: tuple[sp.Symbol, ...],
) -> tuple[tuple[Exponent, sp.Expr], ...]:
    """Return immutable exponent/coefficient pairs for the expanded polynomial."""
    poly = sp.Poly(sp.expand(expr), *variables)
    return tuple(
        (tuple(int(power) for power in powers), coeff) for powers, coeff in poly.terms()
    )


def coefficient_map(
    expr: sp.Expr,
    variables: Sequence[sp.Symbol],
) -> dict[Exponent, sp.Expr]:
    """Map each exponent tuple to its exact polynomial coefficient."""
    data = polynomial_terms(expr, tuple(variables))
    return dict(data)


def add_exponents(first: Exponent, second: Exponent) -> Exponent:
    """Add exponent vectors component by component in the fixed coordinate order."""
    return tuple(a + b for a, b in zip(first, second))


def add_three(
    first: Exponent,
    second: Exponent,
    third: Exponent,
) -> Exponent:
    """Add exponent vectors component by component in the fixed coordinate order."""
    return tuple(a + b + c for a, b, c in zip(first, second, third))


@lru_cache(maxsize=16384)
def moment_symbol(powers: Exponent) -> sp.Symbol:
    """Return the real moment symbol associated with an exponent tuple."""
    suffix = "_".join(str(power) for power in powers)
    return sp.Symbol(f"y_{suffix}", real=True)
