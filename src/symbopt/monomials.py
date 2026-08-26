from __future__ import annotations

from functools import lru_cache
from itertools import product
from typing import Dict, Sequence, Tuple

import sympy as sp

Exponent = Tuple[int, ...]


@lru_cache(maxsize=4096)
def exponent_tuples(size: int, degree: int) -> Tuple[Exponent, ...]:
    powers = [
        values
        for values in product(range(degree + 1), repeat=size)
        if sum(values) <= degree
    ]
    powers.sort(key=lambda values: (sum(values), values))
    return tuple(powers)


@lru_cache(maxsize=4096)
def monomial_basis(
    variables: Tuple[sp.Symbol, ...],
    degree: int,
) -> Tuple[sp.Expr, ...]:
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
    variables: Tuple[sp.Symbol, ...],
) -> Tuple[sp.Expr, Exponent]:
    poly = sp.Poly(term, *variables)
    terms = poly.terms()
    if len(terms) != 1:
        raise ValueError("Expected a single monomial term.")
    powers, coeff = terms[0]
    return coeff, tuple(int(power) for power in powers)


@lru_cache(maxsize=4096)
def basis_data(
    basis: Tuple[sp.Expr, ...],
    variables: Tuple[sp.Symbol, ...],
) -> Tuple[Tuple[sp.Expr, Exponent], ...]:
    return tuple(monomial_data(term, variables) for term in basis)


@lru_cache(maxsize=4096)
def polynomial_terms(
    expr: sp.Expr,
    variables: Tuple[sp.Symbol, ...],
) -> Tuple[Tuple[Exponent, sp.Expr], ...]:
    poly = sp.Poly(sp.expand(expr), *variables)
    return tuple(
        (tuple(int(power) for power in powers), coeff) for powers, coeff in poly.terms()
    )


def coefficient_map(
    expr: sp.Expr,
    variables: Sequence[sp.Symbol],
) -> Dict[Exponent, sp.Expr]:
    data = polynomial_terms(expr, tuple(variables))
    return dict(data)


def add_exponents(first: Exponent, second: Exponent) -> Exponent:
    return tuple(a + b for a, b in zip(first, second))


def add_three(
    first: Exponent,
    second: Exponent,
    third: Exponent,
) -> Exponent:
    return tuple(a + b + c for a, b, c in zip(first, second, third))


@lru_cache(maxsize=16384)
def moment_symbol(powers: Exponent) -> sp.Symbol:
    suffix = "_".join(str(power) for power in powers)
    return sp.Symbol(f"y_{suffix}", real=True)
