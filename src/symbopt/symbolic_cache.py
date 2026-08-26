from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import sympy as sp

from .monomials import (
    exponent_tuples,
    monomial_basis,
    polynomial_terms,
)


@lru_cache(maxsize=8192)
def canonical_expand(expr: sp.Expr) -> sp.Expr:
    return sp.expand(expr)


@lru_cache(maxsize=4096)
def cached_poly(
    expr: sp.Expr,
    variables: Tuple[sp.Symbol, ...],
) -> sp.Poly:
    return sp.Poly(canonical_expand(expr), *variables)


cached_poly_terms = polynomial_terms
cached_monomial_powers = exponent_tuples
cached_monomial_basis = monomial_basis


@lru_cache(maxsize=4096)
def cached_gradient(
    expr: sp.Expr,
    variables: Tuple[sp.Symbol, ...],
):
    return tuple(sp.diff(expr, variable) for variable in variables)


@lru_cache(maxsize=2048)
def cached_hessian(
    expr: sp.Expr,
    variables: Tuple[sp.Symbol, ...],
):
    return sp.hessian(expr, variables)


def clear_symbolic_caches() -> None:
    """Clear all shared symbolic caches owned by the cache module.

    This is mainly useful for reproducible cold-cache benchmarks or long-lived
    interactive sessions under memory pressure."""
    canonical_expand.cache_clear()
    cached_poly.cache_clear()
    polynomial_terms.cache_clear()
    exponent_tuples.cache_clear()
    monomial_basis.cache_clear()
    cached_gradient.cache_clear()
    cached_hessian.cache_clear()
    try:
        from .verified_intervals import compile_arb, enclose_expression

        compile_arb.cache_clear()
        enclose_expression.cache_clear()
    except ImportError:
        pass


def cache_info():
    """Return cache statistics for the shared symbolic caches.

    Useful for profiling repeated symbolic work and verifying cache reuse in
    benchmarks."""
    info = {
        "canonical_expand": canonical_expand.cache_info(),
        "poly": cached_poly.cache_info(),
        "poly_terms": polynomial_terms.cache_info(),
        "monomial_exponents": exponent_tuples.cache_info(),
        "monomial_basis": monomial_basis.cache_info(),
        "gradient": cached_gradient.cache_info(),
        "hessian": cached_hessian.cache_info(),
    }
    try:
        from .verified_intervals import compile_arb, enclose_expression

        info["arb_program"] = compile_arb.cache_info()
        info["interval_enclosure"] = enclose_expression.cache_info()
    except ImportError:
        pass
    return info
