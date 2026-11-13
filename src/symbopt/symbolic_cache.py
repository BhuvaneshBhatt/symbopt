from __future__ import annotations

from functools import lru_cache

import sympy as sp

from .monomials import (
    exponent_tuples,
    monomial_basis,
    polynomial_terms,
)


@lru_cache(maxsize=8192)
def canonical_expand(expr: sp.Expr) -> sp.Expr:
    """Return the cached expanded symbolic expression."""
    return sp.expand(expr)


@lru_cache(maxsize=4096)
def cached_poly(
    expr: sp.Expr,
    variables: tuple[sp.Symbol, ...],
) -> sp.Poly:
    """Construct a polynomial once for the expression and ordered coordinates."""
    return sp.Poly(canonical_expand(expr), *variables)


@lru_cache(maxsize=2048)
def structural_coordinate(name, problem_key, assumptions=()):
    """Return a capture-free coordinate stable across identical structural queries."""
    return sp.Dummy(name, **(dict(assumptions) or {"real": True}))


def simplify_scalar(value):
    """Simplify a scalar while keeping parameter-dependent Piecewise guards intact."""
    value = sp.sympify(value)
    if value.free_symbols and value.has(sp.Piecewise):
        return value
    return sp.simplify(value)


def affine_coefficients(expr, variables):
    """Return affine coordinate coefficients and constant, or None for nonaffine input."""
    return _cached_affine(sp.sympify(expr), tuple(variables))


@lru_cache(maxsize=4096)
def _cached_affine(expr, variables):
    try:
        poly = cached_poly(expr, variables)
    except (sp.PolynomialError, TypeError, ValueError):
        return None
    coefficients = [sp.S.Zero] * len(variables)
    constant = sp.S.Zero
    for powers, coefficient in poly.terms():
        if sum(powers) > 1:
            return None
        coefficient = coefficient if coefficient.is_Atom else sp.simplify(coefficient)
        if 1 in powers:
            coefficients[powers.index(1)] = coefficient
        else:
            constant = coefficient
    return tuple(coefficients), constant


cached_poly_terms = polynomial_terms
cached_monomial_powers = exponent_tuples
cached_monomial_basis = monomial_basis


@lru_cache(maxsize=4096)
def cached_gradient(
    expr: sp.Expr,
    variables: tuple[sp.Symbol, ...],
):
    """Return the cached symbolic gradient in the requested coordinate order."""
    return tuple(sp.diff(expr, variable) for variable in variables)


@lru_cache(maxsize=2048)
def cached_hessian(
    expr: sp.Expr,
    variables: tuple[sp.Symbol, ...],
):
    """Return the cached symbolic Hessian in the requested coordinate order."""
    return sp.hessian(expr, variables)


def clear_symbolic_caches() -> None:
    """Clear all shared symbolic caches owned by the cache module.

    This is mainly useful for reproducible cold-cache benchmarks or long-lived
    interactive sessions under memory pressure."""
    canonical_expand.cache_clear()
    cached_poly.cache_clear()
    _cached_affine.cache_clear()
    structural_coordinate.cache_clear()
    from .fast_rules import _recognize_fast_rules, _rule_profile

    _rule_profile.cache_clear()
    _recognize_fast_rules.cache_clear()
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
        "affine_coefficients": _cached_affine.cache_info(),
        "structural_coordinates": structural_coordinate.cache_info(),
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
    from .fast_rules import _recognize_fast_rules, _rule_profile

    info["structural_profiles"] = _rule_profile.cache_info()
    info["structural_recognition"] = _recognize_fast_rules.cache_info()
    return info
