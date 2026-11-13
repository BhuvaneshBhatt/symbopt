from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations

import sympy as sp
from sympy.polys.polyerrors import PolynomialError


@dataclass
class VariableInteractionGraph:
    """An undirected graph of variable or matrix-coordinate interactions."""

    variables: list[sp.Symbol]
    edges: set[tuple[sp.Symbol, sp.Symbol]]

    def neighbors(self, v: sp.Symbol) -> set[sp.Symbol]:
        """Return the adjacent vertices of the requested graph vertex."""
        out = set()
        for a, b in self.edges:
            if a == v:
                out.add(b)
            elif b == v:
                out.add(a)
        return out

    def connected_components(self) -> list[list[sp.Symbol]]:
        """Return the disjoint connected components of the stored interaction graph."""
        rem = set(self.variables)
        comps = []
        while rem:
            seed = rem.pop()
            stack = [seed]
            comp = {seed}
            while stack:
                u = stack.pop()
                for w in self.neighbors(u):
                    if w not in comp:
                        comp.add(w)
                        rem.discard(w)
                        stack.append(w)
            comps.append(sorted(comp, key=sp.sstr))
        return comps


@dataclass
class SymmetryReport:
    """Detected sign and permutation symmetries with their polynomial support metadata."""

    even_variables: list[sp.Symbol]
    permutation_pairs: list[tuple[sp.Symbol, sp.Symbol]]
    message: str

    def pretty(self) -> str:
        """Format the stored mathematical result and diagnostic metadata for inspection."""
        return "\n".join(
            [
                f"even_variables: {[sp.sstr(v) for v in self.even_variables]}",
                f"permutation_pairs: {[(sp.sstr(a), sp.sstr(b)) for a, b in self.permutation_pairs]}",
                f"message: {self.message}",
            ]
        )


def _as_exprs(exprs: sp.Expr | Iterable[sp.Expr]) -> tuple[sp.Expr, ...]:
    """Normalize one expression or an iterable of expressions to a tuple."""
    if isinstance(exprs, sp.Basic):
        return (exprs,)
    return tuple(exprs)


def _as_symbols(vars: sp.Symbol | Sequence[sp.Symbol]) -> list[sp.Symbol]:
    """Normalize one symbol or a sequence of symbols to a list."""
    if isinstance(vars, sp.Symbol):
        return [vars]
    return list(vars)


def build_sparsity_graph(
    exprs: sp.Expr | Iterable[sp.Expr],
    vars: sp.Symbol | Sequence[sp.Symbol],
) -> VariableInteractionGraph:
    """Build the variable-interaction graph induced by expression monomials.

    ``exprs`` may be either one SymPy expression or an iterable of expressions;
    ``vars`` may be one symbol or a sequence. Two variables are connected when
    they occur together in the support of a polynomial monomial. For
    nonpolynomial expressions, free-symbol co-occurrence is used as a
    conservative fallback.
    """
    exprs = _as_exprs(exprs)
    vars = _as_symbols(vars)
    edges: set[tuple[sp.Symbol, sp.Symbol]] = set()
    vset = set(vars)
    for expr in exprs:
        try:
            poly = sp.Poly(sp.expand(expr), *vars)
            monoms = poly.monoms()
            for monom in monoms:
                support = [v for v, e in zip(vars, monom) if e != 0]
                for a, b in combinations(sorted(support, key=sp.sstr), 2):
                    edges.add((a, b))
        except (
            PolynomialError,
            TypeError,
            ValueError,
            ArithmeticError,
            NotImplementedError,
        ):
            support = sorted(expr.free_symbols.intersection(vset), key=sp.sstr)
            for a, b in combinations(support, 2):
                edges.add((a, b))
    return VariableInteractionGraph(vars, edges)


def monomial_support(term: sp.Expr, vars: Sequence[sp.Symbol]) -> set[sp.Symbol]:
    """Extract exponent support in the specified polynomial coordinates."""
    try:
        mon = sp.Poly(term, *vars).monoms()[0]
        return {v for v, e in zip(vars, mon) if e != 0}
    except (
        PolynomialError,
        TypeError,
        ValueError,
        ArithmeticError,
        NotImplementedError,
    ):
        return set(term.free_symbols).intersection(set(vars))


def sparse_supported_basis(
    basis: Sequence[sp.Expr],
    components: Sequence[Sequence[sp.Symbol]],
    vars: Sequence[sp.Symbol],
) -> list[sp.Expr]:
    """Retain basis monomials supported inside the detected variable components."""
    comps = [set(c) for c in components]
    out = []
    for b in basis:
        supp = monomial_support(b, vars)
        if not supp or any(supp.issubset(c) for c in comps):
            out.append(b)
    return out if out else list(basis)


def detect_even_symmetry(
    exprs: Sequence[sp.Expr], vars: Sequence[sp.Symbol]
) -> list[sp.Symbol]:
    """Find coordinates whose exponents are even in every supplied polynomial term."""
    out = []
    for v in vars:
        ok = True
        for expr in exprs:
            try:
                if sp.expand(expr.subs(v, -v) - expr) != 0:
                    ok = False
                    break
            except (
                PolynomialError,
                TypeError,
                ValueError,
                ArithmeticError,
                NotImplementedError,
            ):
                ok = False
                break
        if ok:
            out.append(v)
    return out


def find_swap_symmetries(
    exprs: Sequence[sp.Expr], vars: Sequence[sp.Symbol]
) -> list[tuple[sp.Symbol, sp.Symbol]]:
    """Find supported coordinate transpositions that preserve the supplied polynomials."""
    pairs = []
    for a, b in combinations(vars, 2):
        swap = {a: b, b: a}
        ok = True
        for expr in exprs:
            try:
                if sp.expand(expr.xreplace(swap) - expr) != 0:
                    ok = False
                    break
            except (
                PolynomialError,
                TypeError,
                ValueError,
                ArithmeticError,
                NotImplementedError,
            ):
                ok = False
                break
        if ok:
            pairs.append((a, b))
    return pairs


def build_symmetry_report(
    objective: sp.Expr,
    constraints: Sequence[sp.Rel] | sp.Rel | sp.Symbol | Sequence[sp.Symbol] = (),
    vars: sp.Symbol | Sequence[sp.Symbol] | None = None,
) -> SymmetryReport:
    """Detect lightweight sign and pairwise-permutation symmetries.

    For an unconstrained problem, ``build_symmetry_report(objective, vars)`` is
    accepted as a shorthand. A single relational constraint may also be passed
    directly instead of wrapping it in a list.
    """
    if vars is None:
        vars = constraints
        constraints = ()
    vars = _as_symbols(vars)
    if isinstance(constraints, sp.Rel):
        constraints = (constraints,)
    else:
        constraints = tuple(constraints)
    exprs = [objective] + [
        sp.expand(c.lhs - c.rhs)
        for c in constraints
        if hasattr(c, "lhs") and hasattr(c, "rhs")
    ]
    even_vars = detect_even_symmetry(exprs, vars)
    perm_pairs = find_swap_symmetries(exprs, vars)
    return SymmetryReport(
        even_variables=even_vars,
        permutation_pairs=perm_pairs,
        message="Detected lightweight sign/permutation symmetries.",
    )


def reduce_sign_symmetry(
    objective: sp.Expr,
    constraints: Sequence[sp.Rel],
    vars: Sequence[sp.Symbol],
    box_bounds=None,
):
    """Apply the supported square-coordinate reduction for detected even symmetries."""
    exprs = [objective] + [
        sp.expand(c.lhs - c.rhs)
        for c in constraints
        if hasattr(c, "lhs") and hasattr(c, "rhs")
    ]
    even_vars = detect_even_symmetry(exprs, vars)
    added_constraints = []
    new_bounds = dict(box_bounds or {})
    for v in even_vars:
        if v in new_bounds:
            a, b = new_bounds[v]
            if abs(a + b) <= 1e-12:
                new_bounds[v] = (0.0, max(abs(a), abs(b)))
                added_constraints.append(sp.Ge(v, 0))
    return (
        added_constraints,
        new_bounds,
        build_symmetry_report(objective, constraints, vars),
    )
