from __future__ import annotations
import sympy as sp


def elimination_polynomials(equations, eliminate_vars, keep_vars):
    try:
        G = sp.groebner(
            list(equations), *(list(eliminate_vars) + list(keep_vars)), order="lex"
        )
        out = []
        seen = set()
        for g in G:
            e = sp.expand(g)
            if e != 0 and not e.free_symbols.intersection(eliminate_vars):
                k = sp.srepr(e)
                if k not in seen:
                    seen.add(k)
                    out.append(e)
        return out
    except (TypeError, ValueError, ArithmeticError, NotImplementedError):
        return []


def solve_with_elimination(equations, unknowns):
    try:
        sols = sp.solve(list(equations), list(unknowns), dict=True, simplify=True)
        if sols:
            return sols
    except (TypeError, ValueError, ArithmeticError, NotImplementedError):
        pass
    # If direct symbolic solving fails, eliminate all but one variable and recover solutions from the reduced system.
    if len(unknowns) >= 2:
        reduced = elimination_polynomials(equations, unknowns[1:], [unknowns[0]])
        if reduced:
            try:
                roots = sp.solve(reduced, [unknowns[0]], dict=True, simplify=True)
                return roots
            except (TypeError, ValueError, ArithmeticError, NotImplementedError):
                return []
    return []
