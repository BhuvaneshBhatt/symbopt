"""Internal symbols kept distinct from user coordinates and parameters."""

import sympy as sp


def fresh_symbols(names, occupied, **assumptions):
    """Create deterministic scoped auxiliary symbols without capturing user coordinates."""
    used = {symbol.name for symbol in occupied}
    result = []
    for name in names:
        candidate = name
        suffix = 1
        while candidate in used:
            candidate = f"{name}_{suffix}"
            suffix += 1
        result.append(sp.Symbol(candidate, **assumptions))
        used.add(candidate)
    return tuple(result)
