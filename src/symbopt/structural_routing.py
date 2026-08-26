from __future__ import annotations

import sympy as sp

from .linear_fractional import try_linear_fractional
from .precision import float64_backends_allowed
from .transcendental_analysis import try_asymptotic_solver, try_transcendental
from .trigonometric_optimization import try_exact_periodic_trig


def try_structural_solver(objective, constraints, variables, working_precision=None):
    """Try specialized global solvers in inexpensive, conservative order."""
    solvers = (
        try_linear_fractional,
        try_exact_periodic_trig,
        try_transcendental,
        try_asymptotic_solver,
    )
    for solver in solvers:
        if solver is try_linear_fractional and not float64_backends_allowed(
            objective, constraints, working_precision
        ):
            continue
        try:
            result = solver(objective, constraints, variables)
        except (
            ArithmeticError,
            ValueError,
            TypeError,
            NotImplementedError,
            sp.SympifyError,
        ):
            result = None
        if result is not None:
            return result
    return None
