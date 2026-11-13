from __future__ import annotations

import sympy as sp

from .linear_fractional import try_linear_fractional
from .mixed_periodic import try_mixed_periodic
from .precision import float64_backends_allowed
from .quadratic_optimization import try_parametric_quadratic
from .rational_optimization import try_univariate_rational
from .transcendental_analysis import try_asymptotic_solver, try_transcendental
from .trigonometric_optimization import try_exact_periodic_trig
from .univariate_analysis import analyze_univariate


def try_structural_solver(objective, constraints, variables, working_precision=None):
    """Try specialized global solvers in inexpensive, conservative order."""
    variables = tuple(variables)
    quadratic = try_parametric_quadratic(objective, constraints, variables)
    if quadratic is not None:
        return quadratic
    analysis = (
        analyze_univariate(objective, constraints, variables[0])
        if len(variables) == 1
        else None
    )
    solvers = (
        (try_linear_fractional, False),
        (try_univariate_rational, False),
        (try_exact_periodic_trig, True),
        (try_mixed_periodic, True),
        (try_transcendental, True),
        (try_asymptotic_solver, True),
    )
    for solver, uses_analysis in solvers:
        if solver is try_linear_fractional and not float64_backends_allowed(
            objective, constraints, working_precision
        ):
            continue
        try:
            if solver is try_univariate_rational:
                # Keep direct LP/QP postprocessing for affine closed constraints.
                try:
                    simple = sp.Poly(objective, *variables).total_degree() <= 2 and all(
                        isinstance(r, (sp.Equality, sp.GreaterThan, sp.LessThan))
                        and sp.Poly(r.lhs - r.rhs, *variables).total_degree() <= 1
                        for r in constraints
                    )
                except sp.PolynomialError:
                    simple = False
                if simple:
                    continue
            result = (
                solver(objective, constraints, variables, analysis=analysis)
                if uses_analysis
                else solver(objective, constraints, variables)
            )
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
