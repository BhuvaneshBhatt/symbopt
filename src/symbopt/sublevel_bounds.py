"""Exact boxes containing improving sublevel sets of structured polynomials."""

from dataclasses import dataclass
from functools import cmp_to_key

import sympy as sp

from .algebraic_numbers import is_fixed_real_algebraic
from .input_normalization import normalize_variables
from .rational_optimization import _compare, _polynomial


def _majorant(objective, variables):
    """Bound the leading form below using weighted arithmetic-geometric means.

    An indefinite mixed monomial of degree m costs |c|*alpha_i/m from each
    pure even power x_i**m. Nonnegative even monomials can simply be dropped.
    Strictly positive remaining margins imply coercivity. For r=max|x_i|>=1,
    all lower terms are bounded below by -B*r**(m-1).
    """
    if objective.has(sp.Float) or objective.free_symbols - set(variables):
        return None
    poly = _polynomial(objective, variables)
    degree = poly.total_degree()
    if degree < 2 or degree % 2:
        return None
    margins = [poly.coeff_monomial(v**degree) for v in variables]
    remainder = sp.S.Zero
    for powers, coefficient in poly.terms():
        total = sum(powers)
        if total < degree:
            remainder += abs(coefficient)
        elif max(powers) != degree:
            if all(p % 2 == 0 for p in powers) and _compare(coefficient, 0) >= 0:
                continue
            for i, power in enumerate(powers):
                margins[i] -= abs(coefficient) * sp.Rational(power, degree)
    margins = tuple(sp.simplify(value) for value in margins)
    if any(_compare(value, 0) <= 0 for value in margins):
        return None
    return degree, margins, remainder


@dataclass(frozen=True)
class _SublevelCertificate:
    """A replayable sufficient proof that f(x)>U outside a finite box."""

    objective: sp.Expr
    variables: tuple
    upper_bound: sp.Expr
    radius: sp.Integer
    degree: int
    leading_margins: tuple
    remainder_bound: sp.Expr

    @property
    def bounds(self):
        """Finite coordinate bounds justified by the polynomial sublevel certificate."""
        return {v: (-self.radius, self.radius) for v in self.variables}

    def verify(self):
        """Recompute the polynomial bound and check the strict exterior inequality."""
        try:
            data = _majorant(self.objective, self.variables)
            if data != (self.degree, self.leading_margins, self.remainder_bound):
                return False
            if (
                self.upper_bound.has(sp.Float)
                or self.upper_bound.free_symbols
                or self.upper_bound.is_real is not True
                or not is_fixed_real_algebraic(self.upper_bound)
            ):
                return False
            if self.radius.is_Integer is not True or self.radius < 1:
                return False
            margin = min(self.leading_margins, key=cmp_to_key(_compare))
            return (
                _compare(
                    margin * self.radius - self.remainder_bound, abs(self.upper_bound)
                )
                > 0
            )
        except (sp.PolynomialError, NotImplementedError, TypeError, ValueError):
            return False


def _sublevel_certificate(objective, variables, upper_bound):
    try:
        objective, upper_bound = map(sp.sympify, (objective, upper_bound))
        if (
            upper_bound.free_symbols
            or upper_bound.has(sp.Float)
            or upper_bound.is_real is not True
            or not is_fixed_real_algebraic(upper_bound)
        ):
            return None
        data = _majorant(objective, variables)
        if data is None:
            return None
        degree, margins, remainder = data
        margin = min(margins, key=cmp_to_key(_compare))
        threshold = sp.ceiling(1 + (remainder + abs(upper_bound) + 1) / margin)
        if not isinstance(threshold, sp.Integer):
            return None
        # Powers of two are also exact float64 endpoints for the interval engine.
        radius = sp.Integer(2 ** int(threshold).bit_length())
        certificate = _SublevelCertificate(
            objective,
            tuple(variables),
            upper_bound,
            radius,
            degree,
            margins,
            remainder,
        )
        return certificate if certificate.verify() else None
    except (sp.PolynomialError, NotImplementedError, TypeError, ValueError):
        return None


def polynomial_sublevel_bounds(objective, variables, upper_bound):
    """Return an exact box containing {x: objective(x)<=upper_bound}, or None.

    Supported polynomials have fixed real algebraic coefficients and an even
    leading degree with strictly positive pure-power margins after bounding
    mixed terms. Outside the box the objective is strictly greater than the
    supplied level. Using the box for optimization additionally requires a
    feasible point whose objective is at most that level. A refusal does not
    establish noncoercivity or unboundedness.
    """
    variables = tuple(normalize_variables(variables) or ())
    if not variables:
        raise ValueError("provide at least one decision variable")
    certificate = _sublevel_certificate(objective, variables, upper_bound)
    return None if certificate is None else certificate.bounds


def restrict_polynomial_sublevel(problem, bounds, candidate=None):
    """Restrict interval search only after checking an exact feasible upper bound."""
    variables = tuple(problem.original_variables)
    if (
        not variables
        or problem.auxiliary_variables
        or problem.parameter_symbols
        or all(v in bounds for v in variables)
        or any(r.has(sp.Float) for r in problem.original_constraints)
    ):
        return bounds, None
    points = []
    if candidate is not None:
        points.append(candidate.original_variable_values or candidate.primal_values)
    points.extend(dict.fromkeys(variables, sp.Integer(value)) for value in (0, 1, -1))
    for point in points:
        if not all(v in point for v in variables):
            continue
        if any(
            sp.sympify(point[v]).has(sp.Float)
            or sp.sympify(point[v]).free_symbols
            or sp.sympify(point[v]).is_real is not True
            for v in variables
        ):
            continue
        if not all(
            sp.simplify(r.subs(point, simultaneous=True)) is sp.true
            for r in problem.original_constraints
        ):
            continue
        if any(not (lo <= point[v] <= hi) for v, (lo, hi) in bounds.items()):
            continue
        upper = problem.original_objective.subs(point, simultaneous=True)
        certificate = _sublevel_certificate(
            problem.original_objective, variables, upper
        )
        if certificate is None or certificate.radius > 2**1022:
            continue
        restricted = certificate.bounds
        for variable, (left, right) in bounds.items():
            restricted[variable] = (
                max(left, -certificate.radius),
                min(right, certificate.radius),
            )
        payload = {**vars(certificate), "feasible_point": dict(point)}
        return restricted, payload
    return bounds, None
