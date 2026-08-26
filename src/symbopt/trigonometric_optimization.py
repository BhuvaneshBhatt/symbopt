from __future__ import annotations

from functools import lru_cache

import sympy as sp

from .structural_results import StructuralGlobalResult
from .univariate_analysis import _range_result, _univariate_domain


def _trig_frequency_data(objective, variable):
    """Return a common rational angular frequency for a trig polynomial.

    The route accepts expressions built algebraically from ``sin``/``cos`` of
    affine arguments whose variable slopes are nonzero rationals.  Constant
    phases are currently required to be integer multiples of ``pi``; this keeps
    the tangent-half-angle polynomial over QQ and makes the stationary-root
    enumeration exact and inexpensive.
    """
    atoms = tuple(objective.atoms(sp.sin, sp.cos))
    if not atoms or objective.has(sp.tan, sp.cot, sp.sec, sp.csc):
        return None
    forbidden = (
        sp.exp,
        sp.log,
        sp.sinh,
        sp.cosh,
        sp.tanh,
        sp.asin,
        sp.acos,
        sp.atan,
    )
    if objective.has(*forbidden):
        return None
    slopes = []
    for atom in atoms:
        arg = sp.expand(atom.args[0])
        slope = sp.simplify(sp.diff(arg, variable))
        if slope == 0 or slope.is_Rational is not True:
            return None
        if sp.diff(arg, variable, 2) != 0:
            return None
        phase = sp.simplify(arg - slope * variable)
        phase_over_pi = sp.simplify(phase / sp.pi)
        if phase_over_pi.is_Integer is not True:
            return None
        slopes.append(sp.Rational(slope))
    numerators = [abs(int(value.p)) for value in slopes]
    denominators = [int(value.q) for value in slopes]
    from math import gcd, lcm

    common_num = numerators[0]
    for value in numerators[1:]:
        common_num = gcd(common_num, value)
    common_den = denominators[0]
    for value in denominators[1:]:
        common_den = lcm(common_den, value)
    omega = sp.Rational(common_num, common_den)
    if omega <= 0:
        return None
    harmonics = tuple(int(sp.simplify(value / omega)) for value in slopes)
    # Expanding very high harmonics into a tangent-half-angle polynomial can
    # itself be exponential in expression size.  Leave such cases to interval
    # or other transcendental routes rather than creating an enormous exact
    # polynomial.
    if max(abs(value) for value in harmonics) > 12:
        return None
    return omega, harmonics


def _half_angle_rational(objective, variable, omega):
    theta = sp.Dummy("_theta", real=True)
    tangent = sp.Dummy("_t", real=True)
    scaled = sp.expand_trig(sp.expand(objective.subs(variable, theta / omega)))
    scaled = sp.expand_trig(scaled)
    transformed = scaled.subs(
        {
            sp.sin(theta): 2 * tangent / (1 + tangent**2),
            sp.cos(theta): (1 - tangent**2) / (1 + tangent**2),
        }
    )
    transformed = sp.factor(sp.cancel(sp.together(transformed)))
    if transformed.has(sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc, theta):
        return None
    numerator, denominator = sp.fraction(transformed)
    try:
        sp.Poly(numerator, tangent, domain="QQ")
        sp.Poly(denominator, tangent, domain="QQ")
    except (sp.PolynomialError, TypeError, ValueError):
        return None
    return tangent, transformed


def _periodic_representative(base_point, period, domain):
    if domain is sp.S.Reals:
        return sp.simplify(base_point)
    if isinstance(domain, sp.Interval):
        try:
            base_n = float(sp.N(base_point, 40))
            period_n = float(sp.N(period, 40))
            lo_n = (
                float("-inf")
                if domain.start is -sp.oo
                else float(sp.N(domain.start, 40))
            )
            hi_n = float("inf") if domain.end is sp.oo else float(sp.N(domain.end, 40))
        except (TypeError, ValueError, OverflowError):
            return None
        import math

        k_lo = -3 if math.isinf(lo_n) else math.ceil((lo_n - base_n) / period_n) - 1
        k_hi = 3 if math.isinf(hi_n) else math.floor((hi_n - base_n) / period_n) + 1
        if k_hi - k_lo > 10000:
            k_lo, k_hi = -3, 3
        for integer in range(k_lo, k_hi + 1):
            candidate = sp.simplify(base_point + integer * period)
            contains = domain.contains(candidate)
            if contains is sp.S.true or contains is True:
                return candidate
    return None


@lru_cache(maxsize=128)
def _exact_trig_result(objective, constraints, variable, domain):
    """Optimize a supported commensurate-frequency trigonometric polynomial exactly."""
    data = _trig_frequency_data(objective, variable)
    if data is None:
        return None
    omega, harmonics = data
    transformed_data = _half_angle_rational(objective, variable, omega)
    if transformed_data is None:
        return None
    tangent, transformed = transformed_data
    derivative_num = sp.factor(sp.fraction(sp.cancel(sp.diff(transformed, tangent)))[0])
    try:
        derivative_poly = sp.Poly(derivative_num, tangent, domain="QQ")
        roots = derivative_poly.all_roots()
    except (sp.PolynomialError, NotImplementedError, ValueError, TypeError):
        return None
    real_roots = [root for root in roots if root.is_real is not False]
    period = sp.simplify(2 * sp.pi / omega)
    candidates = []
    for root in real_roots:
        base_point = sp.simplify(2 * sp.atan(root) / omega)
        point = _periodic_representative(base_point, period, domain)
        if point is None:
            continue
        value = sp.simplify(transformed.subs(tangent, root))
        candidates.append((value, point, root))

    # t = +/- infinity is the theta = pi point omitted by tangent half-angle.
    try:
        infinity_value = sp.simplify(sp.limit(transformed, tangent, sp.oo))
    except (ValueError, TypeError, NotImplementedError):
        infinity_value = None
    if infinity_value is not None and infinity_value.is_real is not False:
        base_point = sp.simplify(sp.pi / omega)
        point = _periodic_representative(base_point, period, domain)
        if point is not None:
            candidates.append((infinity_value, point, sp.oo))

    # On an interval shorter than a period the boundary can beat all stationary
    # representatives.  Exact endpoint evaluation handles both open and closed
    # endpoints; open endpoint values are included as limits but not as points.
    if isinstance(domain, sp.Interval):
        for endpoint, direction, is_open in (
            (domain.start, "+", domain.left_open),
            (domain.end, "-", domain.right_open),
        ):
            if endpoint in (-sp.oo, sp.oo):
                continue
            try:
                value = (
                    sp.simplify(sp.limit(objective, variable, endpoint, dir=direction))
                    if is_open
                    else sp.simplify(objective.subs(variable, endpoint))
                )
            except (ValueError, TypeError, NotImplementedError):
                continue
            candidates.append((value, None if is_open else endpoint, None))

    if not candidates:
        return None
    try:
        # SymPy's algebraic comparisons are exact for the supported RootOf/QQ
        # values. If exact ordering cannot be decided, decline this certified
        # route rather than promoting a high-precision numerical comparison.
        best = min(candidates, key=lambda item: item[0])
    except (TypeError, ValueError):
        return None
    best_value = sp.simplify(best[0])
    points = tuple(
        {variable: sp.simplify(point)}
        for value, point, _ in candidates
        if point is not None and sp.simplify(value - best_value) == 0
    )
    attained = bool(points)
    return StructuralGlobalResult(
        kind="rational_frequency_trig",
        value=best_value,
        points=points,
        attained=attained,
        certified=True,
        metadata={
            "common_frequency": sp.sstr(omega),
            "harmonics": harmonics,
            "period": sp.sstr(period),
            "stationary_polynomial": sp.sstr(derivative_poly.as_expr()),
            "stationary_root_count": len(real_roots),
            "transformation": "tangent-half-angle",
            "domain": sp.sstr(domain),
        },
    )


def try_exact_periodic_trig(objective, constraints, variables):
    variables = tuple(variables)
    if len(variables) != 1:
        return None
    variable = variables[0]
    domain = _univariate_domain(constraints, variable)
    if domain is None:
        return None
    exact = _exact_trig_result(objective, tuple(constraints), variable, domain)
    if exact is not None:
        return exact
    atoms = tuple(objective.atoms(sp.sin, sp.cos))
    if not atoms:
        return None
    slopes = []
    for atom in atoms:
        arg = sp.expand(atom.args[0])
        slope = sp.diff(arg, variable)
        if slope == 0 or slope.free_symbols or sp.diff(arg, variable, 2) != 0:
            return None
        slopes.append(sp.Abs(slope))
    if not all(sp.simplify(slope - slopes[0]) == 0 for slope in slopes[1:]):
        return None
    return _range_result("periodic_trig", objective, constraints, variable, domain)
