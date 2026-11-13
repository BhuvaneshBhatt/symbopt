from __future__ import annotations

from functools import lru_cache
from math import gcd, lcm

import sympy as sp

from .structural_results import StructuralGlobalResult
from .univariate_analysis import _range_result, analyze_univariate

_TRIG_FUNCS = (sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc)
_FORBIDDEN_FUNCS = (
    sp.exp,
    sp.log,
    sp.sinh,
    sp.cosh,
    sp.tanh,
    sp.asin,
    sp.acos,
    sp.atan,
)
_SYMBOLIC_ERRORS = (
    TypeError,
    ValueError,
    NotImplementedError,
    sp.PolynomialError,
    sp.polys.polyerrors.CoercionFailed,
)


def _trig_frequency_data(objective, variable):
    """Return the exact common angular frequency and harmonic data.

    All trigonometric atoms must have affine arguments with nonzero rational
    slopes.  Constant phases may be any exact constants whose sine and cosine
    simplify to algebraic numbers.  This includes rational multiples of pi and
    many radical/algebraic phase descriptions while keeping the transformed
    rational function in an exact algebraic number field.
    """
    atoms = tuple(objective.atoms(*_TRIG_FUNCS))
    if not atoms or objective.has(*_FORBIDDEN_FUNCS):
        return None
    slopes = []
    phases = []
    for atom in atoms:
        arg = sp.expand(atom.args[0])
        try:
            slope = sp.simplify(sp.diff(arg, variable))
            affine = sp.simplify(sp.diff(arg, variable, 2)) == 0
        except _SYMBOLIC_ERRORS:
            return None
        if not affine or slope == 0 or slope.is_Rational is not True:
            return None
        phase = sp.simplify(arg - slope * variable)
        if phase.has(variable):
            return None
        try:
            sine_phase = sp.trigsimp(sp.expand_trig(sp.sin(phase)))
            cosine_phase = sp.trigsimp(sp.expand_trig(sp.cos(phase)))
        except _SYMBOLIC_ERRORS:
            return None
        if sine_phase.is_algebraic is not True or cosine_phase.is_algebraic is not True:
            return None
        slopes.append(sp.Rational(slope))
        phases.append(phase)

    numerators = [abs(int(value.p)) for value in slopes]
    denominators = [int(value.q) for value in slopes]
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
    if max(abs(value) for value in harmonics) > 12:
        return None
    return omega, harmonics, tuple(phases)


def _rewrite_reciprocal_trig(expr):
    replacements = {}
    for atom in expr.atoms(sp.tan, sp.cot, sp.sec, sp.csc):
        arg = atom.args[0]
        if atom.func is sp.tan:
            replacements[atom] = sp.sin(arg) / sp.cos(arg)
        elif atom.func is sp.cot:
            replacements[atom] = sp.cos(arg) / sp.sin(arg)
        elif atom.func is sp.sec:
            replacements[atom] = 1 / sp.cos(arg)
        else:
            replacements[atom] = 1 / sp.sin(arg)
    return expr.xreplace(replacements)


def _exact_poly(expr, variable):
    try:
        return sp.Poly(expr, variable, domain="QQ")
    except _SYMBOLIC_ERRORS:
        try:
            return sp.Poly(expr, variable, extension=True)
        except _SYMBOLIC_ERRORS:
            return None


def _half_angle_rational(objective, variable, omega):
    theta = sp.Dummy("_theta", real=True)
    tangent = sp.Dummy("_t", real=True)
    scaled = _rewrite_reciprocal_trig(objective)
    scaled = scaled.subs(variable, theta / omega)
    try:
        scaled = sp.expand_trig(scaled)
        transformed = scaled.subs(
            {
                sp.sin(theta): 2 * tangent / (1 + tangent**2),
                sp.cos(theta): (1 - tangent**2) / (1 + tangent**2),
            }
        )
        transformed = sp.factor(sp.cancel(sp.together(transformed)))
    except _SYMBOLIC_ERRORS:
        return None
    if transformed.has(*_TRIG_FUNCS, theta):
        return None
    numerator, denominator = sp.fraction(transformed)
    numerator_poly = _exact_poly(numerator, tangent)
    denominator_poly = _exact_poly(denominator, tangent)
    if numerator_poly is None or denominator_poly is None:
        return None
    coefficient_exprs = (*numerator_poly.all_coeffs(), *denominator_poly.all_coeffs())
    if any(coefficient.is_algebraic is not True for coefficient in coefficient_exprs):
        return None
    if denominator_poly.is_zero:
        return None
    return tangent, transformed, numerator_poly, denominator_poly


def _periodic_representative(base_point, period, domain, *, closure=False):
    target = domain.closure if closure and isinstance(domain, sp.Set) else domain
    if target is sp.S.Reals:
        return sp.simplify(base_point)
    if isinstance(target, sp.Interval):
        try:
            base_n = float(sp.N(base_point, 40))
            period_n = float(sp.N(period, 40))
            lo_n = (
                float("-inf")
                if target.start is -sp.oo
                else float(sp.N(target.start, 40))
            )
            hi_n = float("inf") if target.end is sp.oo else float(sp.N(target.end, 40))
        except (TypeError, ValueError, OverflowError):
            return None
        import math

        k_lo = -3 if math.isinf(lo_n) else math.ceil((lo_n - base_n) / period_n) - 1
        k_hi = 3 if math.isinf(hi_n) else math.floor((hi_n - base_n) / period_n) + 1
        if k_hi - k_lo > 10000:
            k_lo, k_hi = -3, 3
        for integer in range(k_lo, k_hi + 1):
            candidate = sp.simplify(base_point + integer * period)
            contains = target.contains(candidate)
            if contains is sp.S.true or contains is True:
                return candidate
    return None


def _real_roots(poly):
    try:
        roots = poly.all_roots()
    except _SYMBOLIC_ERRORS:
        return None
    real_roots = []
    for root in roots:
        if root.is_real is True:
            if not any(root == seen for seen in real_roots):
                real_roots.append(root)
        elif root.is_real is None:
            return None
    return tuple(real_roots)


def _pole_limits(transformed, tangent, root):
    limits = []
    for direction in ("-", "+"):
        try:
            value = sp.simplify(sp.limit(transformed, tangent, root, dir=direction))
        except _SYMBOLIC_ERRORS:
            return None
        if value not in (-sp.oo, sp.oo):
            return None
        limits.append(value)
    return tuple(limits)


def _infinity_limits(transformed, tangent):
    values = []
    for infinity in (-sp.oo, sp.oo):
        try:
            value = sp.simplify(sp.limit(transformed, tangent, infinity))
        except _SYMBOLIC_ERRORS:
            return None
        values.append(value)
    return tuple(values)


def _contains_interior(domain, point):
    if domain is sp.S.Reals:
        return True
    if not isinstance(domain, sp.Interval):
        return False
    contains = domain.contains(point)
    if contains is not sp.S.true and contains is not True:
        return False
    return point != domain.start and point != domain.end


def _candidate_boundary_values(objective, variable, domain):
    if not isinstance(domain, sp.Interval):
        return []
    candidates = []
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
        except _SYMBOLIC_ERRORS:
            continue
        if value in (sp.zoo, sp.nan):
            continue
        candidates.append((value, None if is_open else endpoint, "boundary"))
    return candidates


def _exact_best(candidates):
    if not candidates:
        return None
    try:
        return min(candidates, key=lambda item: item[0])
    except (TypeError, ValueError):
        return None


def _same_exact_value(first, second):
    if first == second:
        return True
    try:
        numerical_gap = abs(float(sp.N(first - second, 40)))
    except (TypeError, ValueError, OverflowError):
        numerical_gap = 0.0
    if numerical_gap > 1e-20:
        return False
    return sp.simplify(first - second) == 0


@lru_cache(maxsize=128)
def _exact_trig_result(objective, constraints, variable, domain, analysis=None):
    """Optimize a supported commensurate-frequency rational trig expression exactly."""
    data = _trig_frequency_data(objective, variable)
    if data is None:
        return None
    omega, harmonics, phases = data
    transformed_data = _half_angle_rational(objective, variable, omega)
    if transformed_data is None:
        return None
    tangent, transformed, numerator_poly, denominator_poly = transformed_data
    period = sp.simplify(2 * sp.pi / omega)

    denominator_roots = _real_roots(denominator_poly)
    if denominator_roots is None:
        return None
    pole_records = []
    negative_infinity = False
    for root in denominator_roots:
        base_point = sp.simplify(2 * sp.atan(root) / omega)
        point = _periodic_representative(base_point, period, domain)
        if point is None or not _contains_interior(domain, point):
            continue
        limits = _pole_limits(transformed, tangent, root)
        if limits is None:
            return None
        pole_records.append((root, point, limits))
        if -sp.oo in limits:
            negative_infinity = True

    infinity_limits = _infinity_limits(transformed, tangent)
    infinity_point = _periodic_representative(sp.pi / omega, period, domain)
    infinity_is_pole = infinity_limits is not None and any(
        value in (-sp.oo, sp.oo) for value in infinity_limits
    )
    if (
        infinity_is_pole
        and infinity_point is not None
        and _contains_interior(domain, infinity_point)
        and -sp.oo in infinity_limits
    ):
        negative_infinity = True

    boundary_candidates = _candidate_boundary_values(objective, variable, domain)
    if any(value == -sp.oo for value, _, _ in boundary_candidates):
        negative_infinity = True

    if negative_infinity:
        return StructuralGlobalResult(
            kind="rational_frequency_trig",
            value=-sp.oo,
            points=(),
            attained=False,
            certified=True,
            metadata={
                "common_frequency": sp.sstr(omega),
                "harmonics": harmonics,
                "phases": tuple(sp.sstr(phase) for phase in phases),
                "period": sp.sstr(period),
                "transformation": "tangent-half-angle",
                "rational_trig": True,
                "denominator": sp.sstr(denominator_poly.as_expr()),
                "real_denominator_roots": tuple(
                    sp.sstr(root) for root in denominator_roots
                ),
                "pole_limits": tuple(
                    (
                        sp.sstr(root),
                        sp.sstr(point),
                        tuple(sp.sstr(value) for value in limits),
                    )
                    for root, point, limits in pole_records
                ),
                "infinity_limits": None
                if infinity_limits is None
                else tuple(sp.sstr(value) for value in infinity_limits),
                "domain": sp.sstr(domain),
                "property_period": sp.sstr(analysis.period)
                if analysis is not None
                else "None",
                "providers": analysis.provider_summary if analysis is not None else {},
            },
        )

    derivative_num = sp.factor(sp.fraction(sp.cancel(sp.diff(transformed, tangent)))[0])
    derivative_poly = _exact_poly(derivative_num, tangent)
    if derivative_poly is None:
        return None
    stationary_roots = _real_roots(derivative_poly)
    if stationary_roots is None:
        return None

    candidates = []
    for root in stationary_roots:
        if denominator_poly.eval(root) == 0:
            continue
        base_point = sp.simplify(2 * sp.atan(root) / omega)
        point = _periodic_representative(base_point, period, domain)
        if point is None:
            continue
        try:
            value = sp.simplify(transformed.subs(tangent, root))
        except _SYMBOLIC_ERRORS:
            continue
        if value.is_real is False or value in (sp.zoo, sp.nan, sp.oo, -sp.oo):
            continue
        candidates.append((value, point, "stationary"))

    if (
        infinity_limits is not None
        and all(value not in (sp.oo, -sp.oo) for value in infinity_limits)
        and len(infinity_limits) == 2
        and _same_exact_value(*infinity_limits)
    ):
        point = _periodic_representative(sp.pi / omega, period, domain)
        if point is not None:
            candidates.append((sp.simplify(infinity_limits[0]), point, "phase-pi"))

    candidates.extend(boundary_candidates)
    candidates = [candidate for candidate in candidates if candidate[0] != sp.oo]
    best = _exact_best(candidates)
    if best is None:
        return None
    best_value = sp.simplify(best[0])
    points = tuple(
        {variable: sp.simplify(point)}
        for value, point, _ in candidates
        if point is not None and _same_exact_value(value, best_value)
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
            "phases": tuple(sp.sstr(phase) for phase in phases),
            "period": sp.sstr(period),
            "stationary_polynomial": sp.sstr(derivative_poly.as_expr()),
            "stationary_root_count": len(stationary_roots),
            "transformation": "tangent-half-angle",
            "rational_trig": True,
            "numerator": sp.sstr(numerator_poly.as_expr()),
            "denominator": sp.sstr(denominator_poly.as_expr()),
            "real_denominator_roots": tuple(
                sp.sstr(root) for root in denominator_roots
            ),
            "pole_limits": tuple(
                (
                    sp.sstr(root),
                    sp.sstr(point),
                    tuple(sp.sstr(value) for value in limits),
                )
                for root, point, limits in pole_records
            ),
            "infinity_limits": None
            if infinity_limits is None
            else tuple(sp.sstr(value) for value in infinity_limits),
            "domain": sp.sstr(domain),
            "property_period": sp.sstr(analysis.period)
            if analysis is not None
            else "None",
            "providers": analysis.provider_summary if analysis is not None else {},
        },
    )


def try_exact_periodic_trig(objective, constraints, variables, analysis=None):
    """Find exact extrema of supported commensurate trigonometric expressions via rational coordinates."""
    variables = tuple(variables)
    if len(variables) != 1:
        return None
    variable = variables[0]
    analysis = analysis or analyze_univariate(objective, constraints, variable)
    if analysis is None:
        return None
    domain = analysis.requested_domain
    exact = _exact_trig_result(
        objective, tuple(constraints), variable, domain, analysis=analysis
    )
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
    return _range_result(
        "periodic_trig",
        objective,
        constraints,
        variable,
        analysis=analysis,
    )
