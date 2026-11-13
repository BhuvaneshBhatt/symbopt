from __future__ import annotations

from dataclasses import dataclass

import sympy as sp
from sympy.polys.numberfields import to_number_field

from .structural_results import StructuralGlobalResult
from .univariate_analysis import analyze_univariate


@dataclass(frozen=True)
class MixedPeriodicDecomposition:
    """Exact decomposition of a univariate expression into periodic components."""

    outer: sp.Expr
    components: tuple[sp.Expr, ...]
    coordinates: tuple[sp.Symbol, ...]
    periods: tuple[sp.Expr, ...]


def _period(expr: sp.Expr, variable: sp.Symbol) -> sp.Expr | None:
    analysis = analyze_univariate(expr, (), variable)
    return analysis.period if analysis is not None else None


def _commensurable(first: sp.Expr, second: sp.Expr) -> bool | None:
    ratio = sp.simplify(first / second)
    if ratio.is_Rational is True:
        return True
    if ratio.is_Rational is False:
        return False
    return None


def _group_periodic_args(expr: sp.Expr, variable: sp.Symbol) -> tuple[sp.Expr, ...]:
    """Combine directly commensurate Add/Mul arguments before decomposition."""
    if not (expr.is_Add or expr.is_Mul):
        return expr.args
    constants = [arg for arg in expr.args if not arg.has(variable)]
    varying = [arg for arg in expr.args if arg.has(variable)]
    periodic: list[tuple[sp.Expr, sp.Expr]] = []
    unresolved: list[sp.Expr] = []
    for arg in varying:
        period = _period(arg, variable)
        if period is None:
            unresolved.append(arg)
        else:
            periodic.append((arg, period))

    groups: list[list[tuple[sp.Expr, sp.Expr]]] = []
    for item in periodic:
        for group in groups:
            relation = _commensurable(item[1], group[0][1])
            if relation is True:
                group.append(item)
                break
        else:
            groups.append([item])

    combined = [expr.func(*(item[0] for item in group)) for group in groups]
    return tuple(constants + unresolved + combined)


def _decompose_node(
    expr: sp.Expr,
    variable: sp.Symbol,
    coordinates: list[sp.Symbol],
    components: list[sp.Expr],
    periods: list[sp.Expr],
) -> sp.Expr | None:
    if not expr.has(variable):
        return expr
    period = _period(expr, variable)
    if period is not None:
        coordinate = sp.Dummy(f"_mp{len(coordinates)}", real=True)
        coordinates.append(coordinate)
        components.append(expr)
        periods.append(period)
        return coordinate
    if not expr.args:
        return None

    args = _group_periodic_args(expr, variable)
    transformed = []
    for arg in args:
        mapped = _decompose_node(arg, variable, coordinates, components, periods)
        if mapped is None:
            return None
        transformed.append(mapped)
    try:
        return expr.func(*transformed)
    except (TypeError, ValueError):
        return None


def _q_linearly_independent(values: tuple[sp.Expr, ...]) -> bool:
    """Certify Q-linear independence of exact values through a common number field."""
    if len(values) < 2 or any(value == 0 for value in values):
        return False
    scale = values[0]
    ratios = tuple(sp.simplify(value / scale) for value in values)
    if not all(ratio.is_algebraic is True for ratio in ratios):
        return False
    try:
        primitive = to_number_field(ratios)
        degree = sp.Poly(primitive.minpoly).degree()
        rows = []
        for ratio in ratios:
            algebraic = to_number_field(ratio, primitive)
            coeffs = list(algebraic.native_coeffs())
            coeffs = [sp.Rational(value) for value in coeffs]
            rows.append([sp.S.Zero] * (degree - len(coeffs)) + coeffs)
        return sp.Matrix(rows).rank() == len(values)
    except (NotImplementedError, TypeError, ValueError, sp.PolynomialError):
        return False


def decompose_mixed_periodic(
    objective: sp.Expr, variable: sp.Symbol
) -> MixedPeriodicDecomposition | None:
    """Return a certified incommensurate-period decomposition when recognizable."""
    objective = sp.sympify(objective)
    variable = sp.sympify(variable)
    if _period(objective, variable) is not None:
        return None
    coordinates: list[sp.Symbol] = []
    components: list[sp.Expr] = []
    periods: list[sp.Expr] = []
    outer = _decompose_node(objective, variable, coordinates, components, periods)
    if outer is None or len(components) < 2:
        return None

    inverse_periods = tuple(sp.simplify(1 / period) for period in periods)
    if not _q_linearly_independent(inverse_periods):
        return None
    return MixedPeriodicDecomposition(
        sp.simplify(outer), tuple(components), tuple(coordinates), tuple(periods)
    )


def _set_condition(value: sp.Symbol, image: sp.Set) -> sp.Expr | None:
    if image is sp.S.Reals:
        return sp.S.true
    if isinstance(image, sp.Interval):
        lower = (
            sp.S.true
            if image.start is -sp.oo
            else (value > image.start if image.left_open else value >= image.start)
        )
        upper = (
            sp.S.true
            if image.end is sp.oo
            else (value < image.end if image.right_open else value <= image.end)
        )
        return sp.And(lower, upper)
    if isinstance(image, sp.FiniteSet):
        return sp.Or(*(sp.Eq(value, item) for item in image))
    if isinstance(image, sp.Union):
        pieces = [_set_condition(value, part) for part in image.args]
        if any(piece is None for piece in pieces):
            return None
        return sp.Or(*pieces)
    return None


def _component_range(component: sp.Expr, variable: sp.Symbol, value: sp.Symbol):
    analysis = analyze_univariate(component, (), variable)
    if analysis is None:
        return None
    # Keep commensurate trigonometric extrema in the rational half-angle
    # coordinate. Expanding them back into trig functions of RootOf values can
    # turn later endpoint comparisons into expensive transcendental queries.
    from .trigonometric_optimization import try_exact_periodic_trig

    lower = try_exact_periodic_trig(component, (), (variable,), analysis=analysis)
    if lower is not None and lower.attained is True:
        upper = try_exact_periodic_trig(-component, (), (variable,))
        if upper is not None and upper.attained is True:
            return sp.And(value >= lower.value, value <= -upper.value)
    image = analysis.range_set
    if isinstance(image, sp.Set):
        condition = _set_condition(value, image)
        if condition is not None:
            return condition
    if analysis.range_condition is not None:
        range_var, condition = analysis.range_condition
        return sp.simplify(condition.xreplace({range_var: value}))
    return None


def _unbounded_interval(domain: sp.Set) -> bool:
    if domain is sp.S.Reals:
        return True
    return isinstance(domain, sp.Interval) and (
        domain.start is -sp.oo or domain.end is sp.oo
    )


def _common_preimage(
    decomposition: MixedPeriodicDecomposition,
    coordinate_values: dict[sp.Symbol, sp.Expr],
    variable: sp.Symbol,
    domain: sp.Set,
):
    equations = [
        sp.Eq(component, coordinate_values[coordinate])
        for component, coordinate in zip(
            decomposition.components, decomposition.coordinates, strict=True
        )
        if coordinate in coordinate_values
    ]
    if len(equations) != len(decomposition.components):
        return ()
    try:
        solutions = sp.solve(equations, [variable], dict=True)
    except (NotImplementedError, TypeError, ValueError):
        return ()
    points = []
    for solution in solutions:
        if variable not in solution:
            continue
        point = sp.simplify(solution[variable])
        contains = domain.contains(point)
        if contains is sp.S.true or contains is True:
            points.append({variable: point})
    return tuple(points)


def try_mixed_periodic(objective, constraints, variables, analysis=None):
    """Optimize over the dense product range of incommensurate periodic components."""
    variables = tuple(variables)
    if len(variables) != 1:
        return None
    variable = variables[0]
    analysis = analysis or analyze_univariate(objective, constraints, variable)
    if analysis is None:
        return None
    domain = analysis.requested_domain
    if not _unbounded_interval(domain):
        return None

    decomposition = decompose_mixed_periodic(objective, variable)
    if decomposition is None:
        return None
    for component in decomposition.components:
        component_analysis = analyze_univariate(component, (), variable)
        if (
            component_analysis is None
            or component_analysis.natural_domain != sp.S.Reals
        ):
            return None
        if component_analysis.discontinuities:
            return None

    range_constraints = []
    for component, coordinate in zip(
        decomposition.components, decomposition.coordinates, strict=True
    ):
        condition = _component_range(component, variable, coordinate)
        if condition is None:
            return None
        range_constraints.append(condition)

    # Import lazily to avoid a module cycle: structural routing calls this function.
    from .solver_pipeline import minimize_result

    reduced = minimize_result(
        decomposition.outer,
        constraints=range_constraints,
        variables=decomposition.coordinates,
        use_symbol_assumptions=False,
        _skip_monotone_reduction=False,
        _skip_decomposition=False,
    )
    reduced_value = reduced.optimum_value
    reduced_certified = bool(reduced.certified)
    certificate = reduced.certificate
    if (
        not reduced_certified
        and certificate is not None
        and certificate.status == "proven_optimal"
        and certificate.lower_bound is not None
        and certificate.upper_bound is not None
        and sp.simplify(certificate.lower_bound - certificate.upper_bound) == 0
    ):
        reduced_certified = True
        reduced_value = sp.simplify(certificate.lower_bound)
    if not reduced_certified or reduced_value is None:
        return None

    points = ()
    attained: bool | None = None
    if reduced.attained is True and reduced.best_candidate is not None:
        coordinate_values = reduced.best_candidate.original_variable_values
        points = _common_preimage(decomposition, coordinate_values, variable, domain)
        if points:
            attained = True

    return StructuralGlobalResult(
        kind="mixed_periodic",
        value=sp.simplify(reduced_value),
        points=points,
        attained=attained,
        certified=True,
        metadata={
            "components": tuple(sp.sstr(item) for item in decomposition.components),
            "periods": tuple(sp.sstr(item) for item in decomposition.periods),
            "inverse_periods_q_independent": True,
            "outer_problem": sp.sstr(decomposition.outer),
            "range_constraints": tuple(sp.sstr(item) for item in range_constraints),
            "reduced_method": reduced.method,
            "domain": sp.sstr(domain),
            "providers": analysis.provider_summary,
            "attainment": "proven" if attained is True else "unknown",
        },
    )
