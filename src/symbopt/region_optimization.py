"""Optimization on structural covers and rigorously pruned region components."""

from dataclasses import replace

import sympy as sp

from .input_normalization import normalize_bounds_input, normalize_constraint_branches
from .precision import outward_float
from .problem_models import ProofArtifact
from .structural_results import StructuralGlobalResult
from .verified_intervals import enclose_expression


def constraint_formula(constraints):
    """Form a Boolean domain from the existing conjunctive/disjunctive inputs."""
    return sp.Or(
        *(sp.And(*branch) for branch in normalize_constraint_branches(constraints))
    )


def _constraints(formula):
    return sp.Eq(0, 1, evaluate=False) if formula is sp.false else formula


def solve_cover(prepared, objective, constraints, bounds, assumptions, solve, merge):
    """Solve a full structural cover and lift exact results into ambient coordinates.

    No injectivity or Jacobian is required. Each chart includes its complete
    parameter domain, including singular images and boundaries. Only exact
    completed solves participate; otherwise the ordinary ambient route runs.
    """
    cover = prepared.cover
    if cover is None or not cover.exact or not cover.charts:
        return None
    variables = prepared.variables
    for chart in cover.charts:
        expressions = (
            *chart.mapping,
            chart.condition,
            *(e for _, lo, hi in chart.bounds for e in (lo, hi)),
        )
        if any(expression.has(sp.Float) for expression in expressions):
            return None
        coefficients = set().union(*(e.free_symbols for e in expressions)) - set(
            chart.parameters
        )
        if coefficients.intersection(variables):
            return None
    if all(len(chart.parameters) >= len(variables) for chart in cover.charts):
        from semialg.standard_regions import ParametricRegion, TransformedRegion

        if not isinstance(prepared.source, (ParametricRegion, TransformedRegion)):
            return None
    # Trigonometric ball charts rarely improve the algebraic shape rules.
    if any(
        any(expr.has(sp.sin, sp.cos) for expr in chart.mapping)
        for chart in cover.charts
    ):
        return None
    extra, _ = normalize_bounds_input(bounds, variables)
    domain = sp.And(constraint_formula(constraints), *extra, *assumptions)
    if domain.has(sp.Float) or sp.sympify(objective).has(sp.Float):
        return None
    results = []
    for chart in cover.charts:
        rename = {p: sp.Dummy(p.name, real=True) for p in chart.parameters}
        parameters = tuple(rename[p] for p in chart.parameters)
        mapping = dict(
            zip(variables, (expr.xreplace(rename) for expr in chart.mapping))
        )
        condition = sp.And(
            chart.parameter_formula().xreplace(rename), domain.xreplace(mapping)
        )
        from .rational_optimization import preserve_rational_domain

        for original, expression in zip(chart.mapping, mapping.values()):
            if parameters and not expression.is_rational_function(*parameters):
                return None
            numerator, denominator = sp.fraction(sp.cancel(expression))
            coefficients = (
                (
                    sp.Poly(numerator, *parameters).coeffs()
                    + sp.Poly(denominator, *parameters).coeffs()
                )
                if parameters
                else [expression]
            )
            if any(
                c.is_real is not True or c.is_finite is not True for c in coefficients
            ):
                return None
            guarded = preserve_rational_domain(original, [], chart.parameters)
            condition = sp.And(
                condition, *(guard.xreplace(rename) for guard in guarded)
            )
        reduced = sp.sympify(objective).xreplace(mapping)
        if not parameters:
            dummy = sp.Dummy("point", real=True)
            parameters = (dummy,)
            condition = sp.And(condition, sp.Eq(dummy, 0))
        result = solve(reduced, _constraints(condition), parameters)
        if (
            not result.certified
            or result.optimum_value is None
            or result.optimum_value.has(sp.Float)
        ):
            return None
        # A noninjective chart can attain the image infimum even if a particular
        # source witness is unavailable. Retain exact value evidence separately.
        points = []
        from .solver_pipeline import _optimizer_points

        for point in _optimizer_points(result):
            mapped = {
                v: sp.simplify(expr.subs(point, simultaneous=True))
                for v, expr in mapping.items()
            }
            if any(
                expr.free_symbols.intersection(parameters) for expr in mapped.values()
            ):
                continue
            points.append(mapped)
        from .problem_models import CandidateSolution

        candidates = [
            CandidateSolution(
                p,
                {},
                result.optimum_value,
                _numeric_value(result.optimum_value),
                True,
                "region_cover",
                original_variable_values=p,
            )
            for p in points
        ]
        problem = replace(
            result.lifted_problem,
            original_objective=sp.sympify(objective),
            original_variables=list(variables),
            all_variables=list(variables),
            original_constraints=[sp.And(prepared.source_formula, domain)],
            lifted_constraints=[sp.And(prepared.source_formula, domain)],
            defining_equalities=[],
            branch_inequalities=[],
            strict_inequalities=[],
            graph_equalities=[],
            graph_inequalities=[],
            auxiliary_variables=[],
            substitution_map={},
            lift_records=[],
            lifted_objective=sp.sympify(objective),
            parameter_symbols=sorted(
                (
                    sp.sympify(objective).free_symbols
                    | domain.free_symbols
                    | prepared.source_formula.free_symbols
                )
                - set(variables),
                key=sp.default_sort_key,
            ),
            structural_metadata={"region_chart": chart, "chart_result": result},
        )
        structural = StructuralGlobalResult(
            "region_cover",
            result.optimum_value,
            tuple(points),
            result.attained,
            metadata={"chart": chart},
        )
        certificate = result.certificate
        if certificate is not None:
            certificate = replace(
                certificate,
                primal_solution=points[0] if points else {},
                primal_value=result.optimum_value if points else None,
                proof_objects=[
                    *certificate.proof_objects,
                    ProofArtifact(
                        "region_cover",
                        "Full covering parameterization",
                        {"chart": chart, "source_result": result},
                    ),
                ],
            )
        results.append(
            replace(
                result,
                lifted_problem=problem,
                candidates=candidates,
                best_candidate=candidates[0] if candidates else None,
                kkt_system=None,
                structural_result=structural,
                certificate=certificate,
                method_label="region_cover",
                interval_result=None,
                semialg_result=None,
                semialg_range=None,
                inferred_box_bounds={},
                critical_point_results=[],
                sos_result=None,
                kkt_objective=None,
            )
        )
    result = results[0] if len(results) == 1 else merge(results)
    if len(results) > 1 and result.certified:
        from .solver_pipeline import _optimizer_points

        points = []
        for branch in results:
            if sp.simplify(branch.optimum_value - result.optimum_value) == 0:
                for point in _optimizer_points(branch):
                    if point not in points:
                        points.append(point)
        result.structural_result = StructuralGlobalResult(
            "region_cover",
            result.optimum_value,
            tuple(points),
            result.attained,
            metadata={"chart_results": tuple(results)},
        )
    result.region_preparation = prepared
    return result


def component_lower_bound(objective, constraints, variables):
    """Enclose an objective over an inferred product box for safe branch pruning."""
    from .rule_utils import RuleContext

    if any(
        sp.sympify(expression).has(sp.Float) for expression in (objective, *constraints)
    ):
        return None
    try:
        context = RuleContext(
            sp.sympify(objective), tuple(constraints), tuple(variables)
        )
        rectangle = context.rectangle
        if not rectangle or any(not interval.finite for interval in rectangle.values()):
            return None
        box = {
            v: (outward_float(interval.left, -1), outward_float(interval.right, 1))
            for v, interval in rectangle.items()
        }
        bound = enclose_expression(
            sp.sympify(objective), tuple(variables), tuple(box[v] for v in variables)
        )
    except (
        TypeError,
        ValueError,
        ArithmeticError,
        NotImplementedError,
        sp.PolynomialError,
    ):
        return None
    if bound.rigorous and bound.domain == "valid" and bound.lower != float("-inf"):
        return sp.Rational(bound.lower)
    return None


def vertex_minimum(prepared, objective):
    """Minimize an affine objective on a closed vertex hull or polygon exactly."""
    from semialg.standard_regions import Polygon, Polytope, Simplex

    from .rule_utils import polynomial_degree_bound

    if (
        not isinstance(prepared.source, (Polygon, Polytope, Simplex))
        or sp.And(*prepared.source.conditions) is not sp.true
    ):
        return None
    if sp.sympify(objective).has(sp.Float) or any(
        coordinate.has(sp.Float)
        for vertex in prepared.source.vertices
        for coordinate in vertex
    ):
        return None
    degree = polynomial_degree_bound(sp.sympify(objective), prepared.variables)
    if degree is None or degree > 1:
        return None
    if any(
        coordinate.free_symbols.intersection(prepared.variables)
        for vertex in prepared.source.vertices
        for coordinate in vertex
    ):
        return None
    vertices = tuple(
        dict(zip(prepared.variables, vertex)) for vertex in prepared.source.vertices
    )
    values = tuple(
        sp.simplify(sp.sympify(objective).xreplace(point)) for point in vertices
    )
    if any(
        value.is_real is not True or value.is_finite is not True for value in values
    ):
        return None
    optimum = sp.Min(*values)
    points = tuple(
        point
        for point, value in zip(vertices, values)
        if sp.simplify(value - optimum) == 0
    )
    return StructuralGlobalResult(
        "region_vertices",
        optimum,
        points,
        True,
        metadata={"vertices": vertices, "values": values},
    )


def _numeric_value(value):
    from .numeric_utils import eval_float

    try:
        return eval_float(value)
    except (ValueError, TypeError, OverflowError):
        return None
