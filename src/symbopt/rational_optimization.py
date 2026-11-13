"""Exact optimization of rational functions over real algebraic domains."""

from functools import cmp_to_key, lru_cache
from itertools import pairwise

import sympy as sp
from sympy.core.relational import Relational

from .algebraic_numbers import algebraic_compare
from .input_normalization import (
    normalize_constraint_branches,
    normalize_constraints_input,
    normalize_variables,
)
from .structural_results import StructuralGlobalResult

_EXACT_ERRORS = (sp.PolynomialError, NotImplementedError, TypeError, ValueError)


def _compare(left, right):
    if left == right:
        return 0
    if left == -sp.oo or right == sp.oo:
        return -1
    if left == sp.oo or right == -sp.oo:
        return 1
    result = algebraic_compare(left, right)
    if result is None:
        root = sp.polys.numberfields.to_number_field(left - right).to_root()
        if root.is_zero is True:
            return 0
        if root.is_positive is True:
            return 1
        if root.is_negative is True:
            return -1
        raise NotImplementedError("exact algebraic comparison is unresolved")
    return result


def _polynomial(expression, variables):
    poly = sp.Poly(expression, *variables, extension=True)
    if not (
        poly.domain.is_ZZ or poly.domain.is_QQ or poly.domain.is_AlgebraicField
    ) or any(c.is_real is not True for c in poly.coeffs()):
        raise NotImplementedError("coefficients must be fixed real algebraic numbers")
    return poly


def _fraction(expression, variables):
    numerator, denominator = sp.fraction(sp.cancel(expression))
    return _polynomial(numerator, variables), _polynomial(denominator, variables)


def _exclusions(expression, variables):
    """Retain definedness of the input expression before rational cancellation."""
    exclusions = []
    for node in sp.preorder_traversal(expression):
        if isinstance(node, sp.Pow) and node.exp.is_Integer and node.exp < 0:
            numerator, _ = _fraction(node.base, variables)
            exclusions.append(sp.Ne(numerator.as_expr(), 0, evaluate=False))
    return exclusions


def _prepare_rational(objective, constraints, variables):
    """Polynomialize rational relations without multiplying by an unknown sign."""
    expressions = [objective]
    for relation in constraints:
        if not isinstance(relation, Relational):
            raise NotImplementedError("a conjunction of rational relations is required")
        expressions.extend((relation.lhs, relation.rhs))
    if any(e.has(sp.Float) or e.free_symbols - set(variables) for e in expressions):
        raise NotImplementedError("exact fixed coefficients are required")
    conditions = []
    for expression in expressions:
        conditions.extend(_exclusions(expression, variables))
    numerator, denominator = _fraction(objective, variables)
    conditions.append(sp.Ne(denominator.as_expr(), 0, evaluate=False))
    for relation in constraints:
        p, q = _fraction(relation.lhs - relation.rhs, variables)
        conditions.append(sp.Ne(q.as_expr(), 0, evaluate=False))
        residual = p.as_expr()
        if not isinstance(relation, (sp.Equality, sp.Unequality)):
            residual *= q.as_expr()
        conditions.append(relation.func(sp.expand(residual), 0, evaluate=False))
    return numerator, denominator, tuple(dict.fromkeys(conditions))


def rational_graph(objective, constraints, variables, value_symbol):
    """Return the exact polynomial graph formula of a rational objective.

    The decision and value symbols must be distinct. Coefficients must be exact
    real algebraic constants. Denominator exclusions in the supplied expression
    and constraints survive cancellation; exclusions already erased by the
    expression's construction must be supplied explicitly as constraints.
    """
    variables = tuple(normalize_variables(variables) or ())
    if not variables or not isinstance(value_symbol, sp.Symbol):
        raise ValueError("provide decision variables and a value symbol")
    objective = sp.sympify(objective)
    constraints = normalize_constraints_input(constraints)
    symbols = objective.free_symbols | set().union(
        *(r.free_symbols for r in constraints)
    )
    if value_symbol in symbols or value_symbol in variables:
        raise ValueError("the value symbol must not occur in the input problem")
    p, q, conditions = _prepare_rational(objective, constraints, variables)
    return sp.And(
        *(r.func(r.lhs, r.rhs) for r in conditions),
        sp.Eq(value_symbol * q.as_expr(), p.as_expr()),
    )


def preserve_rational_domain(objective, constraints, variables):
    """Add input denominator exclusions before preprocessing can cancel factors."""
    try:
        expressions = [objective]
        for relation in constraints:
            expressions.extend((relation.lhs, relation.rhs))
        exclusions = []
        for expression in expressions:
            for node in sp.preorder_traversal(expression):
                if not (
                    isinstance(node, sp.Pow) and node.exp.is_Integer and node.exp < 0
                ):
                    continue
                numerator, denominator = sp.fraction(sp.together(node.base))
                if not (
                    numerator.is_polynomial(*variables)
                    and denominator.is_polynomial(*variables)
                ):
                    continue
                # Definedness also matters for symbolic or inexact coefficients,
                # even when the exact optimizer must decline those coefficients.
                condition = sp.Ne(numerator, 0)
                if condition is sp.false:
                    condition = sp.Eq(0, 1, evaluate=False)
                if condition is not sp.true:
                    exclusions.append(condition)
        return list(dict.fromkeys([*constraints, *exclusions]))
    except _EXACT_ERRORS:
        return constraints


@lru_cache(maxsize=1024)
def _real_roots(expression, variable):
    poly = _polynomial(expression, (variable,))
    if poly.is_zero or poly.degree() == 0:
        return ()
    try:
        roots = sp.real_roots(poly, extension=True)
    except TypeError:
        # Older SymPy accepts rational coefficients without the extension option.
        roots = sp.real_roots(poly)
    return tuple(dict.fromkeys(roots))


def _relation_truth(relation, variable, point):
    sign = _compare((relation.lhs - relation.rhs).subs(variable, point), sp.S.Zero)
    return {
        sp.Equality: sign == 0,
        sp.Unequality: sign != 0,
        sp.GreaterThan: sign >= 0,
        sp.StrictGreaterThan: sign > 0,
        sp.LessThan: sign <= 0,
        sp.StrictLessThan: sign < 0,
    }[type(relation)]


def _valuation(poly, point):
    """Return the zero order and first nonzero Taylor coefficient at an endpoint."""
    derivative = poly
    for order in range(poly.degree() + 1):
        coefficient = derivative.eval(point)
        if _compare(coefficient, sp.S.Zero) != 0:
            return order, coefficient / sp.factorial(order)
        derivative = derivative.diff()
    raise ValueError("valuation requires a nonzero polynomial")


def _rational_limit(p, q, point, side):
    """Compute one-sided rational limits by exact orders, including infinity."""
    if p.is_zero:
        return sp.S.Zero
    if point in (-sp.oo, sp.oo):
        power = p.degree() - q.degree()
        ratio = p.LC() / q.LC()
        if power < 0:
            return sp.S.Zero
        if power == 0:
            return ratio
        sign = _compare(ratio, 0) * (-1 if point == -sp.oo and power % 2 else 1)
        return sp.oo if sign > 0 else -sp.oo
    p_order, p_coefficient = _valuation(p, point)
    q_order, q_coefficient = _valuation(q, point)
    power = p_order - q_order
    ratio = sp.cancel(p_coefficient / q_coefficient)
    if power > 0:
        return sp.S.Zero
    if power == 0:
        return ratio
    sign = _compare(ratio, 0) * (-1 if side == "-" and power % 2 else 1)
    return sp.oo if sign > 0 else -sp.oo


def _sample(left, right):
    if left == -sp.oo:
        return sp.S.Zero if right == sp.oo else right - 1
    return left + 1 if right == sp.oo else (left + right) / 2


def try_univariate_rational(objective, constraints, variables):
    """Enumerate all critical, boundary and limiting values in one variable.

    Any incomplete root isolation, sign test or value comparison refuses the
    route. The returned certificate records exact enumeration, including empty
    domains and unattained limits. Constant intervals retain an interior witness.
    """
    if len(variables) != 1:
        return None
    variable = variables[0]
    try:
        p, q, conditions = _prepare_rational(objective, constraints, variables)
        derivative = p.diff() * q - p * q.diff()
        cuts = set(_real_roots(derivative.as_expr(), variable))
        for relation in conditions:
            cuts.update(_real_roots(relation.lhs - relation.rhs, variable))
        ordered = sorted(cuts, key=cmp_to_key(_compare))
        candidates = []
        for point in ordered:
            if all(_relation_truth(r, variable, point) for r in conditions):
                candidates.append((sp.cancel(objective.subs(variable, point)), point))
        endpoints = [-sp.oo, *ordered, sp.oo]
        intervals = []
        for left, right in pairwise(endpoints):
            sample = _sample(left, right)
            if not all(_relation_truth(r, variable, sample) for r in conditions):
                continue
            intervals.append((left, right))
            candidates.extend(
                (
                    (_rational_limit(p, q, left, "+"), None),
                    (_rational_limit(p, q, right, "-"), None),
                )
            )
            if derivative.is_zero:
                candidates.append((sp.cancel(objective.subs(variable, sample)), sample))
        if not candidates:
            value, points = sp.oo, ()
        else:
            value = candidates[0][0]
            for candidate, _ in candidates[1:]:
                if _compare(candidate, value) < 0:
                    value = candidate
            points = tuple(
                {variable: point}
                for candidate, point in candidates
                if point is not None and _compare(candidate, value) == 0
            )
        return StructuralGlobalResult(
            kind="univariate_rational" if q.degree() else "univariate_polynomial",
            value=value,
            points=points,
            attained=bool(points),
            metadata={
                "enumerated_candidates": tuple(candidates),
                "critical_points_complete": True,
                "cuts": tuple(ordered),
                "feasible_intervals": tuple(intervals),
                "domain": sp.And(*conditions),
                "infeasible": not candidates,
                "singularities": _real_roots(q.as_expr(), variable),
            },
        )
    except _EXACT_ERRORS:
        return None


def try_rational_range(objective, constraints, variables, *, cost_limit=12):
    """Project a rational graph with semialg and minimize its exact real image.

    The cost limit bounds a degree/dimension estimate before quantifier
    elimination; None explicitly permits any estimate. Only a certified,
    quantifier-free image can establish the infimum or attainment. Failure to
    recover a concrete optimizer does not invalidate an exact image proof.
    """
    if len(variables) < 2:
        return None
    try:
        p, q, _ = _prepare_rational(objective, constraints, variables)
        if q.total_degree() == 0 and not _exclusions(objective, variables):
            return None
        value_symbol = sp.Dummy("rational_value", real=True)
        graph = rational_graph(objective, constraints, variables, value_symbol)
        cost = len(variables) * (p.total_degree() + q.total_degree())
        cost += sum(
            _polynomial(r.lhs - r.rhs, variables).total_degree()
            for r in graph.atoms(Relational)
            if value_symbol not in r.free_symbols
        )
        if cost_limit is not None and cost > cost_limit:
            return None
        import semialg

        eliminate = getattr(semialg, "quantifier_eliminate", None)
        if eliminate is None:
            return None
        projection = eliminate(
            graph,
            tuple(("exists", v) for v in variables),
            variables=(value_symbol, *variables),
            return_result=True,
        )
        if getattr(projection, "certified", False) is not True:
            return None
        image = projection.formula
        if image.free_symbols - {value_symbol}:
            return None
        if image is sp.false:
            return StructuralGlobalResult(
                "rational_range",
                sp.oo,
                metadata={"infeasible": True, "graph": graph},
            )
        branches = normalize_constraint_branches(image)
        ranges = [
            try_univariate_rational(value_symbol, b, (value_symbol,)) for b in branches
        ]
        if not ranges or any(r is None for r in ranges):
            return None
        value = min((r.value for r in ranges), key=cmp_to_key(_compare))
        attained = any(r.attained for r in ranges if _compare(r.value, value) == 0)
        points = ()
        if attained:
            try:
                witness = semialg.find_instance(
                    graph.subs(value_symbol, value),
                    variables,
                    exact=True,
                )
                point = _exact_witness(graph.subs(value_symbol, value), witness)
                if point is not None:
                    points = (point,)
            except _EXACT_ERRORS:
                pass
            if not points:
                try:
                    candidate = semialg.semialgebraic_minimize(
                        objective,
                        sp.And(*constraints),
                        variables,
                        certification="candidate",
                        return_result=True,
                    )
                    for witness in candidate.points:
                        point = _exact_witness(graph.subs(value_symbol, value), witness)
                        if point is not None:
                            points = (point,)
                            break
                except _EXACT_ERRORS:
                    pass
        return StructuralGlobalResult(
            "rational_range",
            value,
            points=points,
            attained=attained,
            metadata={
                "graph": graph,
                "range_formula": image,
                "value_symbol": value_symbol,
                "projection_method": projection.method,
                "cost_estimate": cost,
            },
        )
    except (ImportError, *_EXACT_ERRORS):
        return None


def _exact_witness(graph, witness):
    """Resolve triangular backend substitutions and check a concrete real point."""
    if not isinstance(witness, dict):
        return None
    point = {v: sp.sympify(value) for v, value in witness.items()}
    for _ in range(len(point)):
        expanded = {
            v: sp.simplify(value.subs(point, simultaneous=True))
            for v, value in point.items()
        }
        if expanded == point:
            break
        point = expanded
    if any(
        value.free_symbols or value.has(sp.Float) or value.is_real is not True
        for value in point.values()
    ):
        return None
    return (
        point if sp.simplify(graph.subs(point, simultaneous=True)) is sp.true else None
    )
