"""Shared exact domain, sign and witness checks for structural optimization rules."""

from dataclasses import dataclass
from functools import cached_property
from itertools import product

import sympy as sp

from .structural_results import StructuralGlobalResult
from .symbolic_cache import affine_coefficients, structural_coordinate

RULE_ERRORS = (
    sp.PolynomialError,
    NotImplementedError,
    TypeError,
    ValueError,
    ArithmeticError,
)


def polynomial_degree_bound(expression, variables):
    """Compute a syntactic degree bound without expanding sparse large powers."""
    if not expression.free_symbols.intersection(variables):
        return 0
    if expression in variables:
        return 1
    if expression.is_Add or expression.is_Mul:
        degrees = [polynomial_degree_bound(a, variables) for a in expression.args]
        if any(d is None for d in degrees):
            return None
        return max(degrees) if expression.is_Add else sum(degrees)
    if expression.is_Pow and expression.exp.is_Integer and expression.exp >= 0:
        degree = polynomial_degree_bound(expression.base, variables)
        return None if degree is None else degree * int(expression.exp)
    return None


def affine_data(expression, variables):
    """Use the shared coefficient cache only after a cheap affine-shape check."""
    degree = polynomial_degree_bound(expression, variables)
    return (
        affine_coefficients(expression, variables)
        if degree is not None and degree <= 1
        else None
    )


def truth(condition):
    """Return True, False or None after exact symbolic condition simplification."""
    value = sp.simplify(condition)
    return True if value is sp.true else False if value is sp.false else None


def checked_point(objective, constraints, variables, point):
    """Check a concrete or parameter-dependent real point in the original domain."""
    if (
        any(v.is_real is False for v in variables)
        or objective.has(sp.Float)
        or any(r.has(sp.Float) for r in constraints)
    ):
        return None
    if not all(v in point for v in variables):
        return None
    point = {v: sp.simplify(sp.sympify(point[v])) for v in variables}
    if any(
        value.is_real is not True or value.free_symbols.intersection(variables)
        for value in point.values()
    ):
        return None
    if any(truth(r.subs(point, simultaneous=True)) is not True for r in constraints):
        return None
    value = sp.simplify(objective.subs(point, simultaneous=True))
    if value.is_real is not True or value.has(sp.Float):
        return None
    return point, value


def point_result(kind, objective, constraints, variables, point, **proof):
    """Construct a structural result only after checking the original source point and constraints."""
    result = checked_point(objective, constraints, variables, point)
    if result is None:
        return None
    point, value = result
    return StructuralGlobalResult(kind, value, (point,), True, metadata=proof)


def linear_solution(equations, variables):
    """Solve an affine system and choose zero for free decision coordinates."""
    if not equations:
        return dict.fromkeys(variables, sp.S.Zero)
    if any(affine_data(eq, variables) is None for eq in equations):
        return None
    solutions = sp.linsolve(equations, variables)
    if not isinstance(solutions, sp.FiniteSet) or len(solutions) != 1:
        return None
    zero = dict.fromkeys(variables, sp.S.Zero)
    point = {
        v: sp.simplify(value.subs(zero, simultaneous=True))
        for v, value in zip(variables, next(iter(solutions)))
    }
    if any(value.is_real is not True for value in point.values()):
        return None
    if any(
        truth(sp.Eq(eq.subs(point, simultaneous=True), 0)) is not True
        for eq in equations
    ):
        return None
    return point


@dataclass(frozen=True)
class AxisInterval:
    """Exact Cartesian bounds with independent endpoint-inclusion flags."""

    left: sp.Expr = -sp.oo
    right: sp.Expr = sp.oo
    left_open: bool = True
    right_open: bool = True

    @property
    def finite(self):
        """Whether both interval endpoints are finite exact values."""
        return self.left.is_finite is True and self.right.is_finite is True

    def sample(self):
        """Choose an interior sample, or the singleton coordinate of a collapsed interval."""
        if self.finite:
            return (self.left + self.right) / 2
        if self.left != -sp.oo:
            return self.left + 1
        return self.right - 1 if self.right != sp.oo else sp.S.Zero


@dataclass
class RuleContext:
    """Shared facts for a real optimization problem without changing its symbols."""

    objective: sp.Expr
    constraints: tuple
    variables: tuple

    @cached_property
    def rectangle(self):
        """Recognize a nonempty Cartesian interval domain, including strict bounds."""
        bounds = {v: [-sp.oo, sp.oo, True, True] for v in self.variables}
        exclusions = []
        # |x| <= b (or < b) is exactly the pair -b <= x <= b for
        # real x and a finite, nonnegative constant b.
        relations = []
        for relation in self.constraints:
            if (
                isinstance(relation, (sp.LessThan, sp.StrictLessThan))
                and relation.lhs.func is sp.Abs
                and relation.lhs.args[0] in self.variables
                and relation.lhs.args[0].is_real is True
                and relation.rhs.is_number is True
                and relation.rhs.is_finite is True
                and relation.rhs.is_nonnegative is True
            ):
                variable = relation.lhs.args[0]
                relations.extend(
                    (
                        relation.func(-relation.rhs, variable),
                        relation.func(variable, relation.rhs),
                    )
                )
            else:
                relations.append(relation)
        for relation in relations:
            if isinstance(relation, sp.Unequality):
                exclusions.append(relation.lhs - relation.rhs)
                continue
            if truth(relation) is True:
                continue
            used = relation.free_symbols.intersection(self.variables)
            if len(used) != 1:
                return None
            variable = next(iter(used))
            data = affine_data(relation.lhs - relation.rhs, (variable,))
            if data is None:
                return None
            (slope,), constant = data
            if slope.is_positive is not True and slope.is_negative is not True:
                return None
            endpoint = sp.simplify(-constant / slope)
            if endpoint.is_real is not True:
                return None
            if isinstance(relation, sp.Equality):
                choices = ((0, False), (1, False))
            else:
                greater = isinstance(relation, (sp.GreaterThan, sp.StrictGreaterThan))
                lower = greater == (slope.is_positive is True)
                choices = (
                    (
                        0 if lower else 1,
                        isinstance(relation, (sp.StrictGreaterThan, sp.StrictLessThan)),
                    ),
                )
            for side, opened in choices:
                current = bounds[variable][side]
                stronger = endpoint > current if side == 0 else endpoint < current
                if truth(stronger) is True:
                    bounds[variable][side] = endpoint
                    bounds[variable][side + 2] = opened
                elif truth(sp.Eq(endpoint, current)) is True:
                    bounds[variable][side + 2] |= opened
                elif truth(stronger) is not False:
                    return None
        result = {v: AxisInterval(*values) for v, values in bounds.items()}
        for interval in result.values():
            if truth(interval.left < interval.right) is True:
                continue
            if (
                interval.left == interval.right
                and not interval.left_open
                and not interval.right_open
            ):
                continue
            return None
        aliases = interval_symbols(self.variables, result)
        for expression in exclusions:
            if expression.xreplace(aliases).is_zero is False:
                continue
            if len(self.variables) > 8 or not all(i.finite for i in result.values()):
                return None
            poly = sp.Poly(expression, *self.variables)
            if any(poly.degree(v) > 1 for v in self.variables):
                return None
            values = [
                expression.subs(dict(zip(self.variables, point)), simultaneous=True)
                for point in product(*((i.left, i.right) for i in result.values()))
            ]
            if not (
                all(v.is_positive is True for v in values)
                or all(v.is_negative is True for v in values)
            ):
                return None
        return result

    @cached_property
    def real_symbols(self):
        """Real coordinate aliases with only the signs justified by axis restrictions."""
        rectangle = self.rectangle
        if rectangle is None:
            # Axis facts remain valid under additional coupled constraints, but
            # those constraints are never dropped by box or witness checks.
            axis = tuple(
                r
                for r in self.constraints
                if not isinstance(r, sp.Unequality)
                and len(r.free_symbols.intersection(self.variables)) == 1
            )
            rectangle = RuleContext(self.objective, axis, self.variables).rectangle
        return interval_symbols(
            self.variables,
            rectangle,
            (self.objective, self.constraints, self.variables),
        )

    def sign(self, expression):
        """Return an exact supported sign code under the axis restrictions, or None."""
        expression = expression.xreplace(self.real_symbols)
        if expression.is_zero is True:
            return 0
        if expression.is_positive is True:
            return 1
        if expression.is_negative is True:
            return -1
        if expression.is_nonnegative is True:
            return 2
        if expression.is_nonpositive is True:
            return -2
        return None

    def real(self, expression):
        """Whether the expression is proved real using the established axis signs."""
        return expression.xreplace(self.real_symbols).is_real is True

    def seeds(self):
        """Yield deterministic interior and coordinate seeds for exact witness checks."""
        if self.rectangle:
            yield {v: i.sample() for v, i in self.rectangle.items()}
        for scalar in (0, 1, -1):
            yield dict.fromkeys(self.variables, sp.Integer(scalar))

    def result(self, kind, point, **proof):
        """Construct a structural result only after checking the original source point and constraints."""
        return point_result(
            kind, self.objective, self.constraints, self.variables, point, **proof
        )

    def corners(self):
        """Return Cartesian corners when all axis intervals are finite and small enough."""
        if not self.rectangle or not all(i.finite for i in self.rectangle.values()):
            return ()
        if len(self.variables) > 8:
            return ()
        return tuple(
            dict(zip(self.variables, values))
            for values in product(*((i.left, i.right) for i in self.rectangle.values()))
        )


def interval_symbols(variables, rectangle, problem_key=()):
    """Create real aliases with only signs justified by Cartesian bounds."""
    substitutions = {}
    for variable in variables:
        assumptions = {"real": True}
        interval = rectangle.get(variable) if rectangle else None
        if interval is not None:
            if truth(interval.left >= 0) is True:
                assumptions = (
                    {"positive": True}
                    if interval.left_open or truth(interval.left > 0) is True
                    else {"nonnegative": True}
                )
            elif truth(interval.right <= 0) is True:
                assumptions = (
                    {"negative": True}
                    if interval.right_open or truth(interval.right < 0) is True
                    else {"nonpositive": True}
                )
        substitutions[variable] = structural_coordinate(
            variable.name, (problem_key, variable), tuple(sorted(assumptions.items()))
        )
    return substitutions
