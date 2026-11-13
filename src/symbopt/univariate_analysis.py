from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache

import sympy as sp
from sympy.calculus.util import continuous_domain, function_range
from sympy.core.relational import Relational

from .structural_results import StructuralGlobalResult

_SYMBOLIC_ERRORS = (
    TypeError,
    ValueError,
    NotImplementedError,
    RecursionError,
)


@lru_cache(maxsize=1)
def _funcprops_module():
    """Return the optional funcprops module when it is installed."""
    try:
        import funcprops
    except ImportError:
        return None
    return funcprops


def _funcprops_function(name):
    module = _funcprops_module()
    if module is None:
        return None
    function = getattr(module, name, None)
    return function if callable(function) else None


def _univariate_domain(constraints, variable):
    """Convert simple one-variable relational constraints to a real set."""
    convert = _funcprops_function("condition_to_real_set")
    if convert is not None:
        return convert(sp.And(*constraints), variable)
    domain = sp.S.Reals
    for relation in constraints:
        if not isinstance(relation, Relational):
            return None
        other_symbols = relation.free_symbols - {variable}
        if other_symbols:
            return None
        lhs, rhs = relation.lhs, relation.rhs
        if lhs == variable and rhs.is_extended_real:
            if isinstance(relation, sp.GreaterThan):
                piece = sp.Interval(rhs, sp.oo)
            elif isinstance(relation, sp.StrictGreaterThan):
                piece = sp.Interval.open(rhs, sp.oo)
            elif isinstance(relation, sp.LessThan):
                piece = sp.Interval(-sp.oo, rhs)
            elif isinstance(relation, sp.StrictLessThan):
                piece = sp.Interval.open(-sp.oo, rhs)
            elif isinstance(relation, sp.Equality):
                piece = sp.FiniteSet(rhs)
            else:
                return None
        elif rhs == variable and lhs.is_extended_real:
            if isinstance(relation, sp.GreaterThan):
                piece = sp.Interval(-sp.oo, lhs)
            elif isinstance(relation, sp.StrictGreaterThan):
                piece = sp.Interval.open(-sp.oo, lhs)
            elif isinstance(relation, sp.LessThan):
                piece = sp.Interval(lhs, sp.oo)
            elif isinstance(relation, sp.StrictLessThan):
                piece = sp.Interval.open(lhs, sp.oo)
            elif isinstance(relation, sp.Equality):
                piece = sp.FiniteSet(lhs)
            else:
                return None
        else:
            return None
        domain = sp.Intersection(domain, piece)
    return domain


def _condition_to_set(condition, variable):
    if isinstance(condition, sp.Set):
        return condition
    if condition is sp.S.true or condition is True:
        return sp.S.Reals
    if condition is sp.S.false or condition is False:
        return sp.S.EmptySet
    if not isinstance(condition, sp.Basic) or not condition.has(variable):
        return None
    try:
        result = sp.solveset(condition, variable, domain=sp.S.Reals)
    except (AttributeError, *_SYMBOLIC_ERRORS):
        try:
            convert = _funcprops_function("condition_to_real_set")
            result = (
                convert(condition, variable)
                if convert is not None
                else condition.as_set()
            )
        except (AttributeError, *_SYMBOLIC_ERRORS):
            return None
    return (
        result
        if isinstance(result, sp.Set) and not result.has(sp.ConditionSet)
        else None
    )


def _set_attains(range_set, value):
    if value in (sp.oo, -sp.oo):
        return False
    if not isinstance(range_set, sp.Set):
        return None
    try:
        contains = range_set.contains(value)
    except _SYMBOLIC_ERRORS:
        return None
    if contains is sp.S.true or contains is True:
        return True
    if contains is sp.S.false or contains is False:
        return False
    return None


def _point_for_value(objective, variable, domain, value):
    """Return one exact optimizer representative when it can be recovered."""
    if value in (sp.oo, -sp.oo):
        return ()
    parts = domain.args if isinstance(domain, sp.Union) else (domain,)
    endpoints = []
    for part in parts:
        if isinstance(part, sp.FiniteSet):
            endpoints.extend(part)
        elif isinstance(part, sp.Interval):
            endpoints.extend(p for p in (part.start, part.end) if p.is_finite is True)
    for point in endpoints:
        if (
            domain.contains(point) is sp.true
            and sp.simplify(objective.subs(variable, point) - value) == 0
        ):
            return ({variable: point},)
    # A derivative root is only a proposal here: direct substitution and domain
    # membership establish the witness, independently of root-list completeness.
    try:
        critical = sp.solveset(sp.diff(objective, variable), variable, domain=domain)
    except _SYMBOLIC_ERRORS:
        critical = None
    if isinstance(critical, sp.FiniteSet):
        for point in critical:
            if (
                point.is_real is True
                and domain.contains(point) is sp.true
                and sp.simplify(objective.subs(variable, point) - value) == 0
            ):
                return ({variable: point},)
    try:
        solutions = sp.solveset(sp.Eq(objective, value), variable, domain=domain)
    except _SYMBOLIC_ERRORS:
        return ()
    if isinstance(solutions, sp.FiniteSet):
        for root in solutions:
            if root.is_real is True and domain.contains(root) is sp.true:
                return ({variable: sp.simplify(root)},)
    if isinstance(solutions, sp.ImageSet):
        try:
            lam = solutions.lamda
            base_sets = solutions.base_sets
            if len(lam.variables) == 1 and base_sets == (sp.S.Integers,):
                root = sp.simplify(lam.expr.subs(lam.variables[0], 0))
                if domain.contains(root) is sp.S.true:
                    return ({variable: root},)
        except _SYMBOLIC_ERRORS:
            pass
    if isinstance(solutions, sp.Union):
        for component in solutions.args:
            if not isinstance(component, sp.ImageSet):
                continue
            try:
                lam = component.lamda
                base_sets = component.base_sets
                if len(lam.variables) == 1 and base_sets == (sp.S.Integers,):
                    for index in (0, 1, -1, 2, -2):
                        root = sp.simplify(lam.expr.subs(lam.variables[0], index))
                        if domain.contains(root) is sp.S.true:
                            return ({variable: root},)
            except _SYMBOLIC_ERRORS:
                continue
    return ()


def certified_monotone_direction(expr, variable, domain):
    """Return a funcprops-certified scalar monotone direction when available."""
    query = _funcprops_function("monotonicity")
    if query is None:
        return None
    try:
        result = query(expr, variable, domain=domain, return_result=True)
    except (ImportError, *_SYMBOLIC_ERRORS):
        return None
    value = getattr(result, "primary", result)
    value = getattr(value, "value", value)
    if getattr(result, "cases", ()):
        return None
    if value in ("strictly_increasing", "nondecreasing"):
        return True
    if value in ("strictly_decreasing", "nonincreasing"):
        return False
    return None


def _needs_real_locus(expression):
    """Avoid narrowing the domain of expressions with complex intermediates."""
    if expression.has(sp.I):
        return True

    def real_domain_primitives(node):
        # Only a real linear combination of at most one potentially complex
        # log or even-denominator root is safe to restrict by real operands.
        projections = {sp.Abs, sp.re, sp.im, sp.conjugate}
        if node.func in projections and any(
            arg.is_real is not True for arg in node.args
        ):
            return None
        if node.is_real is True:
            return 0
        if node.is_Add:
            parts = [real_domain_primitives(arg) for arg in node.args]
            return sum(parts) if all(part is not None for part in parts) else None
        if node.is_Mul:
            coefficient, dependent = node.as_coeff_Mul()
            if coefficient.is_real is True and coefficient != 1 and dependent != node:
                return real_domain_primitives(dependent)
            return None
        if node.func is sp.log and node.args[0].is_real is True:
            return 1
        if (
            node.is_Pow
            and node.exp.is_Rational
            and node.exp > 0
            and node.exp.q % 2 == 0
            and node.base.is_real is True
        ):
            return 1
        return None

    count = real_domain_primitives(expression)
    return count is None or count > 1


@dataclass(frozen=True)
class UnivariateAnalysis:
    """Shared lazy structural analysis for one real optimization variable.

    The object normalizes user and natural domains once, then lazily caches
    function-property facts used by the range, transcendental, periodic and
    endpoint/stationary-point routes.  ``funcprops`` is the preferred provider;
    SymPy is a conservative fallback when the optional package is unavailable
    or returns an inconclusive result.
    """

    objective: sp.Expr
    variable: sp.Symbol
    requested_domain: sp.Set
    assumptions: sp.Expr = sp.S.true

    @cached_property
    def natural_domain(self):
        """Return the known natural real-output domain, or None when it is unresolved."""
        query = _funcprops_function("domain")
        try:
            condition = (
                query(self.objective, self.variable, assumptions=self.assumptions)
                if query is not None
                else None
            )
            domain = _condition_to_set(condition, self.variable)
            if domain is not None:
                return domain
        except (ImportError, *_SYMBOLIC_ERRORS):
            pass
        if _needs_real_locus(self.objective):
            return None
        try:
            return continuous_domain(self.objective, self.variable, sp.S.Reals)
        except _SYMBOLIC_ERRORS:
            return None

    @cached_property
    def domain_record(self):
        """Source-defined real-output coverage on the requested domain, when known."""
        query = _funcprops_function("track_domain")
        if query is None:
            return None
        try:
            return query(self.objective, self.variable, self.requested_domain)
        except (ImportError, *_SYMBOLIC_ERRORS):
            return None

    @cached_property
    def feasible_domain(self):
        """Intersect source definedness with the request, retaining real outputs with complex intermediates."""
        record = self.domain_record
        if record is not None and record.complete:
            return record.effective
        natural = self.natural_domain
        if natural is None:
            return None
        try:
            return sp.Intersection(self.requested_domain, natural)
        except _SYMBOLIC_ERRORS:
            return None

    @cached_property
    def period_result(self):
        """Return the optional provider record for periodicity on the effective domain."""
        query = _funcprops_function("periodicity")
        if query is None:
            return None
        try:
            domain = self.feasible_domain
            if domain is None:
                domain = self.requested_domain
            result = query(
                self.objective,
                self.variable,
                domain=domain,
                assumptions=self.assumptions,
                return_result=True,
            )
            return result
        except (ImportError, *_SYMBOLIC_ERRORS):
            return None

    @cached_property
    def period(self):
        """Return a proved positive period, or None when no supported period is known."""
        result = self.period_result
        if result is not None:
            classification = getattr(
                getattr(result, "classification", None), "value", None
            )
            conditions = getattr(result, "conditions", sp.S.true)
            period = getattr(result, "period", None)
            if (
                classification == "periodic"
                and conditions is sp.S.true
                and period is not None
            ):
                period = sp.simplify(period)
                if period.is_positive is True:
                    return period
        try:
            period = sp.periodicity(self.objective, self.variable)
        except _SYMBOLIC_ERRORS:
            return None
        if period in (None, 0):
            return None
        period = sp.simplify(period)
        return period if period.is_positive is True else None

    @cached_property
    def singularity_analysis(self):
        """Return the optional provider's classified real singularities."""
        query = _funcprops_function("singularity_analysis")
        if query is None:
            return None
        try:
            return query(
                self.objective,
                self.variable,
                domain=sp.S.Reals,
                assumptions=self.assumptions,
            )
        except (ImportError, *_SYMBOLIC_ERRORS):
            return None

    @cached_property
    def singularity_set(self):
        """Return the known real singularity set, or None when unresolved."""
        query = _funcprops_function("singularities")
        if query is not None:
            try:
                condition = query(
                    self.objective,
                    self.variable,
                    domain=sp.S.Reals,
                    assumptions=self.assumptions,
                )
                result = _condition_to_set(condition, self.variable)
                if result is not None:
                    return result
            except (ImportError, *_SYMBOLIC_ERRORS):
                pass
        try:
            from sympy.calculus.singularities import singularities

            result = singularities(self.objective, self.variable, domain=sp.S.Reals)
            return result if isinstance(result, sp.Set) else None
        except _SYMBOLIC_ERRORS:
            return None

    @cached_property
    def discontinuities(self):
        """Return supplied discontinuity records on the real analysis coordinate."""
        query = _funcprops_function("discontinuities")
        if query is None:
            return ()
        try:
            return tuple(
                query(
                    self.objective,
                    self.variable,
                    assumptions=self.assumptions,
                    return_result=True,
                )
            )
        except (ImportError, *_SYMBOLIC_ERRORS):
            return ()

    @cached_property
    def monotonic_partition(self):
        """Return the optional partition of the feasible source into monotone cells."""
        domain = self.feasible_domain
        if domain is None or domain == sp.S.EmptySet:
            return None
        query = _funcprops_function("monotonic_partition")
        if query is None:
            return None
        try:
            return query(
                self.objective,
                self.variable,
                domain=domain,
                return_result=True,
            )
        except (ImportError, *_SYMBOLIC_ERRORS):
            return None

    @cached_property
    def partition_range(self):
        """Assemble an image only when the monotone partition supplies complete coverage."""
        partition = self.monotonic_partition
        if partition is None:
            return None
        query = _funcprops_function("range_from_monotonic_partition")
        if query is None:
            return None
        try:
            result = query(
                self.objective,
                self.variable,
                partition,
                domain=self.feasible_domain,
            )
        except (ImportError, *_SYMBOLIC_ERRORS):
            return None
        if getattr(result, "complete", False) and isinstance(
            getattr(result, "image", None), sp.Set
        ):
            return result.image
        return None

    @cached_property
    def extremum_range(self):
        """Return a complete extrema envelope from the optional symbolic provider."""
        domain = self.feasible_domain
        if domain is None or domain == sp.S.EmptySet:
            return None
        query = _funcprops_function("symbolic_extremum_range")
        if query is None:
            return None
        try:
            result = query(
                self.objective,
                self.variable,
                domain,
            )
        except (ImportError, *_SYMBOLIC_ERRORS):
            return None
        return result if getattr(result, "complete", False) else None

    @cached_property
    def range_condition(self):
        """Return an output symbol and its exact supported image condition."""
        domain = self.feasible_domain
        if domain is None or domain == sp.S.EmptySet:
            return None
        y = sp.Dummy("_range_value", real=True)
        query = _funcprops_function("range")
        if query is None:
            return None
        try:
            condition = query(
                self.objective,
                self.variable,
                y,
                domain=domain,
            )
        except (ImportError, *_SYMBOLIC_ERRORS):
            return None
        if isinstance(condition, sp.logic.boolalg.Boolean):
            return y, condition
        return None

    @cached_property
    def range_set(self):
        """Return the supported exact image as a SymPy set, or None when unresolved."""
        if self.partition_range is not None:
            return self.partition_range
        if self.extremum_range is not None:
            result = self.extremum_range
            if result.lower_attained is None or result.upper_attained is None:
                return None
            try:
                return sp.Interval(
                    result.lower,
                    result.upper,
                    left_open=result.lower_attained is False,
                    right_open=result.upper_attained is False,
                )
            except _SYMBOLIC_ERRORS:
                pass
        if self.range_condition is not None:
            y, condition = self.range_condition
            image = _condition_to_set(condition, y)
            if image is not None:
                return image
        domain = self.feasible_domain
        if domain is None:
            return None
        try:
            image = function_range(self.objective, self.variable, domain)
        except _SYMBOLIC_ERRORS:
            return None
        return image if isinstance(image, sp.Set) else None

    @cached_property
    def stationary_points(self):
        """Return exhaustive finite stationary coordinates when the analysis can establish them."""
        partition = self.monotonic_partition
        if (
            partition is not None
            and getattr(partition, "indexed", None) is None
            and getattr(partition, "critical_points_complete", False)
        ):
            return tuple(getattr(partition, "critical_points", ()))
        domain = self.feasible_domain
        if domain is None:
            return None
        try:
            roots = sp.solveset(
                sp.Eq(sp.diff(self.objective, self.variable), 0),
                self.variable,
                domain=domain,
            )
        except _SYMBOLIC_ERRORS:
            return None
        if roots is sp.S.EmptySet:
            return ()
        if isinstance(roots, sp.FiniteSet):
            return tuple(
                sp.simplify(root) for root in roots if root.is_real is not False
            )
        return None

    def endpoint_value(self, point, *, direction="+", attained=False):
        """Evaluate an attained endpoint or the corresponding one-sided limit."""
        if point is -sp.oo:
            try:
                return sp.simplify(sp.limit(self.objective, self.variable, -sp.oo))
            except _SYMBOLIC_ERRORS:
                return None
        if point is sp.oo:
            try:
                return sp.simplify(sp.limit(self.objective, self.variable, sp.oo))
            except _SYMBOLIC_ERRORS:
                return None
        if attained:
            try:
                return sp.simplify(self.objective.subs(self.variable, point))
            except _SYMBOLIC_ERRORS:
                return None
        try:
            # A monotonic partition carries endpoint limits with its cells, so
            # reuse those values before requesting another symbolic limit.
            part = self.monotonic_partition
            for cell in getattr(part, "cells", ()) if part is not None else ():
                if (
                    cell.interval.start == point
                    and direction == "+"
                    and cell.left_limit is not None
                ):
                    return cell.left_limit
                if (
                    cell.interval.end == point
                    and direction == "-"
                    and cell.right_limit is not None
                ):
                    return cell.right_limit
            return sp.simplify(
                sp.limit(self.objective, self.variable, point, dir=direction)
            )
        except _SYMBOLIC_ERRORS:
            return None

    @property
    def provider_summary(self):
        """Describe already-computed facts without triggering new analysis."""
        cached = tuple(
            name
            for name in (
                "natural_domain",
                "period_result",
                "period",
                "singularity_analysis",
                "singularity_set",
                "discontinuities",
                "monotonic_partition",
                "partition_range",
                "extremum_range",
                "range_condition",
                "range_set",
                "stationary_points",
            )
            if name in self.__dict__
        )
        return {
            "preferred_provider": "funcprops",
            "fallback_provider": "sympy",
            "cached_facts": cached,
        }


@lru_cache(maxsize=512)
def _analyze_univariate_cached(objective, constraints, variable, assumptions):
    requested = _univariate_domain(constraints, variable)
    if requested is None:
        return None
    return UnivariateAnalysis(objective, variable, requested, assumptions)


def analyze_univariate(objective, constraints, variable, *, assumptions=sp.S.true):
    """Return the cached shared analysis for a normalized univariate problem."""
    objective = sp.sympify(objective)
    constraints = tuple(sp.sympify(item) for item in constraints)
    variable = sp.sympify(variable)
    assumptions = sp.sympify(assumptions)
    try:
        return _analyze_univariate_cached(objective, constraints, variable, assumptions)
    except TypeError:
        requested = _univariate_domain(constraints, variable)
        if requested is None:
            return None
        return UnivariateAnalysis(objective, variable, requested, assumptions)


def clear_univariate_analysis_cache():
    """Clear cached univariate analyses, primarily for isolated test runs."""
    _analyze_univariate_cached.cache_clear()


def _logarithmic_escape(objective, variable, requested_domain, kind):
    """Prove unboundedness along a real, continuous logarithmic source interval."""
    if objective.free_symbols - {variable}:
        return None
    if any(function.func is not sp.log for function in objective.atoms(sp.Function)):
        return None
    if any(
        power.exp.is_Integer is not True
        and not (power.is_number and power.is_real is True)
        for power in objective.atoms(sp.Pow)
    ):
        return None
    if any(
        node.is_number and node.is_real is not True
        for node in sp.preorder_traversal(objective)
    ):
        return None
    # A vanishing imaginary term can leave a real infinite limit even though
    # the nearby objective is complex. Only real rational/log combinations
    # establish the real source path used by this certificate.
    # A real logarithmic pole inside a checked source interval already proves
    # unboundedness; constructing every critical cell cannot improve that claim.
    if objective.has(sp.log):
        try:
            subset = continuous_domain(objective, variable, requested_domain)
        except _SYMBOLIC_ERRORS:
            subset = sp.S.EmptySet
        parts = subset.args if isinstance(subset, sp.Union) else (subset,)
        if len(parts) <= 8:
            logarithms = objective.atoms(sp.log)
            for part in parts:
                if not isinstance(part, sp.Interval):
                    continue
                for endpoint, direction, opened in (
                    (part.start, "+", part.left_open),
                    (part.end, "-", part.right_open),
                ):
                    if not opened or endpoint.is_finite is not True:
                        continue
                    if not any(
                        log.args[0].subs(variable, endpoint).is_zero is True
                        for log in logarithms
                    ):
                        continue
                    try:
                        limit = sp.limit(objective, variable, endpoint, dir=direction)
                    except _SYMBOLIC_ERRORS:
                        continue
                    if limit is -sp.oo:
                        return StructuralGlobalResult(
                            kind=kind,
                            value=-sp.oo,
                            attained=False,
                            certified=True,
                            metadata={
                                "domain": subset,
                                "endpoint": endpoint,
                                "direction": direction,
                                "analysis": "real logarithmic escape",
                            },
                        )
    return None


def _range_result(kind, objective, constraints, variable, analysis=None):
    """Prove an infimum from an exact image or a real unbounded source path."""
    analysis = analysis or analyze_univariate(objective, constraints, variable)
    if analysis is None:
        return None
    escape = _logarithmic_escape(objective, variable, analysis.requested_domain, kind)
    if escape is not None:
        return escape
    feasible_domain = analysis.feasible_domain
    if feasible_domain == sp.S.EmptySet:
        return StructuralGlobalResult(
            kind=kind,
            value=sp.oo,
            points=(),
            attained=False,
            certified=True,
            metadata={
                "domain": sp.sstr(feasible_domain),
                "analysis": "exact empty feasible domain",
                "infeasible": True,
                "providers": analysis.provider_summary,
            },
        )
    image = analysis.range_set
    if image is None:
        extrema = analysis.extremum_range
        if extrema is None:
            return None
        return StructuralGlobalResult(
            kind=kind,
            value=sp.sympify(extrema.lower),
            points=(),
            attained=extrema.lower_attained,
            certified=True,
            metadata={
                "domain": sp.sstr(feasible_domain),
                "range_evidence": extrema,
                "analysis": "exact extremum envelope",
                "providers": analysis.provider_summary,
            },
        )
    try:
        infimum = sp.simplify(image.inf)
    except _SYMBOLIC_ERRORS:
        return None
    attained = _set_attains(image, infimum)
    points = (
        _point_for_value(objective, variable, feasible_domain, infimum)
        if attained
        else ()
    )
    return StructuralGlobalResult(
        kind=kind,
        value=infimum,
        points=points,
        attained=attained,
        certified=True,
        metadata={
            "domain": sp.sstr(feasible_domain),
            "range": sp.sstr(image),
            "analysis": "shared univariate function-property analysis",
            "providers": analysis.provider_summary,
        },
    )


def try_univariate_range(objective, constraints, variables, analysis=None):
    """Return an exact global univariate range result from the shared analysis."""
    if len(variables) != 1:
        return None
    variable = variables[0]
    analysis = analysis or analyze_univariate(objective, constraints, variable)
    if analysis is None:
        return None
    return _range_result(
        "univariate_range",
        objective,
        constraints,
        variable,
        analysis=analysis,
    )
