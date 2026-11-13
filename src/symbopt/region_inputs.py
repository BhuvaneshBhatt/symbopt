"""Lower geometric membership to the optimization constraint interface."""

from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType

import sympy as sp
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean

from .input_normalization import normalize_variables


def _set_formula(region, coordinates):
    """Form real-coordinate set membership without guessing an ambient dimension."""
    if isinstance(region, (sp.Union, sp.Intersection)):
        operator = sp.Or if isinstance(region, sp.Union) else sp.And
        return operator(*(_set_formula(part, coordinates) for part in region.args))
    if isinstance(region, sp.Complement):
        left, right = region.args
        return sp.And(
            _set_formula(left, coordinates), sp.Not(_set_formula(right, coordinates))
        )
    if region in (sp.S.EmptySet, sp.S.UniversalSet):
        return sp.false if region is sp.S.EmptySet else sp.true
    size = len(coordinates)
    if isinstance(region, sp.ProductSet):
        if len(region.sets) != size or any(
            isinstance(part, sp.ProductSet) for part in region.sets
        ):
            raise ValueError("ProductSet must have one scalar factor per coordinate")
        return sp.And(
            *(
                _set_formula(part, (coordinate,))
                for part, coordinate in zip(region.sets, coordinates)
            )
        )
    elif isinstance(region, sp.FiniteSet) and size > 1:
        if any(
            not isinstance(point, sp.Tuple) or len(point) != size for point in region
        ):
            raise ValueError("finite region points must match the coordinate dimension")
    elif isinstance(region, sp.FiniteSet) and any(
        isinstance(point, sp.Tuple) for point in region
    ):
        raise ValueError("use scalar finite-set points for a univariate region")
    elif size != 1:
        raise ValueError("use a ProductSet or tuples of points for a multivariate set")

    # Optimization coordinates are real even when their symbols have no real
    # assumption. Real dummies avoid unresolved Contains(x, Reals) predicates.
    real_coordinates = tuple(sp.Dummy(real=True) for _ in coordinates)
    if isinstance(region, sp.FiniteSet) and size > 1:
        formula = sp.Or(
            *(
                sp.And(*(sp.Eq(v, c) for v, c in zip(real_coordinates, point)))
                for point in region
            )
        )
        return formula.xreplace(dict(zip(real_coordinates, coordinates)))
    member = (
        sp.Tuple(*real_coordinates)
        if size > 1 or isinstance(region, sp.ProductSet)
        else real_coordinates[0]
    )
    formula = region.contains(member)
    return formula.xreplace(dict(zip(real_coordinates, coordinates)))


def _normalize_region(region, variables):
    """Return exact membership constraints and the region's ordered coordinates.

    Formulas, SymPy sets and coordinate-free geometry require explicit variables.
    Semialg symbolic/CAD regions own their coordinate order. Geometry conversion
    and any required quantifier elimination are delegated to semialg; incomplete
    or unsupported conversion raises without changing the feasible set.
    """
    coordinates = normalize_variables(variables)
    if isinstance(region, (bool, Boolean, sp.Set)):
        if coordinates is None:
            raise ValueError(
                "variables= is required for a formula, set or coordinate-free region"
            )
        formula = (
            _set_formula(region, coordinates)
            if isinstance(region, sp.Set)
            else sp.sympify(region)
        )
    else:
        try:
            from semialg import CADRegion, Geometry, SemialgebraicRegion
        except ImportError as exc:
            raise ImportError(
                "geometric region objects require symbopt[semialg]"
            ) from exc
        if isinstance(region, (SemialgebraicRegion, CADRegion)):
            owned = normalize_variables(region.variables)
            if coordinates is not None and coordinates != owned:
                raise ValueError("variables= must match the region's coordinate order")
            coordinates = owned
            symbolic = region.as_region() if isinstance(region, CADRegion) else region
        elif isinstance(region, Geometry):
            if coordinates is None:
                raise ValueError("variables= is required for coordinate-free geometry")
            symbolic = region.as_semialgebraic_region(tuple(coordinates))
        else:
            raise TypeError(
                "region must be a Boolean formula, SymPy set or semialg region"
            )
        formula = symbolic.quantifier_free_formula()

    if any(variable.is_real is False for variable in coordinates):
        raise ValueError("region coordinates must be real")
    formula = sp.to_nnf(formula, simplify=True)

    def relational(condition):
        if condition is sp.false:
            # An empty region is a valid domain. Keep an explicit contradiction
            # so it does not depend on the separate constraints=False policy.
            return sp.Eq(0, 1, evaluate=False)
        if condition is sp.true or isinstance(condition, Relational):
            return condition
        if isinstance(condition, (sp.And, sp.Or)):
            return condition.func(*(relational(part) for part in condition.args))
        raise NotImplementedError(
            "region conversion must produce quantifier-free real relations"
        )

    return relational(formula), coordinates


class RegionConversionError(NotImplementedError):
    """A region conversion failed or exceeded its deterministic admission budget."""

    def __init__(self, stage, reason):
        super().__init__(f"region {stage}: {reason}")
        self.stage = stage
        self.reason = reason


@dataclass(frozen=True)
class PreparedRegion:
    """Reusable exact region input with lazy membership, covers and provenance.

    Construct with :func:`prepare_region`. Operation and quantifier limits are
    admission limits, not wall-clock deadlines on an individual CAD operation.
    Charts and outer bounds are optional structural facts, never replacements
    for source membership.
    """

    source: object
    variables: tuple
    max_operations: int | None = 5000
    max_quantifiers: int | None = 12

    def __post_init__(self):
        coordinates = tuple(normalize_variables(self.variables))
        if any(v.is_real is False for v in coordinates):
            raise ValueError("region coordinates must be real")
        object.__setattr__(self, "variables", coordinates)
        for limit in (self.max_operations, self.max_quantifiers):
            if limit is not None and (
                isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
            ):
                raise ValueError(
                    "region admission limits must be nonnegative integers or None"
                )

    @cached_property
    def is_geometry(self):
        """Whether the source exposes semialg's explicit geometry protocol."""
        if isinstance(self.source, (bool, Boolean, sp.Set)):
            return False
        from semialg import Geometry

        return isinstance(self.source, Geometry)

    @cached_property
    def source_formula(self):
        """Return membership before quantifier elimination, preserving binders."""
        if isinstance(self.source, sp.Set):
            return _set_formula(self.source, self.variables)
        if isinstance(self.source, (bool, Boolean)):
            return sp.sympify(self.source)
        symbolic = (
            self.source.as_region()
            if hasattr(self.source, "as_region")
            else self.source
        )
        if hasattr(symbolic, "as_semialgebraic_region"):
            symbolic = symbolic.as_semialgebraic_region(self.variables)
        return symbolic.formula

    def check_budget(self, formula):
        """Reject expensive conversion inputs before invoking an exact backend."""
        operations = sp.count_ops(formula)
        quantified = sum(
            len(node.variables)
            for node in sp.preorder_traversal(formula)
            if getattr(node, "is_Quantifier", False)
        )
        if self.max_operations is not None and operations > self.max_operations:
            raise RegionConversionError("admission", "operation limit exceeded")
        if self.max_quantifiers is not None and quantified > self.max_quantifiers:
            raise RegionConversionError("admission", "quantifier limit exceeded")

    @cached_property
    def membership(self):
        """Return cached quantifier-free membership or a traceable conversion error."""
        self.check_budget(self.source_formula)
        try:
            return _normalize_region(self.source, self.variables)[0]
        except NotImplementedError as exc:
            raise RegionConversionError("membership", str(exc)) from exc

    @cached_property
    def cover(self):
        """Return full structural charts, or None when no covering rule applies."""
        if not self.is_geometry:
            return None
        from semialg.standard_regions import (
            Ball,
            Ellipsoid,
            EllipsoidBoundary,
            Interval,
            Sphere,
        )

        if isinstance(
            self.source, (Ball, Sphere, Ellipsoid, EllipsoidBoundary, Interval)
        ):
            return None
        try:
            from semialg.region_covers import optimization_cover
        except ImportError:
            return None

        try:
            return optimization_cover(self.source, self.variables)
        except (NotImplementedError, TypeError):
            return None

    @cached_property
    def outer_bounds(self):
        """Return immutable proved outer bounds without optimization or CAD."""
        return MappingProxyType(self._outer_bounds())

    def _outer_bounds(self):
        """Compute structural coordinate covers while keeping geometry optional."""
        if isinstance(self.source, sp.Interval):
            return {self.variables[0]: (self.source.start, self.source.end)}
        if (
            isinstance(self.source, sp.ProductSet)
            and len(self.source.sets) == len(self.variables)
            and all(isinstance(part, sp.Interval) for part in self.source.sets)
        ):
            return {
                v: (part.start, part.end)
                for v, part in zip(self.variables, self.source.sets)
            }
        if isinstance(self.source, (bool, Boolean, sp.Set)):
            return {}
        if not self.is_geometry:
            return {}
        try:
            from semialg.region_covers import bounding_cover
        except ImportError:
            return {}
        try:
            return bounding_cover(self.source, self.variables)
        except (NotImplementedError, TypeError):
            return {}

    @property
    def provenance(self):
        """Describe conversion ownership and which reusable facts were computed."""
        return {
            "source": type(self.source).__name__,
            "coordinates": self.variables,
            "membership_ready": "membership" in self.__dict__,
            "max_operations": self.max_operations,
            "max_quantifiers": self.max_quantifiers,
        }


def prepare_region(region, variables=None, *, max_operations=5000, max_quantifiers=12):
    """Prepare a reusable exact region, deferring costly membership elimination.

    Pass the result as ``region=`` in repeated optimization calls. Coordinates
    must match those of owned symbolic/CAD regions. None disables an admission
    limit; zero prohibits any corresponding conversion work.
    """
    if isinstance(region, PreparedRegion):
        coordinates = normalize_variables(variables)
        if coordinates is not None and tuple(coordinates) != region.variables:
            raise ValueError(
                "variables= must match the prepared region's coordinate order"
            )
        return region
    coordinates = normalize_variables(variables)
    if isinstance(region, (bool, Boolean, sp.Set)):
        if coordinates is None:
            raise ValueError("variables= is required for formulas and sets")
    else:
        from semialg import CADRegion, Geometry, SemialgebraicRegion

        if isinstance(region, (CADRegion, SemialgebraicRegion)):
            owned = list(region.variables)
            if coordinates is not None and coordinates != owned:
                raise ValueError("variables= must match the region's coordinate order")
            coordinates = owned
        elif isinstance(region, Geometry):
            if coordinates is None or len(coordinates) != region.ambient_dimension():
                raise ValueError("explicit variables must match the ambient dimension")
        else:
            raise TypeError("unsupported region input")
    if any(v.is_real is False for v in coordinates):
        raise ValueError("region coordinates must be real")
    return PreparedRegion(region, coordinates, max_operations, max_quantifiers)


def normalize_region(region, variables):
    """Return exact membership and ordered coordinates, reusing prepared inputs."""
    if isinstance(region, PreparedRegion):
        prepared = prepare_region(region, variables)
        return prepared.membership, list(prepared.variables)
    return _normalize_region(region, variables)
