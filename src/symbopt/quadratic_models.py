"""Validated fixed-coefficient quadratic problems, results and search controls."""

import math
from dataclasses import dataclass, field

import sympy as sp
from sympy.core.relational import Relational

from .input_normalization import (
    assumption_constraints,
    normalize_bounds_input,
    normalize_constraint_branches,
    normalize_variables,
)
from .rule_utils import polynomial_degree_bound
from .symbolic_cache import cached_poly


@dataclass(frozen=True)
class QuadraticProblem:
    """A scalar quadratic objective and quadratic constraints over fixed real data."""

    objective: sp.Expr
    constraints: tuple
    variables: tuple

    @property
    def expressions(self):
        """Original objective and constraint expressions used by quadratic arithmetic."""
        return (self.objective, *(r.lhs - r.rhs for r in self.constraints))

    @property
    def rational(self):
        """Whether every coefficient is rational in the declared decision coordinates."""
        return all(
            all(c.is_Rational for c in cached_poly(e, self.variables).coeffs())
            for e in self.expressions
        )

    @property
    def integer_variables(self):
        """Declared decision coordinates whose intrinsic assumptions imply integrality."""
        return tuple(v for v in self.variables if v.is_integer is True)

    def value(self, point):
        """Evaluate the original quadratic objective at an ordered coordinate tuple."""
        return sp.cancel(
            self.objective.subs(dict(zip(self.variables, point)), simultaneous=True)
        )

    def feasible(self, point):
        """Check exact feasibility of an ordered coordinate tuple in the stored system."""
        if len(point) != len(self.variables) or any(
            sp.sympify(v).is_real is not True or sp.sympify(v).has(sp.Float)
            for v in point
        ):
            return False
        mapping = dict(zip(self.variables, point))
        return all(
            sp.simplify(r.subs(mapping, simultaneous=True)) is sp.true
            for r in self.constraints
        )


def quadratic_problem(
    objective,
    constraints=None,
    variables=None,
    bounds=None,
    *,
    use_symbol_assumptions=True,
    region=None,
):
    """Normalize one conjunction of a fixed real quadratic optimization problem."""
    objective = sp.sympify(objective)
    if region is not None:
        from .region_inputs import normalize_region

        membership, variables = normalize_region(region, variables)
        constraints = [constraints, membership]
    branches = normalize_constraint_branches(constraints)
    if len(branches) != 1:
        raise NotImplementedError(
            "Global quadratic solvers require one constraint conjunction"
        )
    constraints = list(branches[0])
    values = normalize_variables(variables)
    if values is None:
        values = sorted(
            objective.free_symbols.union(*(r.free_symbols for r in constraints)),
            key=sp.default_sort_key,
        )
    if not values or any(v.is_real is False for v in values):
        raise ValueError("Quadratic decision variables must be nonempty real symbols")
    if use_symbol_assumptions:
        constraints.extend(assumption_constraints(values))
    additional, _ = normalize_bounds_input(bounds, values)
    constraints.extend(additional)
    if any(not isinstance(r, Relational) for r in constraints):
        raise TypeError("Quadratic constraints must be relations")
    problem = QuadraticProblem(objective, tuple(constraints), tuple(values))
    for expression in problem.expressions:
        degree = polynomial_degree_bound(expression, problem.variables)
        if degree is None or degree > 2:
            raise NotImplementedError(
                "Objective and constraints must be polynomial of degree at most two"
            )
        coefficients = cached_poly(expression, problem.variables).coeffs()
        if any(
            c.free_symbols or c.is_real is not True or c.is_finite is not True
            for c in coefficients
        ):
            raise NotImplementedError(
                "Global quadratic solvers require fixed finite real coefficients"
            )
    return problem


@dataclass(frozen=True)
class QuadraticControls:
    """Finite search budgets and stopping gaps shared by global quadratic backends."""

    backend: str = "auto"
    max_faces: int = 4096
    max_nodes: int = 512
    time_limit: float | None = None
    absolute_gap: sp.Expr = field(default_factory=lambda: sp.Rational(1, 10**8))
    relative_gap: sp.Expr = field(default_factory=lambda: sp.Rational(1, 10**8))
    certified: bool = False


def quadratic_controls(
    *,
    backend="auto",
    max_faces=4096,
    max_nodes=512,
    time_limit=None,
    absolute_gap=1e-8,
    relative_gap=1e-8,
    certified=False,
):
    """Validate backend choices, finite budgets, tolerances and proof replay policy."""
    if backend not in {"auto", "faces", "scip", "spatial", "off"}:
        raise ValueError("quadratic backend must be auto, faces, scip, spatial or off")
    for name, value in (("max_faces", max_faces), ("max_nodes", max_nodes)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if time_limit is not None and (
        isinstance(time_limit, bool)
        or not math.isfinite(float(time_limit))
        or time_limit <= 0
    ):
        raise ValueError("time_limit must be finite and positive")
    gaps = []
    for name, value in (("absolute_gap", absolute_gap), ("relative_gap", relative_gap)):
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a nonnegative number")
        gap = sp.Rational(str(value)) if isinstance(value, float) else sp.sympify(value)
        if gap.is_Rational is not True or gap < 0:
            raise ValueError(f"{name} must be a finite nonnegative rational")
        gaps.append(gap)
    if not isinstance(certified, bool):
        raise TypeError("certified must be a boolean")
    return QuadraticControls(
        backend,
        max_faces,
        max_nodes,
        None if time_limit is None else float(time_limit),
        *gaps,
        certified,
    )


@dataclass(frozen=True)
class GlobalQuadraticResult:
    """Global search result separating feasible incumbents, bounds and proof status.

    ``objective_value`` is the incumbent value, or -oo for a proved escaping
    ray. An incumbent does not claim that a limited search completed. ``certified`` is opt-in; ``bounds_verified`` records exact bound
    checks even when a positive optimality gap remains.
    """

    status: str
    variables: tuple
    objective_value: sp.Expr | None = None
    point: tuple = ()
    lower_bound: sp.Expr = -sp.oo
    upper_bound: sp.Expr = sp.oo
    backend: str = ""
    nodes: int = 0
    certified: bool = False
    bounds_verified: bool = False
    attained: bool | None = None
    certificate: object = None
    metadata: dict = field(default_factory=dict)

    @property
    def x(self):
        """Return the incumbent vector in the original variable order, or None."""
        return sp.ImmutableMatrix(self.point) if self.point else None

    @property
    def point_mapping(self):
        """Return the incumbent as a symbol-to-value mapping."""
        return dict(zip(self.variables, self.point))

    @property
    def gap(self):
        """Return the absolute primal-dual gap, including infinity for unknown bounds."""
        if self.status == "infeasible" and self.bounds_verified:
            return sp.S.Zero
        if self.lower_bound == self.upper_bound:
            return sp.S.Zero
        return self.upper_bound - self.lower_bound

    @property
    def exact(self):
        """Whether verified bounds prove an exact optimum or global infeasibility."""
        return (
            self.bounds_verified
            and self.status in {"optimal", "infeasible", "unbounded"}
            and (self.status != "optimal" or self.gap == 0)
        )
