from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import sympy as sp
from sympy.core.relational import Equality, GreaterThan, Relational, StrictGreaterThan

from .precision import finite_midpoint, outward_float

if TYPE_CHECKING:
    from .diagnostics import SolverDiagnostic


@dataclass
class BranchCase:
    """One expression branch with its assumptions and owned feasibility constraints."""

    assumptions: list[Relational]
    objective: sp.Expr
    constraints: list[Relational]

    def pretty(self) -> str:
        """Format the stored mathematical result and diagnostic metadata for inspection."""
        lines = ["Assumptions:"]
        if self.assumptions:
            for a in self.assumptions:
                lines.append(f"  {sp.sstr(a)}")
        else:
            lines.append("  (none)")
        lines.append("Objective:")
        lines.append(f"  {sp.sstr(self.objective)}")
        lines.append("Constraints:")
        if self.constraints:
            for c in self.constraints:
                lines.append(f"  {sp.sstr(c)}")
        else:
            lines.append("  (none)")
        return "\n".join(lines)


@dataclass
class LiftRecord:
    """One auxiliary expression and its defining equations and source-domain restrictions."""

    aux: sp.Symbol
    original_expr: sp.Expr
    kind: str
    defining_eqs: list[Relational] = field(default_factory=list)
    domain_rels: list[Relational] = field(default_factory=list)
    dependencies: list[sp.Symbol] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def pretty(self) -> str:
        """Format the stored mathematical result and diagnostic metadata for inspection."""
        return "\n".join(
            [
                f"aux: {sp.sstr(self.aux)}",
                f"kind: {self.kind}",
                f"original_expr: {sp.sstr(self.original_expr)}",
                f"dependencies: {[sp.sstr(d) for d in self.dependencies]}",
                f"metadata: {self.metadata}",
                f"defining_eqs: {[sp.sstr(r) for r in self.defining_eqs]}",
                f"domain_rels: {[sp.sstr(r) for r in self.domain_rels]}",
            ]
        )


@dataclass
class LiftedProblem:
    """Algebraized optimization problem containing auxiliary variables, graph relations, branch conditions, and lift metadata."""

    original_objective: sp.Expr
    lifted_objective: sp.Expr
    original_constraints: list[Relational]
    lifted_constraints: list[Relational]
    defining_equalities: list[Equality]
    branch_inequalities: list[GreaterThan]
    strict_inequalities: list[StrictGreaterThan]
    graph_equalities: list[Equality]
    graph_inequalities: list[Relational]
    auxiliary_variables: list[sp.Symbol]
    substitution_map: dict[sp.Expr, sp.Symbol]
    all_variables: list[sp.Symbol] = field(default_factory=list)
    lift_records: list[LiftRecord] = field(default_factory=list)
    preprocessing_notes: list[str] = field(default_factory=list)
    strict_closure_warnings: list[str] = field(default_factory=list)
    original_variables: list[sp.Symbol] = field(default_factory=list)
    parameter_symbols: list[sp.Symbol] = field(default_factory=list)
    constraint_residuals: tuple[sp.Expr, ...] = field(default_factory=tuple)
    polynomial_problem: bool | None = None
    structural_metadata: dict[str, object] = field(default_factory=dict)
    _all_constraints_cache: tuple[Relational, ...] = field(
        default_factory=tuple, repr=False
    )
    _poly_residuals_cache: tuple[sp.Expr, ...] = field(
        default_factory=tuple, repr=False
    )
    _def_eq_residuals: tuple[sp.Expr, ...] = field(default_factory=tuple, repr=False)
    _branch_residuals: tuple[sp.Expr, ...] = field(default_factory=tuple, repr=False)
    _strict_residuals: tuple[sp.Expr, ...] = field(default_factory=tuple, repr=False)
    _lift_eq_residuals: tuple[sp.Expr, ...] = field(default_factory=tuple, repr=False)
    _lift_ineq_residuals: tuple[sp.Expr, ...] = field(default_factory=tuple, repr=False)
    _graph_eq_residuals: tuple[sp.Expr, ...] = field(default_factory=tuple, repr=False)
    _graph_ineq_residuals: tuple[sp.Expr, ...] = field(
        default_factory=tuple, repr=False
    )

    def __post_init__(self) -> None:
        self.refresh_metadata()

    def refresh_metadata(self) -> None:
        """Refresh cached residual and structural metadata after internal rewrites."""
        if not self.original_variables:
            symbols = set(self.original_objective.free_symbols)
            for relation in self.original_constraints:
                symbols.update(relation.free_symbols)
            self.original_variables = sorted(
                symbols, key=lambda symbol: symbol.sort_key()
            )

        constraints = (
            *self.defining_equalities,
            *self.branch_inequalities,
            *self.strict_inequalities,
            *self.graph_equalities,
            *self.graph_inequalities,
            *self.lifted_constraints,
        )
        self._all_constraints_cache = tuple(constraints)
        self.constraint_residuals = tuple(
            sp.expand(relation.lhs - relation.rhs)
            for relation in constraints
            if hasattr(relation, "lhs") and hasattr(relation, "rhs")
        )

        def residuals(relations, kinds=None):
            values = []
            for relation in relations:
                if not hasattr(relation, "lhs") or not hasattr(relation, "rhs"):
                    continue
                if kinds is not None:
                    if isinstance(relation, Equality):
                        kind = "eq"
                    elif isinstance(
                        relation,
                        (
                            sp.GreaterThan,
                            sp.StrictGreaterThan,
                            sp.LessThan,
                            sp.StrictLessThan,
                        ),
                    ):
                        kind = "ineq"
                    else:
                        continue
                    if kind not in kinds:
                        continue
                value = sp.expand(relation.lhs - relation.rhs)
                if isinstance(relation, (sp.LessThan, sp.StrictLessThan)):
                    value = -value
                values.append(value)
            return tuple(values)

        self._def_eq_residuals = residuals(self.defining_equalities)
        self._branch_residuals = residuals(self.branch_inequalities)
        self._strict_residuals = residuals(self.strict_inequalities)
        self._graph_eq_residuals = residuals(self.graph_equalities)
        self._graph_ineq_residuals = residuals(self.graph_inequalities, {"ineq"})
        self._lift_eq_residuals = residuals(
            [rel for rel in self.lifted_constraints if isinstance(rel, Equality)]
        )
        self._lift_ineq_residuals = residuals(
            [rel for rel in self.lifted_constraints if not isinstance(rel, Equality)],
            {"ineq"},
        )

        polynomial_residuals = []
        for relation in (
            *self.defining_equalities,
            *self.branch_inequalities,
            *self.strict_inequalities,
            *self.lifted_constraints,
        ):
            if hasattr(relation, "lhs") and hasattr(relation, "rhs"):
                residual = sp.expand(relation.lhs - relation.rhs)
                if isinstance(relation, (sp.LessThan, sp.StrictLessThan)):
                    residual = -residual
                if residual.is_polynomial(*self.all_variables):
                    polynomial_residuals.append(residual)
        self._poly_residuals_cache = tuple(polynomial_residuals)

        self.polynomial_problem = bool(
            self.lifted_objective.is_polynomial(*self.all_variables)
            and all(
                residual.is_polynomial(*self.all_variables)
                for residual in self.constraint_residuals
            )
        )
        self.structural_metadata["constraint_count"] = len(self._all_constraints_cache)
        self.structural_metadata["polynomial_problem"] = self.polynomial_problem

    def update_structure(self, **changes) -> None:
        """Apply internal structural rewrites and refresh all derived caches once."""
        allowed = {
            "lifted_constraints",
            "defining_equalities",
            "branch_inequalities",
            "strict_inequalities",
            "graph_equalities",
            "graph_inequalities",
            "auxiliary_variables",
            "all_variables",
            "lift_records",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(
                "unsupported structural field(s): " + ", ".join(sorted(unknown))
            )
        for name, value in changes.items():
            setattr(self, name, value)
        self.refresh_metadata()

    def closure_constraints(self) -> list[Relational]:
        """Return constraints with strict inequalities replaced by their closed counterparts."""

        def close(rel: Relational) -> Relational:
            if isinstance(rel, sp.StrictGreaterThan):
                return sp.Ge(rel.lhs, rel.rhs)
            if isinstance(rel, sp.StrictLessThan):
                return sp.Le(rel.lhs, rel.rhs)
            return rel

        return [close(rel) for rel in self.all_constraints()]

    def has_strict_constraints(self) -> bool:
        """Whether the original or lifted problem retains strict inequality constraints."""
        return bool(self.strict_inequalities) or any(
            isinstance(rel, (sp.StrictLessThan, sp.StrictGreaterThan))
            for rel in self.lifted_constraints + self.graph_inequalities
        )

    def infimum_only_possible(self) -> bool:
        """Whether strict constraints leave possible nonattainment to be checked."""
        return self.has_strict_constraints()

    def all_constraints(self) -> list[Relational]:
        """Return original lift, graph and branch constraints in the cached order."""
        return list(self._all_constraints_cache)

    def polynomial_residuals(self) -> list[sp.Expr]:
        """Return cached residuals for polynomial constraints."""
        return list(self._poly_residuals_cache)

    def branch_residuals(self) -> list[sp.Expr]:
        """Return cached nonnegative residuals for branch inequalities."""
        return list(self._branch_residuals)

    def sos_residuals(self) -> tuple[list[sp.Expr], list[sp.Expr]]:
        """Return cached equality and inequality residuals used by SOS search."""
        equalities = [*self._def_eq_residuals, *self._lift_eq_residuals]
        inequalities = list(self._branch_residuals)
        return equalities, inequalities

    def kkt_residuals(
        self,
        *,
        include_graph_eq: bool = True,
        include_graph_ineq: bool = True,
        close_strict: bool = True,
    ) -> tuple[list[sp.Expr], list[sp.Expr]]:
        """Return cached equality and nonnegative residuals for KKT construction."""
        if self.has_strict_constraints() and not close_strict:
            raise ValueError(
                "Strict inequalities present; set close_strict=True to relax them."
            )
        if any(isinstance(rel, sp.Unequality) for rel in self.all_constraints()):
            raise ValueError(
                "KKT routing does not support disequality (Ne) constraints"
            )
        equalities = [*self._def_eq_residuals, *self._lift_eq_residuals]
        inequalities = [*self._lift_ineq_residuals, *self._branch_residuals]
        if close_strict:
            inequalities.extend(self._strict_residuals)
        if include_graph_eq:
            equalities.extend(self._graph_eq_residuals)
        if include_graph_ineq:
            inequalities.extend(self._graph_ineq_residuals)
        return equalities, inequalities

    def is_polynomial_problem(self) -> bool:
        """Whether the lifted objective and all retained residuals are polynomial."""
        return bool(self.polynomial_problem)


@dataclass
class ProofArtifact:
    """Named piece of evidence attached to an optimality certificate."""

    kind: str
    summary: str
    payload: dict[str, object] = field(default_factory=dict)


@dataclass
class OptimalityCertificate:
    """Proof-oriented summary of primal data, rigorous bounds, gap, status, and supporting artifacts."""

    primal_solution: dict[sp.Symbol, sp.Expr] = field(default_factory=dict)
    primal_value: sp.Expr | None = None
    lower_bound: sp.Expr | None = None
    upper_bound: sp.Expr | None = None
    gap: sp.Expr | None = None
    proof_objects: list[ProofArtifact] = field(default_factory=list)
    status: str = "unknown"

    def pretty(self) -> str:
        """Format the stored mathematical result and diagnostic metadata for inspection."""
        return "\n".join(
            [
                f"status: {self.status}",
                f"primal_solution: {self.primal_solution}",
                f"primal_value: {self.primal_value}",
                f"lower_bound: {self.lower_bound}",
                f"upper_bound: {self.upper_bound}",
                f"gap: {self.gap}",
                f"proof_objects: {[p.kind for p in self.proof_objects]}",
            ]
        )


@dataclass
class KKTSystem:
    """Structured symbolic KKT system with primal variables, multipliers, stationarity, feasibility, and complementarity equations."""

    primal_variables: list[sp.Symbol]
    equality_multipliers: list[sp.Symbol]
    inequality_multipliers: list[sp.Symbol]
    lagrangian: sp.Expr
    stationarity_equations: list[Equality]
    primal_equalities: list[Equality]
    primal_inequalities: list[GreaterThan]
    dual_inequalities: list[GreaterThan]
    complementarity_eqs: list[Equality]


@dataclass
class IntervalBox:
    """A finite Cartesian search box with outward conversion of exact endpoints."""

    bounds: dict[sp.Symbol, tuple[float, float]]

    def __post_init__(self):
        normalized = {}
        for variable, pair in self.bounds.items():
            if not isinstance(variable, sp.Symbol):
                raise TypeError("IntervalBox keys must be SymPy symbols.")
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("Each interval bound must be a (lower, upper) pair.")
            lower, upper = pair
            try:
                lower = outward_float(lower, -1)
                upper = outward_float(upper, 1)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError("Interval bounds must be real numeric values.") from exc
            if not math.isfinite(lower) or not math.isfinite(upper):
                raise ValueError("Interval bounds must be finite.")
            if lower > upper:
                raise ValueError(f"Lower bound exceeds upper bound for {variable}.")
            normalized[variable] = (lower, upper)
        self.bounds = normalized

    def width(self, var: sp.Symbol) -> float:
        """Width of the selected coordinate interval."""
        a, b = self.bounds[var]
        return b - a

    def max_width_var(self) -> sp.Symbol:
        """Coordinate with the widest current interval."""
        return max(self.bounds, key=self.width)

    def split(self) -> tuple[IntervalBox, IntervalBox]:
        """Bisect the widest coordinate while retaining the entire parent box."""
        var = self.max_width_var()
        a, b = self.bounds[var]
        mid = finite_midpoint(a, b)
        if not a < mid < b:
            raise ValueError("The selected interval has no interior float midpoint.")
        left = dict(self.bounds)
        right = dict(self.bounds)
        left[var] = (a, mid)
        right[var] = (mid, b)
        return IntervalBox(left), IntervalBox(right)

    def midpoint_subs(self) -> dict[sp.Symbol, float]:
        """Return the coordinate midpoint mapping used for numerical proposals."""
        return {v: finite_midpoint(a, b) for v, (a, b) in self.bounds.items()}


@dataclass
class IntervalResult:
    """Branch-and-bound result with lower/upper bounds, best point, rigor provenance, and search statistics."""

    best_lower_bound: float
    best_upper_bound: float
    best_point: dict[sp.Symbol, float]
    boxes_processed: int
    remaining_boxes: int
    feasible_boxes: int
    status: str
    lower_bound_rigorous: bool = False
    upper_bound_rigorous: bool = False
    lower_bound_method: str = "none"
    interval_backend: str = "unknown"
    contractions_applied: int = 0
    contraction_method: str = "none"


@dataclass
class CandidateSolution:
    """A screened proposal with objective data, multipliers and local classification.

    ``feasibility_checked`` records the candidate screen, which can use numerical
    tolerance. Exact certification checks source feasibility and value again.
    ``objective_value_numeric`` is None when parameters prevent evaluation;
    symbolic ordering then uses exact comparisons only.
    """

    primal_values: dict[sp.Symbol, sp.Expr]
    multiplier_values: dict[sp.Symbol, sp.Expr]
    objective_value: sp.Expr
    objective_value_numeric: float | None
    feasibility_checked: bool
    source: str
    local_classification: str | None = None
    original_variable_values: dict[sp.Symbol, sp.Expr] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Integrated optimization result containing candidates, bounds, diagnostics, and certificate evidence."""

    lifted_problem: LiftedProblem
    kkt_system: KKTSystem | None
    candidates: list[CandidateSolution]
    best_candidate: CandidateSolution | None
    interval_result: IntervalResult | None
    inferred_box_bounds: dict[sp.Symbol, tuple[float, float]]
    certified: bool
    certification_gap: float | None
    status: str
    critical_point_results: list[object] = field(default_factory=list)
    sos_result: object | None = None
    sos_lower_bound: sp.Expr | None = None
    sos_bound_numeric: float | None = None
    certificate: OptimalityCertificate | None = None
    diagnostics: list[SolverDiagnostic] = field(default_factory=list)
    semialg_result: object | None = None
    semialg_range: object | None = None
    optimum_value: sp.Expr | None = None
    attained: bool | None = None
    working_precision: int | None = None
    precision_provenance: dict[str, object] = field(default_factory=dict)
    structural_result: object | None = None
    optimization_sense: str = "minimize"
    kkt_objective: sp.Expr | None = None
    solver_plan: object | None = None
    method_label: str | None = None

    optimizer_domain: object = None
    region_preparation: object = None

    @property
    def optimizer_locus(self):
        """Return the complete exact level-set formula when a global value is proved.

        This describes continua and disconnected loci, including points that
        candidate enumeration did not represent. Unknown values return None;
        empty or unbounded problems have no finite optimizer.
        """
        if not self.certified or self.optimum_value is None:
            return None
        if self.optimum_value in (sp.oo, -sp.oo) or self.attained is False:
            return sp.false
        domain = self.optimizer_domain
        if domain is None:
            domain = sp.And(*self.lifted_problem.original_constraints)
        return sp.And(
            domain,
            sp.Eq(
                self.lifted_problem.original_objective,
                self.optimum_value,
                evaluate=False,
            ),
        )

    @property
    def infeasible(self) -> bool:
        """Whether the extended-real result certifies an empty feasible set."""
        if not self.certified or self.optimum_value is None:
            return False
        return bool(
            (self.optimization_sense == "minimize" and self.optimum_value is sp.oo)
            or (self.optimization_sense == "maximize" and self.optimum_value is -sp.oo)
        )

    @property
    def unbounded(self) -> bool:
        """Whether the objective is certified unbounded in the requested sense."""
        if not self.certified or self.optimum_value is None or self.infeasible:
            return False
        return bool(
            (self.optimization_sense == "minimize" and self.optimum_value is -sp.oo)
            or (self.optimization_sense == "maximize" and self.optimum_value is sp.oo)
        )

    @property
    def method(self) -> str:
        """Return a stable public label for the route that produced the result."""
        if self.method_label is not None:
            return self.method_label
        status = self.status.lower()
        for token, label in (
            ("piecewise", "branch"),
            ("univariate", "univariate"),
            ("semialg", "semialg"),
            ("sos", "sos"),
            ("interval", "interval"),
            ("qp", "qp"),
            ("lp", "lp"),
            ("kkt", "kkt"),
        ):
            if token in status:
                return label
        if self.structural_result is not None:
            kind = getattr(self.structural_result, "kind", None)
            if kind:
                return str(kind)
        if self.semialg_result is not None or self.semialg_range is not None:
            return "semialg"
        if self.sos_result is not None:
            return "sos"
        if self.interval_result is not None:
            return "interval"
        if self.kkt_system is not None:
            return "kkt"
        return "unknown"

    def explain(self) -> str:
        """Return a concise user-facing explanation of route and certification."""
        sense_word = "minimum" if self.optimization_sense == "minimize" else "maximum"
        if self.infeasible:
            conclusion = "the feasible set is empty"
        elif self.unbounded:
            conclusion = f"the objective is unbounded for {self.optimization_sense}"
        elif self.optimum_value is not None:
            if self.attained is True:
                attainment = "attained"
            elif self.attained is False:
                attainment = "not attained"
            else:
                attainment = "attainment unknown"
            conclusion = (
                f"{sense_word} value {sp.sstr(self.optimum_value)} ({attainment})"
            )
        else:
            conclusion = "no certified extremum value was obtained"
        proof = "certified" if self.certified else "not globally certified"
        planner = ""
        if self.solver_plan is not None:
            planner = f"; {self.solver_plan.explain()}"
        return f"method={self.method}; {conclusion}; {proof}; status={self.status}{planner}"
