from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import sympy as sp
from sympy.core.relational import Equality, GreaterThan, Relational, StrictGreaterThan

if TYPE_CHECKING:
    from .diagnostics import SolverDiagnostic


@dataclass
class BranchCase:
    assumptions: List[Relational]
    objective: sp.Expr
    constraints: List[Relational]

    def pretty(self) -> str:
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
    aux: sp.Symbol
    original_expr: sp.Expr
    kind: str
    defining_eqs: List[Relational] = field(default_factory=list)
    domain_rels: List[Relational] = field(default_factory=list)
    dependencies: List[sp.Symbol] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def pretty(self) -> str:
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
    original_constraints: List[Relational]
    lifted_constraints: List[Relational]
    defining_equalities: List[Equality]
    branch_inequalities: List[GreaterThan]
    strict_inequalities: List[StrictGreaterThan]
    graph_equalities: List[Equality]
    graph_inequalities: List[Relational]
    auxiliary_variables: List[sp.Symbol]
    substitution_map: Dict[sp.Expr, sp.Symbol]
    all_variables: List[sp.Symbol] = field(default_factory=list)
    lift_records: List[LiftRecord] = field(default_factory=list)
    preprocessing_notes: List[str] = field(default_factory=list)
    strict_closure_warnings: List[str] = field(default_factory=list)
    original_variables: List[sp.Symbol] = field(default_factory=list)
    parameter_symbols: List[sp.Symbol] = field(default_factory=list)
    constraint_residuals: Tuple[sp.Expr, ...] = field(default_factory=tuple)
    polynomial_problem: Optional[bool] = None
    structural_metadata: Dict[str, object] = field(default_factory=dict)
    _all_constraints_cache: Tuple[Relational, ...] = field(
        default_factory=tuple, repr=False
    )
    _poly_residuals_cache: Tuple[sp.Expr, ...] = field(
        default_factory=tuple, repr=False
    )
    _def_eq_residuals: Tuple[sp.Expr, ...] = field(default_factory=tuple, repr=False)
    _branch_residuals: Tuple[sp.Expr, ...] = field(default_factory=tuple, repr=False)
    _strict_residuals: Tuple[sp.Expr, ...] = field(default_factory=tuple, repr=False)
    _lift_eq_residuals: Tuple[sp.Expr, ...] = field(default_factory=tuple, repr=False)
    _lift_ineq_residuals: Tuple[sp.Expr, ...] = field(default_factory=tuple, repr=False)
    _graph_eq_residuals: Tuple[sp.Expr, ...] = field(default_factory=tuple, repr=False)
    _graph_ineq_residuals: Tuple[sp.Expr, ...] = field(
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
                    elif isinstance(relation, (sp.GreaterThan, sp.StrictGreaterThan)):
                        kind = "ineq"
                    elif isinstance(relation, (sp.LessThan, sp.StrictLessThan)):
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

    def closure_constraints(self) -> List[Relational]:
        def close(rel: Relational) -> Relational:
            if isinstance(rel, sp.StrictGreaterThan):
                return sp.Ge(rel.lhs, rel.rhs)
            if isinstance(rel, sp.StrictLessThan):
                return sp.Le(rel.lhs, rel.rhs)
            return rel

        return [close(rel) for rel in self.all_constraints()]

    def has_strict_constraints(self) -> bool:
        return bool(self.strict_inequalities) or any(
            isinstance(rel, sp.StrictLessThan) or isinstance(rel, sp.StrictGreaterThan)
            for rel in self.lifted_constraints + self.graph_inequalities
        )

    def infimum_only_possible(self) -> bool:
        return self.has_strict_constraints()

    def all_constraints(self) -> List[Relational]:
        return list(self._all_constraints_cache)

    def polynomial_residuals(self) -> List[sp.Expr]:
        """Return cached residuals for polynomial constraints."""
        return list(self._poly_residuals_cache)

    def branch_residuals(self) -> List[sp.Expr]:
        """Return cached nonnegative residuals for branch inequalities."""
        return list(self._branch_residuals)

    def sos_residuals(self) -> Tuple[List[sp.Expr], List[sp.Expr]]:
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
    ) -> Tuple[List[sp.Expr], List[sp.Expr]]:
        """Return cached equality and nonnegative residuals for KKT construction."""
        if self.strict_inequalities and not close_strict:
            raise ValueError(
                "Strict inequalities present; set close_strict=True to relax them."
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
        return bool(self.polynomial_problem)


@dataclass
class ProofArtifact:
    """Named piece of evidence attached to an optimality certificate."""

    kind: str
    summary: str
    payload: Dict[str, object] = field(default_factory=dict)


@dataclass
class OptimalityCertificate:
    """Proof-oriented summary of primal data, rigorous bounds, gap, status, and supporting artifacts."""

    primal_solution: Dict[sp.Symbol, sp.Expr] = field(default_factory=dict)
    primal_value: Optional[sp.Expr] = None
    lower_bound: Optional[sp.Expr] = None
    upper_bound: Optional[sp.Expr] = None
    gap: Optional[sp.Expr] = None
    proof_objects: List[ProofArtifact] = field(default_factory=list)
    status: str = "unknown"

    def pretty(self) -> str:
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

    primal_variables: List[sp.Symbol]
    equality_multipliers: List[sp.Symbol]
    inequality_multipliers: List[sp.Symbol]
    lagrangian: sp.Expr
    stationarity_equations: List[Equality]
    primal_equalities: List[Equality]
    primal_inequalities: List[GreaterThan]
    dual_inequalities: List[GreaterThan]
    complementarity_eqs: List[Equality]


@dataclass
class IntervalBox:
    bounds: Dict[sp.Symbol, Tuple[float, float]]

    def __post_init__(self):
        normalized = {}
        for variable, pair in self.bounds.items():
            if not isinstance(variable, sp.Symbol):
                raise TypeError("IntervalBox keys must be SymPy symbols.")
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("Each interval bound must be a (lower, upper) pair.")
            lower, upper = pair
            try:
                lower = float(lower)
                upper = float(upper)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError("Interval bounds must be real numeric values.") from exc
            if not math.isfinite(lower) or not math.isfinite(upper):
                raise ValueError("Interval bounds must be finite.")
            if lower > upper:
                raise ValueError(f"Lower bound exceeds upper bound for {variable}.")
            normalized[variable] = (lower, upper)
        self.bounds = normalized

    def width(self, var: sp.Symbol) -> float:
        a, b = self.bounds[var]
        return b - a

    def max_width_var(self) -> sp.Symbol:
        return max(self.bounds, key=self.width)

    def split(self) -> Tuple["IntervalBox", "IntervalBox"]:
        var = self.max_width_var()
        a, b = self.bounds[var]
        mid = 0.5 * (a + b)
        left = dict(self.bounds)
        right = dict(self.bounds)
        left[var] = (a, mid)
        right[var] = (mid, b)
        return IntervalBox(left), IntervalBox(right)

    def midpoint_subs(self) -> Dict[sp.Symbol, float]:
        return {v: 0.5 * (a + b) for v, (a, b) in self.bounds.items()}


@dataclass
class IntervalResult:
    """Branch-and-bound result with lower/upper bounds, best point, rigor provenance, and search statistics."""

    best_lower_bound: float
    best_upper_bound: float
    best_point: Dict[sp.Symbol, float]
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
    """One feasible candidate point together with objective value, source route, multipliers, and local classification."""

    primal_values: Dict[sp.Symbol, sp.Expr]
    multiplier_values: Dict[sp.Symbol, sp.Expr]
    objective_value: sp.Expr
    objective_value_numeric: float
    feasibility_checked: bool
    source: str
    local_classification: Optional[str] = None
    original_variable_values: Dict[sp.Symbol, sp.Expr] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Integrated optimization result containing candidates, bounds, diagnostics, and certificate evidence."""

    lifted_problem: "LiftedProblem"
    kkt_system: Optional["KKTSystem"]
    candidates: List["CandidateSolution"]
    best_candidate: Optional["CandidateSolution"]
    interval_result: Optional["IntervalResult"]
    inferred_box_bounds: Dict[sp.Symbol, Tuple[float, float]]
    certified: bool
    certification_gap: Optional[float]
    status: str
    critical_point_results: List[object] = field(default_factory=list)
    sos_result: Optional[object] = None
    sos_lower_bound: Optional[sp.Expr] = None
    sos_bound_numeric: Optional[float] = None
    certificate: Optional[OptimalityCertificate] = None
    diagnostics: List["SolverDiagnostic"] = field(default_factory=list)
    semialg_result: Optional[object] = None
    semialg_range: Optional[object] = None
    optimum_value: Optional[sp.Expr] = None
    attained: Optional[bool] = None
    working_precision: Optional[int] = None
    precision_provenance: Dict[str, object] = field(default_factory=dict)
    structural_result: Optional[object] = None
    optimization_sense: str = "minimize"
    kkt_objective: Optional[sp.Expr] = None
