from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple
import heapq
import math

import sympy as sp
from sympy.polys.polyerrors import PolynomialError
from sympy.core.relational import Relational

from .problem_models import IntervalBox, IntervalResult
from .numeric_utils import eval_float
from .relation_checks import normalize_relational, relation_holds
from .verified_intervals import (
    EvalConfig,
    VerifiedBound,
    arb_available,
    enclose_expression,
    verify_point,
)


@lru_cache(maxsize=4096)
def _precondition_forms(expr: sp.Expr) -> Tuple[sp.Expr, ...]:
    forms = [expr]
    transforms = (sp.factor_terms, sp.cancel, sp.trigsimp)
    for transform in transforms:
        try:
            candidate = transform(expr)
        except (TypeError, ValueError, PolynomialError):
            continue
        if candidate == expr:
            continue
        if candidate.free_symbols != expr.free_symbols:
            continue
        if sp.count_ops(candidate) <= max(2, int(1.25 * sp.count_ops(expr))):
            forms.append(candidate)
    unique = []
    seen = set()
    for form in forms:
        key = sp.srepr(form)
        if key not in seen:
            seen.add(key)
            unique.append(form)
    return tuple(unique)


@dataclass(frozen=True)
class IntervalEvaluator:
    expr: sp.Expr
    variables: Tuple[sp.Symbol, ...]
    forms: Tuple[sp.Expr, ...]
    config: EvalConfig = EvalConfig()

    @classmethod
    def compile(cls, expr: sp.Expr, config: EvalConfig = EvalConfig()):
        variables = tuple(sorted(expr.free_symbols, key=sp.sstr))
        return cls(expr, variables, _precondition_forms(expr), config)

    @property
    def dependencies(self):
        return frozenset(self.variables)

    def evaluate(self, box: IntervalBox) -> VerifiedBound:
        bounds = tuple(box.bounds[var] for var in self.variables)
        primary = enclose_expression(self.expr, self.variables, bounds, self.config)
        # Algebraically transformed forms are used only when the original domain
        # is rigorously valid on the box; this prevents cancellation from erasing
        # a removable singularity in the original expression.
        if not primary.rigorous or primary.domain != "valid":
            return primary
        best = primary
        for form in self.forms[1:]:
            alternate = enclose_expression(form, self.variables, bounds, self.config)
            if alternate.rigorous and alternate.domain == "valid":
                best = best.intersect(alternate)
        return best


@dataclass(frozen=True)
class EqualityCorrection:
    symbol: sp.Symbol
    expression: sp.Expr


@dataclass(frozen=True)
class ConstraintEval:
    relation: Relational
    kind: str
    residual: sp.Expr
    evaluator: IntervalEvaluator
    equality_corrections: Tuple[EqualityCorrection, ...] = ()


@dataclass(frozen=True)
class ContractionResult:
    """Result of a verified equality-system box contraction.

    ``infeasible`` means the interval Newton/Krawczyk image proves that the
    supplied equality system has no zero in the input box.  A changed box is
    always a subset guaranteed to contain every zero that was in the original
    box.
    """

    box: IntervalBox
    changed: bool
    infeasible: bool
    method: str
    iterations: int = 0


def _outward(value: float, direction: int) -> float:
    target = float("-inf") if direction < 0 else float("inf")
    return math.nextafter(float(value), target)


def _iadd(a, b):
    return (_outward(a[0] + b[0], -1), _outward(a[1] + b[1], 1))


def _isub(a, b):
    return (_outward(a[0] - b[1], -1), _outward(a[1] - b[0], 1))


def _imul(a, b):
    values = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return (_outward(min(values), -1), _outward(max(values), 1))


def _iscale(c: float, a):
    if c >= 0:
        return (_outward(c * a[0], -1), _outward(c * a[1], 1))
    return (_outward(c * a[1], -1), _outward(c * a[0], 1))


def _idiv_point_by_interval(value, denom):
    if denom[0] <= 0 <= denom[1]:
        return None
    inv = (1.0 / denom[1], 1.0 / denom[0])
    if inv[0] > inv[1]:
        inv = (inv[1], inv[0])
    return _imul(value, inv)


def _intersect_interval(a, b):
    low = max(a[0], b[0])
    high = min(a[1], b[1])
    return None if low > high else (low, high)


def _point_interval(expr: sp.Expr, point: Dict[sp.Symbol, float], config: EvalConfig):
    evaluator = IntervalEvaluator.compile(expr, config)
    box = IntervalBox({symbol: (value, value) for symbol, value in point.items()})
    bound = evaluator.evaluate(box)
    if not bound.rigorous or bound.domain != "valid":
        return None
    return (bound.lower, bound.upper)


def _univariate_newton_step(
    box: IntervalBox,
    equation: sp.Expr,
    variable: sp.Symbol,
    config: EvalConfig,
):
    low, high = box.bounds[variable]
    midpoint = 0.5 * (low + high)
    point = dict(box.midpoint_subs())
    point[variable] = midpoint
    fmid = _point_interval(equation, point, config)
    if fmid is None:
        return box, False, False
    derivative = IntervalEvaluator.compile(
        sp.diff(equation, variable), config
    ).evaluate(box)
    if not derivative.rigorous or derivative.domain != "valid":
        return box, False, False
    quotient = _idiv_point_by_interval(fmid, (derivative.lower, derivative.upper))
    if quotient is None:
        return box, False, False
    newton = _isub((midpoint, midpoint), quotient)
    contracted = _intersect_interval((low, high), newton)
    if contracted is None:
        return box, False, True
    if contracted == (low, high):
        return box, False, False
    bounds = dict(box.bounds)
    bounds[variable] = contracted
    return IntervalBox(bounds), True, False


def _krawczyk_step(
    box: IntervalBox,
    equations: Sequence[sp.Expr],
    variables: Sequence[sp.Symbol],
    config: EvalConfig,
):
    """One verified Krawczyk inclusion step for a square equality system."""
    import numpy as np

    n = len(variables)
    if n == 0 or len(equations) != n:
        return box, False, False
    midpoint = {v: 0.5 * sum(box.bounds[v]) for v in variables}
    f_intervals = []
    jac = []
    midpoint_jac = np.zeros((n, n), dtype=float)
    for i, equation in enumerate(equations):
        fmid = _point_interval(equation, midpoint, config)
        if fmid is None:
            return box, False, False
        f_intervals.append(fmid)
        row = []
        for j, variable in enumerate(variables):
            bound = IntervalEvaluator.compile(
                sp.diff(equation, variable), config
            ).evaluate(box)
            if not bound.rigorous or bound.domain != "valid" or not bound.finite:
                return box, False, False
            interval = (bound.lower, bound.upper)
            row.append(interval)
            midpoint_jac[i, j] = 0.5 * (bound.lower + bound.upper)
        jac.append(row)
    try:
        C = np.linalg.inv(midpoint_jac)
    except np.linalg.LinAlgError:
        return box, False, False
    if not np.all(np.isfinite(C)):
        return box, False, False

    # y = m - C F(m)
    y = []
    for i, variable in enumerate(variables):
        correction = (0.0, 0.0)
        for j in range(n):
            correction = _iadd(correction, _iscale(float(C[i, j]), f_intervals[j]))
        y.append(_isub((midpoint[variable], midpoint[variable]), correction))

    # R = I - C J(X), then K(X) = y + R (X-m).
    radius = [
        (box.bounds[v][0] - midpoint[v], box.bounds[v][1] - midpoint[v])
        for v in variables
    ]
    image = []
    for i in range(n):
        acc = y[i]
        for k in range(n):
            cij = (0.0, 0.0)
            for j in range(n):
                cij = _iadd(cij, _iscale(float(C[i, j]), jac[j][k]))
            identity = (1.0, 1.0) if i == k else (0.0, 0.0)
            rik = _isub(identity, cij)
            acc = _iadd(acc, _imul(rik, radius[k]))
        image.append(acc)

    bounds = dict(box.bounds)
    changed = False
    for variable, kint in zip(variables, image):
        contracted = _intersect_interval(bounds[variable], kint)
        if contracted is None:
            return box, False, True
        if contracted != bounds[variable]:
            changed = True
            bounds[variable] = contracted
    return IntervalBox(bounds), changed, False


def contract_equalities(
    box: IntervalBox,
    equations: Sequence[sp.Expr],
    variables: Sequence[sp.Symbol] | None = None,
    precision: int = 128,
    max_iterations: int = 4,
) -> ContractionResult:
    """Contract a finite box around zeros of equality expressions.

    Scalar equalities first receive interval-Newton steps.  A square coupled
    system then receives Krawczyk steps.  Only rigorous interval evaluations
    are used, so a returned contraction cannot discard an actual zero.
    """
    if variables is None:
        variables = tuple(box.bounds)
    else:
        variables = tuple(variables)
    equations = tuple(sp.expand(eq) for eq in equations)
    config = EvalConfig(precision=precision)
    current = box
    any_changed = False
    used = []
    for iteration in range(1, max_iterations + 1):
        changed = False
        for equation in equations:
            deps = [v for v in variables if v in equation.free_symbols]
            if len(deps) == 1:
                current, did_change, infeasible = _univariate_newton_step(
                    current, equation, deps[0], config
                )
                if infeasible:
                    return ContractionResult(
                        current, any_changed, True, "interval-newton", iteration
                    )
                if did_change:
                    changed = True
                    any_changed = True
                    if "interval-newton" not in used:
                        used.append("interval-newton")
        if len(equations) == len(variables) and len(variables) > 1:
            current, did_change, infeasible = _krawczyk_step(
                current, equations, variables, config
            )
            if infeasible:
                return ContractionResult(
                    current, any_changed, True, "krawczyk", iteration
                )
            if did_change:
                changed = True
                any_changed = True
                if "krawczyk" not in used:
                    used.append("krawczyk")
        if not changed:
            return ContractionResult(
                current, any_changed, False, "+".join(used) or "none", iteration
            )
    return ContractionResult(
        current, any_changed, False, "+".join(used) or "none", max_iterations
    )


@dataclass
class BoxState:
    box: IntervalBox
    objective_bound: VerifiedBound
    constraint_bounds: Tuple[VerifiedBound, ...]


def _compile_constraints(
    constraints: Sequence[Relational],
    config: EvalConfig,
) -> List[ConstraintEval]:
    """Compile interval residual evaluators and reusable equality corrections."""
    compiled = []
    for relation in constraints:
        kind, residual = normalize_relational(relation)
        corrections = []
        if kind == "eq":
            for symbol in sorted(residual.free_symbols, key=sp.sstr):
                try:
                    solutions = sp.solve(sp.Eq(residual, 0), symbol, dict=False)
                except (TypeError, ValueError, ArithmeticError, NotImplementedError):
                    continue
                if len(solutions) == 1 and not solutions[0].has(symbol):
                    corrections.append(EqualityCorrection(symbol, solutions[0]))
        compiled.append(
            ConstraintEval(
                relation,
                kind,
                residual,
                IntervalEvaluator.compile(residual, config),
                tuple(corrections),
            )
        )
    return compiled


def _constraint_status(kind: str, bound: VerifiedBound) -> str:
    if bound.rigorous and bound.domain == "invalid":
        return "infeasible"
    if not bound.rigorous or bound.domain != "valid":
        return "unknown"
    low, high = bound.lower, bound.upper
    if kind == "eq":
        return "infeasible" if low > 0 or high < 0 else "unknown"
    if kind == "ge":
        if high < 0:
            return "infeasible"
        return "feasible" if low >= 0 else "unknown"
    if kind == "gt":
        if high <= 0:
            return "infeasible"
        return "feasible" if low > 0 else "unknown"
    if kind == "neq":
        return "feasible" if low > 0 or high < 0 else "unknown"
    return "unknown"


def _state_status(
    constraints: Sequence[ConstraintEval],
    bounds: Sequence[VerifiedBound],
) -> str:
    status = "feasible"
    for constraint, bound in zip(constraints, bounds):
        current = _constraint_status(constraint.kind, bound)
        if current == "infeasible":
            return "infeasible"
        if current == "unknown":
            status = "unknown"
    return status


def _build_state(
    box: IntervalBox,
    objective: IntervalEvaluator,
    constraints: Sequence[ConstraintEval],
) -> BoxState:
    return BoxState(
        box=box,
        objective_bound=objective.evaluate(box),
        constraint_bounds=tuple(item.evaluator.evaluate(box) for item in constraints),
    )


def _child_state(
    box: IntervalBox,
    split_var: sp.Symbol,
    parent: BoxState,
    objective: IntervalEvaluator,
    constraints: Sequence[ConstraintEval],
) -> BoxState:
    objective_bound = (
        objective.evaluate(box)
        if split_var in objective.dependencies
        else parent.objective_bound
    )
    constraint_bounds = []
    for item, previous in zip(constraints, parent.constraint_bounds):
        constraint_bounds.append(
            item.evaluator.evaluate(box)
            if split_var in item.evaluator.dependencies
            else previous
        )
    return BoxState(box, objective_bound, tuple(constraint_bounds))


def _candidate_from_midpoint(
    box: IntervalBox,
    constraints: Sequence[ConstraintEval],
) -> Dict[sp.Symbol, float]:
    subs = dict(box.midpoint_subs())
    for _ in range(8):
        changed = False
        for item in constraints:
            if item.kind != "eq":
                continue
            for correction in item.equality_corrections:
                if correction.symbol not in subs:
                    continue
                try:
                    value = eval_float(correction.expression.subs(subs))
                except (TypeError, ValueError, ArithmeticError, NotImplementedError):
                    continue
                if abs(subs.get(correction.symbol, value) - value) > 1e-12:
                    subs[correction.symbol] = value
                    changed = True
                    break
        if not changed:
            break
    return subs


def _box_priority(state: BoxState) -> float:
    bound = state.objective_bound
    width = bound.width if bound.finite else float("inf")
    max_width = max(
        (upper - lower for lower, upper in state.box.bounds.values()),
        default=0.0,
    )
    if not math.isfinite(width):
        width = 1e12
    return -(width + 0.25 * max_width)


def _compile_trig_split_info(
    expressions: Sequence[sp.Expr],
    variables: Sequence[sp.Symbol],
) -> Dict[sp.Symbol, Tuple[bool, bool]]:
    info = {}
    for variable in variables:
        has_quarter_period = any(
            expr.has(sp.sin(variable)) or expr.has(sp.cos(variable))
            for expr in expressions
        )
        has_tan = any(expr.has(sp.tan(variable)) for expr in expressions)
        info[variable] = (has_quarter_period, has_tan)
    return info


def _trig_split_points(
    trig_info: Dict[sp.Symbol, Tuple[bool, bool]],
    box: IntervalBox,
    variable: sp.Symbol,
):
    points = set()
    low, high = box.bounds[variable]
    has_quarter_period, has_tan = trig_info.get(variable, (False, False))
    if has_quarter_period:
        first = math.floor(low / (math.pi / 2)) - 1
        last = math.ceil(high / (math.pi / 2)) + 1
        for index in range(first, last + 1):
            point = index * math.pi / 2
            if low + 1e-12 < point < high - 1e-12:
                points.add(float(point))
    if has_tan:
        first = math.floor((low - math.pi / 2) / math.pi) - 1
        last = math.ceil((high - math.pi / 2) / math.pi) + 1
        for index in range(first, last + 1):
            point = math.pi / 2 + index * math.pi
            if low + 1e-12 < point < high - 1e-12:
                points.add(float(point))
    return sorted(points)


@lru_cache(maxsize=2048)
def _stationary_points(expr: sp.Expr, variable: sp.Symbol) -> Tuple[float, ...]:
    if expr.free_symbols - {variable}:
        return ()
    try:
        derivative = sp.diff(expr, variable)
        if derivative == 0 or not derivative.is_polynomial(variable):
            return ()
        poly = sp.Poly(derivative, variable)
        if poly.degree() > 4:
            return ()
        roots = sp.solve(sp.Eq(derivative, 0), variable)
        values = []
        for root in roots:
            if root.is_real is False or root.free_symbols:
                continue
            numeric = float(sp.N(root, 30))
            if math.isfinite(numeric):
                values.append(numeric)
        return tuple(sorted(set(values)))
    except (TypeError, ValueError, PolynomialError, NotImplementedError):
        return ()


def _symbolic_split_points(
    expressions: Sequence[sp.Expr],
    box: IntervalBox,
    variable: sp.Symbol,
):
    low, high = box.bounds[variable]
    points = []
    for expr in expressions:
        points.extend(
            point
            for point in _stationary_points(expr, variable)
            if low + 1e-12 < point < high - 1e-12
        )
    return sorted(set(points))


def _bound_scale(bound: VerifiedBound) -> float:
    if not bound.finite:
        return 10.0
    magnitude = max(abs(bound.lower), abs(bound.upper), 1e-12)
    width = max(0.0, bound.width)
    return min(10.0, magnitude + width)


def _split_score(
    state: BoxState,
    variable: sp.Symbol,
    objective_deriv: IntervalEvaluator | None,
    constraint_derivs: Sequence[IntervalEvaluator | None],
    constraints: Sequence[ConstraintEval],
) -> float:
    box = state.box
    widths = [box.width(item) for item in box.bounds]
    max_width = max(widths, default=1.0) or 1.0
    width_score = box.width(variable) / max_width
    score = 1.0 + 2.0 * width_score

    if objective_deriv is not None:
        bound = objective_deriv.evaluate(box)
        score += width_score * min(8.0, _bound_scale(bound))
        if (
            bound.rigorous
            and bound.domain == "valid"
            and bound.lower <= 0 <= bound.upper
        ):
            score += 1.5

    for item, deriv, residual_bound in zip(
        constraints, constraint_derivs, state.constraint_bounds
    ):
        if variable not in item.evaluator.dependencies:
            continue
        score += 0.25
        if _constraint_status(item.kind, residual_bound) == "unknown":
            score += 1.0
        if residual_bound.domain != "valid":
            score += 2.0
        if deriv is not None:
            score += 0.25 * width_score * min(8.0, _bound_scale(deriv.evaluate(box)))
    return score


def _split_choice(
    state: BoxState,
    expressions: Sequence[sp.Expr],
    derivs: Dict[sp.Symbol, IntervalEvaluator],
    constraint_derivs: Dict[sp.Symbol, Tuple[IntervalEvaluator | None, ...]],
    constraints: Sequence[ConstraintEval],
    trig_info: Dict[sp.Symbol, Tuple[bool, bool]] | None = None,
):
    box = state.box
    if trig_info is None:
        trig_info = _compile_trig_split_info(expressions, tuple(box.bounds))
    scores = {}
    special_points = {}
    for variable in box.bounds:
        trig_points = _trig_split_points(trig_info, box, variable)
        symbolic_points = _symbolic_split_points(expressions, box, variable)
        points = sorted(set([*trig_points, *symbolic_points]))
        special_points[variable] = points
        scores[variable] = _split_score(
            state,
            variable,
            derivs.get(variable),
            constraint_derivs.get(variable, ()),
            constraints,
        ) + (3.0 if points else 0.0)

    best_var = max(scores, key=scores.get)
    low, high = box.bounds[best_var]
    points = special_points[best_var]
    if points:
        midpoint = 0.5 * (low + high)
        split = min(points, key=lambda point: abs(point - midpoint))
    else:
        split = 0.5 * (low + high)
    return best_var, split


def _split_state(
    state: BoxState,
    objective: IntervalEvaluator,
    constraints: Sequence[ConstraintEval],
    expressions: Sequence[sp.Expr],
    trig_info: Dict[sp.Symbol, Tuple[bool, bool]],
    derivs: Dict[sp.Symbol, IntervalEvaluator],
    constraint_derivs: Dict[sp.Symbol, Tuple[IntervalEvaluator | None, ...]],
):
    variable, split = _split_choice(
        state, expressions, derivs, constraint_derivs, constraints, trig_info
    )
    low, high = state.box.bounds[variable]
    left_bounds = dict(state.box.bounds)
    right_bounds = dict(state.box.bounds)
    left_bounds[variable] = (low, split)
    right_bounds[variable] = (split, high)
    return (
        _child_state(IntervalBox(left_bounds), variable, state, objective, constraints),
        _child_state(
            IntervalBox(right_bounds), variable, state, objective, constraints
        ),
    )


def interval_optimize(
    objective: sp.Expr,
    constraints: Sequence[Relational],
    box_bounds: Dict[sp.Symbol, Tuple[float, float]],
    max_boxes: int = 200,
    tol: float = 1e-3,
    precision: int = 128,
) -> IntervalResult:
    """Minimize an expression over a finite box with verified branch-and-bound.

    Rigorous enclosures come from exact polynomial Bernstein bounds and, when
    available, python-flint/Arb real-ball evaluation with derivative/Taylor
    tightening. Heuristic samples may improve upper bounds but cannot create a
    rigorous lower-bound certificate."""
    if not isinstance(max_boxes, int) or max_boxes <= 0:
        raise ValueError("max_boxes must be a positive integer.")
    if not isinstance(precision, int) or precision <= 0:
        raise ValueError("precision must be a positive integer.")
    if not isinstance(tol, (int, float)) or not math.isfinite(float(tol)):
        raise ValueError("tol must be a finite positive number.")
    if float(tol) <= 0:
        raise ValueError("tol must be a finite positive number.")

    if isinstance(constraints, Relational):
        constraints = (constraints,)
    else:
        constraints = tuple(constraints)

    required_symbols = set(objective.free_symbols)
    for relation in constraints:
        required_symbols.update(relation.free_symbols)
    missing = required_symbols.difference(box_bounds)
    if missing:
        names = ", ".join(sorted(map(str, missing)))
        raise ValueError(f"Missing finite box bounds for: {names}.")
    initial_box = IntervalBox(dict(box_bounds))

    config = EvalConfig(precision=precision)
    objective_eval = IntervalEvaluator.compile(objective, config)
    compiled = _compile_constraints(constraints, config)
    expressions = [objective] + [item.residual for item in compiled]
    trig_info = _compile_trig_split_info(expressions, tuple(box_bounds))
    derivs = {
        variable: IntervalEvaluator.compile(sp.diff(objective, variable), config)
        for variable in box_bounds
        if variable in objective.free_symbols
    }

    constraint_derivs = {
        variable: tuple(
            IntervalEvaluator.compile(sp.diff(item.residual, variable), config)
            if variable in item.residual.free_symbols
            else None
            for item in compiled
        )
        for variable in box_bounds
    }

    equality_equations = [item.residual for item in compiled if item.kind == "eq"]
    contraction_count = 0
    contraction_methods = set()
    if equality_equations:
        contraction = contract_equalities(
            initial_box, equality_equations, tuple(box_bounds), precision=precision
        )
        if contraction.infeasible:
            return IntervalResult(
                best_lower_bound=float("inf"),
                best_upper_bound=float("inf"),
                best_point={},
                boxes_processed=1,
                remaining_boxes=0,
                feasible_boxes=0,
                status="no_feasible_point_found",
                lower_bound_rigorous=True,
                upper_bound_rigorous=False,
                lower_bound_method="equality-contractor",
                interval_backend="arb"
                if arb_available()
                else "polynomial-or-heuristic",
                contractions_applied=1,
                contraction_method=contraction.method,
            )
        if contraction.changed:
            initial_box = contraction.box
            contraction_count += 1
            contraction_methods.add(contraction.method)

    first = _build_state(initial_box, objective_eval, compiled)
    pending: List[Tuple[float, int, BoxState]] = [(_box_priority(first), 0, first)]
    leaves: List[BoxState] = []
    counter = 0
    processed = 0
    feasible_boxes = 0
    best_upper = float("inf")
    best_point: Dict[sp.Symbol, float] = {}
    upper_rigorous = False
    closed_lowers: List[float] = []
    closed_methods = set()
    unresolved_nonrigorous = False

    while pending and processed < max_boxes:
        _, _, state = heapq.heappop(pending)
        status = _state_status(compiled, state.constraint_bounds)
        if state.objective_bound.domain == "invalid":
            status = "infeasible"
        if status == "infeasible":
            processed += 1
            continue

        candidate = _candidate_from_midpoint(state.box, compiled)
        try:
            value = eval_float(objective.subs(candidate))
            feasible = all(
                relation_holds(relation, candidate, max(1e-8, tol))
                for relation in constraints
            )
            if feasible and value < best_upper:
                best_upper = value
                best_point = dict(candidate)
                upper_rigorous = verify_point(constraints, candidate, precision)
        except (TypeError, ValueError, ArithmeticError, NotImplementedError):
            pass

        if status == "feasible":
            feasible_boxes += 1

        bound = state.objective_bound
        if (
            bound.rigorous
            and bound.domain == "valid"
            and upper_rigorous
            and best_upper < float("inf")
            and bound.lower >= best_upper - tol
        ):
            closed_lowers.append(bound.lower)
            closed_methods.add(bound.method)
            processed += 1
            continue

        max_width = max(state.box.width(variable) for variable in state.box.bounds)
        if max_width <= tol:
            leaves.append(state)
            if bound.rigorous and bound.domain == "valid":
                closed_lowers.append(bound.lower)
                closed_methods.add(bound.method)
            else:
                unresolved_nonrigorous = True
            processed += 1
            continue

        left, right = _split_state(
            state,
            objective_eval,
            compiled,
            expressions,
            trig_info,
            derivs,
            constraint_derivs,
        )
        for child in (left, right):
            if equality_equations:
                contraction = contract_equalities(
                    child.box,
                    equality_equations,
                    tuple(box_bounds),
                    precision=precision,
                    max_iterations=2,
                )
                if contraction.infeasible:
                    contraction_count += 1
                    contraction_methods.add(contraction.method)
                    continue
                if contraction.changed:
                    contraction_count += 1
                    contraction_methods.add(contraction.method)
                    child = _build_state(contraction.box, objective_eval, compiled)
            counter += 1
            heapq.heappush(pending, (_box_priority(child), counter, child))
        processed += 1

    pending_states = [state for _, _, state in pending]
    pending_lowers = []
    pending_methods = set()
    for state in pending_states:
        bound = state.objective_bound
        if bound.rigorous and bound.domain == "valid":
            pending_lowers.append(bound.lower)
            pending_methods.add(bound.method)
        else:
            unresolved_nonrigorous = True

    rigorous_lowers = closed_lowers + pending_lowers
    lower_rigorous = bool(rigorous_lowers) and not unresolved_nonrigorous
    best_lower = min(rigorous_lowers) if lower_rigorous else float("-inf")

    status = "success" if best_upper < float("inf") else "no_feasible_point_found"
    lower_methods = sorted(closed_methods | pending_methods)
    backend = "arb" if arb_available() else "polynomial-or-heuristic"
    return IntervalResult(
        best_lower_bound=best_lower,
        best_upper_bound=best_upper,
        best_point=best_point,
        boxes_processed=processed,
        remaining_boxes=len(pending),
        feasible_boxes=feasible_boxes,
        status=status,
        lower_bound_rigorous=lower_rigorous,
        upper_bound_rigorous=upper_rigorous,
        lower_bound_method="+".join(lower_methods) if lower_methods else "none",
        interval_backend=backend,
        contractions_applied=contraction_count,
        contraction_method="+".join(sorted(contraction_methods))
        if contraction_methods
        else "none",
    )
