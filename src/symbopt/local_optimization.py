from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from .exact_refinement import refine_exact_system
from .input_normalization import require_continuous_variables
from .precision import PrecisionProvenance, precision_provenance


@dataclass(frozen=True)
class LocalModeResult:
    """Structured result for seeded local minimization or maximization.

    ``objective_hessian`` is the Hessian of the objective exactly as supplied.
    ``curvature_matrix`` is the Hessian in minimization convention: it equals
    the objective Hessian for minimization and its negation for maximization.
    Consequently, at a regular posterior mode found by ``maximize_local``,
    ``curvature_matrix`` is the observed negative Hessian used by a Laplace
    approximation.
    """

    point: Mapping[sp.Symbol, sp.Expr]
    objective_value: sp.Expr
    gradient_at_point: tuple[sp.Expr, ...]
    objective_hessian: sp.Matrix
    curvature_matrix: sp.Matrix
    curvature_eigenvalues: tuple[sp.Expr, ...]
    classification: str
    success: bool
    status: str
    message: str
    target: str
    method: str
    gradient_norm: float
    exact_point: Mapping[sp.Symbol, sp.Expr] | None = None
    exact_refined: bool = False
    stationarity_verified: bool = False
    curvature_verified: bool = False
    iterations: int | None = None
    function_evaluations: int | None = None
    gradient_evaluations: int | None = None
    hessian_evaluations: int | None = None
    precision: PrecisionProvenance | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    @property
    def mode(self) -> Mapping[sp.Symbol, sp.Expr]:
        """Alias for ``point`` convenient for probabilistic inference code."""
        return self.point

    @property
    def observed_information(self) -> sp.Matrix:
        """Return curvature in the positive-at-a-mode convention."""
        return self.curvature_matrix


def _normalize_variables(
    objective: sp.Expr,
    variables: sp.Symbol | Sequence[sp.Symbol] | None,
) -> tuple[sp.Symbol, ...]:
    if variables is None:
        values = tuple(sorted(objective.free_symbols, key=lambda item: item.sort_key()))
    elif isinstance(variables, sp.Symbol):
        values = (variables,)
    else:
        values = tuple(variables)
    if not values:
        raise ValueError("At least one optimization variable is required.")
    if len(set(values)) != len(values) or not all(
        isinstance(variable, sp.Symbol) for variable in values
    ):
        raise TypeError("variables must be distinct SymPy symbols.")
    require_continuous_variables(values)
    extras = objective.free_symbols.difference(values)
    if extras:
        names = ", ".join(sp.sstr(symbol) for symbol in sorted(extras, key=sp.sstr))
        raise ValueError(f"Objective contains unresolved parameter symbols: {names}.")
    return values


def _normalize_initial_point(
    variables: Sequence[sp.Symbol],
    initial_point,
) -> np.ndarray:
    if isinstance(initial_point, Mapping):
        missing = [variable for variable in variables if variable not in initial_point]
        if missing:
            names = ", ".join(sp.sstr(variable) for variable in missing)
            raise ValueError(f"initial_point is missing variables: {names}.")
        values = [initial_point[variable] for variable in variables]
    else:
        values = list(initial_point)
        if len(values) != len(variables):
            raise ValueError("initial_point must provide one value per variable.")
    try:
        result = np.asarray([float(sp.N(value, 17)) for value in values], dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("initial_point values must be finite real numbers.") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError("initial_point values must be finite real numbers.")
    return result


def _normalize_bounds(variables: Sequence[sp.Symbol], bounds):
    if bounds is None:
        return None, tuple((None, None) for _ in variables)
    if isinstance(bounds, Mapping):
        pairs = [bounds.get(variable, (None, None)) for variable in variables]
    else:
        pairs = list(bounds)
        if len(pairs) != len(variables):
            raise ValueError("bounds must provide one pair per variable.")

    scipy_pairs = []
    symbolic_pairs = []
    for pair in pairs:
        if pair is None:
            pair = (None, None)
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("Each bound must be a (lower, upper) pair.")
        lower, upper = pair
        lower_expr = None if lower is None else sp.sympify(lower)
        upper_expr = None if upper is None else sp.sympify(upper)
        lower_float = -np.inf if lower is None else float(sp.N(lower_expr, 17))
        upper_float = np.inf if upper is None else float(sp.N(upper_expr, 17))
        if math.isnan(lower_float) or math.isnan(upper_float):
            raise ValueError("Bounds cannot be NaN.")
        if lower_float > upper_float:
            raise ValueError("A lower bound exceeds its upper bound.")
        scipy_pairs.append((lower_float, upper_float))
        symbolic_pairs.append((lower_expr, upper_expr))
    return scipy_pairs, tuple(symbolic_pairs)


def _make_numeric_functions(objective, gradient, hessian, variables, sign):
    objective_fun = sp.lambdify(variables, sign * objective, modules="numpy")
    gradient_fun = sp.lambdify(
        variables, [sign * value for value in gradient], modules="numpy"
    )
    hessian_fun = sp.lambdify(variables, sign * hessian, modules="numpy")

    def fun(values):
        value = np.asarray(objective_fun(*values))
        scalar = float(value.reshape(-1)[0] if value.ndim else value)
        if not math.isfinite(scalar):
            return np.inf
        return scalar

    def jac(values):
        result = np.asarray(gradient_fun(*values), dtype=float).reshape(-1)
        if result.shape != (len(variables),) or not np.all(np.isfinite(result)):
            raise FloatingPointError(
                "Objective gradient is not finite at the trial point."
            )
        return result

    def hess(values):
        result = np.asarray(hessian_fun(*values), dtype=float)
        result = np.reshape(result, (len(variables), len(variables)))
        if not np.all(np.isfinite(result)):
            raise FloatingPointError(
                "Objective Hessian is not finite at the trial point."
            )
        return result

    return fun, jac, hess


def _choose_scipy_method(method: str, has_bounds: bool) -> str:
    if method != "auto":
        return method
    return "L-BFGS-B" if has_bounds else "BFGS"


def _numeric_classification(
    curvature: np.ndarray, tolerance: float
) -> tuple[str, bool]:
    symmetric = 0.5 * (curvature + curvature.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 1.0)
    cutoff = tolerance * scale
    if np.all(eigenvalues > cutoff):
        return "LocalMinimum", True
    if np.all(eigenvalues < -cutoff):
        return "LocalMaximum", True
    if np.any(eigenvalues > cutoff) and np.any(eigenvalues < -cutoff):
        return "SaddlePoint", True
    return "DegenerateOrInconclusive", False


def _exact_pd(matrix: sp.Matrix) -> bool:
    if matrix.rows == 0:
        return True
    try:
        result = matrix.is_positive_definite
    except (TypeError, ValueError, NotImplementedError):
        return False
    return result is True


def _is_exact_zero(expression: sp.Expr) -> bool:
    simplified = sp.cancel(sp.together(sp.expand(expression)))
    return simplified == 0 or simplified.is_zero is True


def _nsimplify_candidate(
    numeric_point: Sequence[float],
    variables: Sequence[sp.Symbol],
    gradient: Sequence[sp.Expr],
    symbolic_bounds,
    tolerance: float,
):
    candidate = {}
    for index, variable in enumerate(variables):
        lower, upper = symbolic_bounds[index]
        value = numeric_point[index]
        if lower is not None and abs(value - float(sp.N(lower, 17))) <= tolerance:
            candidate[variable] = lower
            continue
        if upper is not None and abs(value - float(sp.N(upper, 17))) <= tolerance:
            candidate[variable] = upper
            continue
        try:
            candidate[variable] = sp.nsimplify(value, tolerance=tolerance, full=True)
        except (ValueError, TypeError, AssertionError):
            candidate[variable] = sp.Float(value, 17)

    free_indices = []
    for index, _variable in enumerate(variables):
        lower, upper = symbolic_bounds[index]
        value = numeric_point[index]
        on_lower = (
            lower is not None and abs(value - float(sp.N(lower, 17))) <= tolerance
        )
        on_upper = (
            upper is not None and abs(value - float(sp.N(upper, 17))) <= tolerance
        )
        if not on_lower and not on_upper:
            free_indices.append(index)

    if all(_is_exact_zero(gradient[index].subs(candidate)) for index in free_indices):
        return candidate
    return None


def _symbolic_candidate(
    numeric_point: Sequence[float],
    variables: Sequence[sp.Symbol],
    gradient: Sequence[sp.Expr],
    symbolic_bounds,
    tolerance: float,
):
    active = {}
    free_variables = []
    equations = []
    for index, variable in enumerate(variables):
        lower, upper = symbolic_bounds[index]
        value = numeric_point[index]
        if lower is not None and abs(value - float(sp.N(lower, 17))) <= tolerance:
            active[variable] = lower
        elif upper is not None and abs(value - float(sp.N(upper, 17))) <= tolerance:
            active[variable] = upper
        else:
            free_variables.append(variable)
            equations.append(sp.expand(gradient[index].subs(active)))

    if not free_variables:
        return active
    reduced_equations = [sp.cancel(sp.together(eq)) for eq in equations]
    seed = [numeric_point[variables.index(variable)] for variable in free_variables]
    refined = refine_exact_system(
        reduced_equations,
        free_variables,
        seed,
        tolerance=tolerance,
        max_variables=7,
        max_degree_product=4096,
    )
    if not refined.verified or refined.point is None:
        return None
    candidate = dict(active)
    candidate.update(refined.point)
    return candidate


def _exact_refine(
    numeric_point,
    variables,
    gradient,
    symbolic_bounds,
    tolerance,
):
    candidate = _nsimplify_candidate(
        numeric_point, variables, gradient, symbolic_bounds, tolerance
    )
    if candidate is not None:
        return candidate, "nsimplify"
    candidate = _symbolic_candidate(
        numeric_point, variables, gradient, symbolic_bounds, tolerance
    )
    if candidate is not None:
        return candidate, "symbolic_stationarity"
    return None, None


def _symbolic_result(
    objective,
    gradient,
    hessian,
    variables,
    point,
    sign,
    numeric_tolerance,
):
    objective_value = sp.simplify(objective.subs(point))
    gradient_at_point = tuple(sp.simplify(value.subs(point)) for value in gradient)
    objective_hessian = sp.Matrix(hessian.subs(point)).applyfunc(sp.simplify)
    curvature = sp.Matrix(sign * objective_hessian).applyfunc(sp.simplify)
    try:
        eigenvalue_map = curvature.eigenvals()
        eigenvalues = tuple(
            sp.simplify(value)
            for value, multiplicity in eigenvalue_map.items()
            for _ in range(int(multiplicity))
        )
    except (TypeError, ValueError, NotImplementedError):
        eigenvalues = ()

    stationarity_verified = all(_is_exact_zero(value) for value in gradient_at_point)
    curvature_verified = _exact_pd(curvature)
    if curvature_verified:
        classification = "LocalMinimum"
    else:
        try:
            numeric_curvature = np.asarray(curvature.evalf(), dtype=float)
            classification, _ = _numeric_classification(
                numeric_curvature, numeric_tolerance
            )
        except (TypeError, ValueError, np.linalg.LinAlgError):
            classification = "Inconclusive"
    return (
        objective_value,
        gradient_at_point,
        objective_hessian,
        curvature,
        eigenvalues,
        stationarity_verified,
        curvature_verified,
        classification,
    )


def _local_optimize(
    objective,
    variables,
    initial_point,
    *,
    target: str,
    bounds=None,
    method: str = "auto",
    tolerance: float = 1e-10,
    max_iterations: int = 1000,
    exact_refine: bool = True,
    working_precision: int | None = None,
    options: Mapping[str, object] | None = None,
) -> LocalModeResult:
    """Run local numerical optimization, then refine and classify the candidate."""
    objective = sp.sympify(objective)
    variables = _normalize_variables(objective, variables)
    if (
        not isinstance(max_iterations, int)
        or isinstance(max_iterations, bool)
        or max_iterations <= 0
    ):
        raise ValueError("max_iterations must be a positive integer.")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    if target not in {"min", "max"}:
        raise ValueError("target must be 'min' or 'max'.")

    initial = _normalize_initial_point(variables, initial_point)
    scipy_bounds, symbolic_bounds = _normalize_bounds(variables, bounds)
    has_bounds = scipy_bounds is not None and any(
        np.isfinite(lower) or np.isfinite(upper) for lower, upper in scipy_bounds
    )
    provenance = precision_provenance(objective, [], working_precision)
    if not provenance.float64_backends_allowed:
        raise ValueError(
            "Seeded local optimization currently uses a float64 numerical backend; "
            "high-precision inexact objectives require exact inputs or "
            "working_precision<=16."
        )

    sign = 1 if target == "min" else -1
    gradient = tuple(sp.diff(objective, variable) for variable in variables)
    hessian = sp.hessian(objective, variables)
    fun, jac, hess = _make_numeric_functions(
        objective, gradient, hessian, variables, sign
    )
    scipy_method = _choose_scipy_method(method, has_bounds)

    from scipy.optimize import minimize as scipy_minimize

    scipy_options = dict(options or {})
    scipy_options.setdefault("maxiter", max_iterations)
    kwargs = {
        "fun": fun,
        "x0": initial,
        "jac": jac,
        "method": scipy_method,
        "tol": tolerance,
        "options": scipy_options,
    }
    if has_bounds:
        kwargs["bounds"] = scipy_bounds
    methods_using_hessian = {
        "Newton-CG",
        "dogleg",
        "trust-ncg",
        "trust-exact",
        "trust-krylov",
        "trust-constr",
    }
    if scipy_method in methods_using_hessian:
        kwargs["hess"] = hess

    result = scipy_minimize(**kwargs)
    numeric_point = np.asarray(result.x, dtype=float)
    point = {
        variable: sp.Float(float(numeric_point[index]), 17)
        for index, variable in enumerate(variables)
    }

    refinement_method = None
    exact_point = None
    if exact_refine and np.all(np.isfinite(numeric_point)):
        refine_tolerance = max(1e-12, min(1e-6, math.sqrt(tolerance)))
        exact_point, refinement_method = _exact_refine(
            numeric_point,
            variables,
            gradient,
            symbolic_bounds,
            refine_tolerance,
        )
        if exact_point is not None:
            point = exact_point

    (
        objective_value,
        gradient_at_point,
        objective_hessian,
        curvature,
        eigenvalues,
        stationarity_verified,
        curvature_verified,
        classification,
    ) = _symbolic_result(
        objective,
        gradient,
        hessian,
        variables,
        point,
        sign,
        max(tolerance, 1e-10),
    )

    try:
        gradient_numeric = np.asarray(
            [float(sp.N(value, 17)) for value in gradient_at_point], dtype=float
        )
        gradient_norm = float(np.linalg.norm(gradient_numeric, ord=np.inf))
    except (TypeError, ValueError, OverflowError):
        gradient_norm = math.inf

    solver_success = bool(result.success and np.all(np.isfinite(numeric_point)))
    local_mode_supported = classification == "LocalMinimum"
    success = solver_success and local_mode_supported
    if success and exact_point is not None and curvature_verified:
        status = "exact_local_mode"
    elif success:
        status = "numerical_local_mode"
    elif solver_success:
        status = "stationary_or_degenerate"
    else:
        status = "solver_failure"

    diagnostics = {
        "scipy_status": int(result.status),
        "refinement_method": refinement_method,
        "solver_success": solver_success,
        "raw_objective_value": (
            float(result.fun) if np.isfinite(result.fun) else result.fun
        ),
        "has_bounds": has_bounds,
    }

    return LocalModeResult(
        point=point,
        objective_value=objective_value,
        gradient_at_point=gradient_at_point,
        objective_hessian=objective_hessian,
        curvature_matrix=curvature,
        curvature_eigenvalues=eigenvalues,
        classification=classification,
        success=success,
        status=status,
        message=str(result.message),
        target=target,
        method=scipy_method,
        gradient_norm=gradient_norm,
        exact_point=exact_point,
        exact_refined=exact_point is not None,
        stationarity_verified=stationarity_verified,
        curvature_verified=curvature_verified,
        iterations=getattr(result, "nit", None),
        function_evaluations=getattr(result, "nfev", None),
        gradient_evaluations=getattr(result, "njev", None),
        hessian_evaluations=getattr(result, "nhev", None),
        precision=provenance,
        diagnostics=diagnostics,
    )


def minimize_local(
    objective,
    variables,
    initial_point,
    *,
    bounds=None,
    method: str = "auto",
    tolerance: float = 1e-10,
    max_iterations: int = 1000,
    exact_refine: bool = True,
    working_precision: int | None = None,
    options: Mapping[str, object] | None = None,
) -> LocalModeResult:
    """Find a seeded local minimum using symbolic derivatives and numerical search."""
    return _local_optimize(
        objective,
        variables,
        initial_point,
        target="min",
        bounds=bounds,
        method=method,
        tolerance=tolerance,
        max_iterations=max_iterations,
        exact_refine=exact_refine,
        working_precision=working_precision,
        options=options,
    )


def maximize_local(
    objective,
    variables,
    initial_point,
    *,
    bounds=None,
    method: str = "auto",
    tolerance: float = 1e-10,
    max_iterations: int = 1000,
    exact_refine: bool = True,
    working_precision: int | None = None,
    options: Mapping[str, object] | None = None,
) -> LocalModeResult:
    """Find a seeded local maximum; curvature is returned as ``-H(objective)``."""
    return _local_optimize(
        objective,
        variables,
        initial_point,
        target="max",
        bounds=bounds,
        method=method,
        tolerance=tolerance,
        max_iterations=max_iterations,
        exact_refine=exact_refine,
        working_precision=working_precision,
        options=options,
    )
