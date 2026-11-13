from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from ._symbols import fresh_symbols
from .exact_refinement import refine_exact_system
from .local_optimization import (
    LocalModeResult,
    _make_numeric_functions,
    _normalize_bounds,
    _normalize_initial_point,
    _normalize_variables,
    _symbolic_result,
)
from .precision import precision_provenance


@dataclass(frozen=True)
class ConstrainedLocalResult:
    """Seeded constrained local optimum with exact KKT refinement metadata."""

    local_result: LocalModeResult
    equality_multipliers: tuple[sp.Expr, ...] = ()
    inequality_multipliers: tuple[sp.Expr, ...] = ()
    active_inequalities: tuple[int, ...] = ()
    primal_verified: bool = False
    dual_verified: bool = False
    complementarity_verified: bool = False
    stationarity_verified: bool = False
    kkt_verified: bool = False
    exact_kkt_refined: bool = False
    kkt_residuals: tuple[sp.Expr, ...] = ()
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    @property
    def point(self):
        """The reported local candidate coordinates in the original variable order."""
        return self.local_result.point

    @property
    def mode(self):
        """The local mode value represented by the constrained result."""
        return self.local_result.mode

    @property
    def observed_information(self):
        """The reported constrained curvature or observed-information matrix."""
        return self.local_result.observed_information

    @property
    def success(self) -> bool:
        """Whether the constrained local solver reports a successful termination."""
        return bool(self.local_result.success and self.kkt_verified)


def _exact_zero(expr):
    value = sp.cancel(sp.together(sp.expand(expr)))
    return value == 0 or value.is_zero is True


def _nonnegative(expr) -> bool:
    value = sp.simplify(expr)
    if value.is_nonnegative is True:
        return True
    if value.is_negative is True:
        return False
    try:
        return float(sp.N(value, 30)) >= -1e-12
    except (TypeError, ValueError, OverflowError):
        return False


def _constraint_functions(expressions, variables, kind):
    result = []
    for expr in expressions:
        expression = sp.sympify(expr)
        fun = sp.lambdify(variables, expression, modules="numpy")
        jac_expr = [sp.diff(expression, variable) for variable in variables]
        jac_fun = sp.lambdify(variables, jac_expr, modules="numpy")

        def value(x, fun=fun):
            return float(np.asarray(fun(*x)).reshape(-1)[0])

        def jac(x, jac_fun=jac_fun):
            return np.asarray(jac_fun(*x), dtype=float).reshape(-1)

        result.append({"type": kind, "fun": value, "jac": jac})
    return result


def _multiplier_seed(gradient, eqs, ineqs, active, variables, point, sign):
    xsubs = {var: float(value) for var, value in zip(variables, point)}
    grad = np.asarray([float(sp.N(sign * item.subs(xsubs), 17)) for item in gradient])
    columns = []
    for expr in eqs:
        columns.append(
            np.asarray(
                [float(sp.N(sp.diff(expr, var).subs(xsubs), 17)) for var in variables]
            )
        )
    for index in active:
        columns.append(
            -np.asarray(
                [
                    float(sp.N(sp.diff(ineqs[index], var).subs(xsubs), 17))
                    for var in variables
                ]
            )
        )
    if not columns:
        return np.zeros(0)
    matrix = np.column_stack(columns)
    try:
        solution, *_ = np.linalg.lstsq(matrix, -grad, rcond=None)
    except np.linalg.LinAlgError:
        solution = np.zeros(matrix.shape[1])
    return solution


def _constrained_optimize(
    objective,
    variables,
    initial_point,
    *,
    target,
    equalities=(),
    inequalities=(),
    bounds=None,
    tolerance=1e-10,
    max_iterations=1000,
    exact_refine=True,
    working_precision=None,
    options=None,
):
    """Solve a constrained local problem and verify an exact active-set refinement."""
    objective = sp.sympify(objective)
    variables = _normalize_variables(objective, variables)
    equalities = tuple(sp.sympify(expr) for expr in equalities)
    inequalities = tuple(sp.sympify(expr) for expr in inequalities)
    allowed = set(variables)
    for expr in (*equalities, *inequalities):
        extras = expr.free_symbols.difference(allowed)
        if extras:
            raise ValueError("Constraints contain unresolved parameter symbols.")
    initial = _normalize_initial_point(variables, initial_point)
    scipy_bounds, _symbolic_bounds = _normalize_bounds(variables, bounds)
    provenance = precision_provenance(objective, [], working_precision)
    if not provenance.float64_backends_allowed:
        raise ValueError(
            "Constrained local optimization currently requires float64-compatible inputs."
        )

    sign = 1 if target == "min" else -1
    gradient = tuple(sp.diff(objective, variable) for variable in variables)
    hessian = sp.hessian(objective, variables)
    fun, jac, _ = _make_numeric_functions(objective, gradient, hessian, variables, sign)
    constraints = _constraint_functions(equalities, variables, "eq")
    constraints += _constraint_functions(inequalities, variables, "ineq")

    from scipy.optimize import minimize as scipy_minimize

    scipy_options = dict(options or {})
    scipy_options.setdefault("maxiter", max_iterations)
    scipy_options.setdefault("ftol", tolerance)
    result = scipy_minimize(
        fun,
        initial,
        jac=jac,
        method="SLSQP",
        bounds=scipy_bounds,
        constraints=constraints,
        options=scipy_options,
    )
    numeric = np.asarray(result.x, dtype=float)
    active_tol = max(1e-8, math.sqrt(tolerance))
    ineq_values = [
        float(sp.N(expr.subs(dict(zip(variables, numeric))), 17))
        for expr in inequalities
    ]
    active = tuple(
        index for index, value in enumerate(ineq_values) if abs(value) <= active_tol
    )

    occupied = objective.free_symbols | set(variables)
    for residual in (*equalities, *inequalities):
        occupied.update(residual.free_symbols)
    lam = fresh_symbols(
        (f"lambda_{i + 1}" for i in range(len(equalities))), occupied, real=True
    )
    mu_active = fresh_symbols(
        (f"mu_{i + 1}" for i in range(len(active))), (*occupied, *lam), real=True
    )
    lagrangian = sign * objective
    for mult, expr in zip(lam, equalities):
        lagrangian += mult * expr
    for mult, index in zip(mu_active, active):
        lagrangian -= mult * inequalities[index]
    kkt_eqs = [sp.diff(lagrangian, variable) for variable in variables]
    kkt_eqs += list(equalities)
    kkt_eqs += [inequalities[index] for index in active]

    seed_mult = _multiplier_seed(
        gradient, equalities, inequalities, active, variables, numeric, sign
    )
    seed = list(numeric) + list(seed_mult)
    all_unknowns = (*variables, *lam, *mu_active)
    refined = None
    if exact_refine:
        refined = refine_exact_system(
            kkt_eqs,
            all_unknowns,
            seed,
            tolerance=active_tol,
            max_variables=9,
            max_degree_product=16384,
        )

    if refined is not None and refined.verified and refined.point is not None:
        primal_point = {var: refined.point[var] for var in variables}
        lam_values = tuple(refined.point[mult] for mult in lam)
        active_mu = tuple(refined.point[mult] for mult in mu_active)
        exact_kkt = True
    else:
        primal_point = {
            var: sp.Float(float(value), 17) for var, value in zip(variables, numeric)
        }
        lam_values = tuple(
            sp.Float(float(value), 17) for value in seed_mult[: len(lam)]
        )
        active_mu = tuple(sp.Float(float(value), 17) for value in seed_mult[len(lam) :])
        exact_kkt = False

    full_mu = [sp.Integer(0)] * len(inequalities)
    for index, value in zip(active, active_mu):
        full_mu[index] = value
    full_mu = tuple(full_mu)

    symbolic = _symbolic_result(
        objective,
        gradient,
        hessian,
        variables,
        primal_point,
        sign,
        max(tolerance, 1e-10),
    )
    (
        obj_value,
        grad_at,
        obj_hess,
        curvature,
        eigs,
        _,
        curvature_verified,
        classification,
    ) = symbolic
    try:
        gradient_norm = float(
            np.linalg.norm([float(sp.N(v, 17)) for v in grad_at], ord=np.inf)
        )
    except (TypeError, ValueError, OverflowError):
        gradient_norm = math.inf

    eq_vals = tuple(sp.simplify(expr.subs(primal_point)) for expr in equalities)
    ineq_vals_exact = tuple(
        sp.simplify(expr.subs(primal_point)) for expr in inequalities
    )
    lag_subs = dict(primal_point)
    lag_subs.update(dict(zip(lam, lam_values)))
    lag_subs.update(dict(zip(mu_active, active_mu)))
    stationarity = tuple(
        sp.simplify(expr.subs(lag_subs)) for expr in kkt_eqs[: len(variables)]
    )
    complementarity = tuple(
        sp.simplify(mult * value) for mult, value in zip(full_mu, ineq_vals_exact)
    )
    primal_ok = all(_exact_zero(value) for value in eq_vals) and all(
        _nonnegative(value) for value in ineq_vals_exact
    )
    dual_ok = all(_nonnegative(value) for value in full_mu)
    comp_ok = all(_exact_zero(value) for value in complementarity)
    stat_ok = all(_exact_zero(value) for value in stationarity)
    kkt_ok = primal_ok and dual_ok and comp_ok and stat_ok

    local = LocalModeResult(
        point=primal_point,
        objective_value=obj_value,
        gradient_at_point=grad_at,
        objective_hessian=obj_hess,
        curvature_matrix=curvature,
        curvature_eigenvalues=eigs,
        classification=classification,
        success=bool(result.success and np.all(np.isfinite(numeric))),
        status="exact_constrained_mode"
        if exact_kkt and kkt_ok
        else "numerical_constrained_mode",
        message=str(result.message),
        target=target,
        method="SLSQP",
        gradient_norm=gradient_norm,
        exact_point=primal_point if exact_kkt else None,
        exact_refined=exact_kkt,
        stationarity_verified=stat_ok,
        curvature_verified=curvature_verified,
        iterations=getattr(result, "nit", None),
        function_evaluations=getattr(result, "nfev", None),
        gradient_evaluations=getattr(result, "njev", None),
        precision=provenance,
        diagnostics={
            "active_inequalities": active,
            "solver_success": bool(result.success),
        },
    )
    return ConstrainedLocalResult(
        local_result=local,
        equality_multipliers=lam_values,
        inequality_multipliers=full_mu,
        active_inequalities=active,
        primal_verified=primal_ok,
        dual_verified=dual_ok,
        complementarity_verified=comp_ok,
        stationarity_verified=stat_ok,
        kkt_verified=kkt_ok,
        exact_kkt_refined=exact_kkt,
        kkt_residuals=(*stationarity, *eq_vals, *complementarity),
        diagnostics={"inequality_values": ineq_vals_exact},
    )


def minimize_local_constrained(objective, variables, initial_point, **kwargs):
    """Find a seeded constrained local minimum and refine its KKT system exactly."""
    return _constrained_optimize(
        objective, variables, initial_point, target="min", **kwargs
    )


def maximize_local_constrained(objective, variables, initial_point, **kwargs):
    """Find a seeded constrained local maximum and refine its KKT system exactly."""
    return _constrained_optimize(
        objective, variables, initial_point, target="max", **kwargs
    )
