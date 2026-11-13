"""Optional numerical global quadratic and mixed-integer solving through SCIP."""

import math

import sympy as sp

from .quadratic_models import GlobalQuadraticResult


def scip_available():
    """Return whether the optional PySCIPOpt native library can be imported."""
    try:
        import pyscipopt
    except (ImportError, OSError):
        return False
    return hasattr(pyscipopt, "Model")


def _expression(expression, variables, mapping):
    poly = sp.Poly(expression, *variables)
    return sum(
        float(c)
        * math.prod(mapping[v] ** power for v, power in zip(variables, powers) if power)
        for powers, c in poly.terms()
    )


def numeric_feasible(problem, point, tolerance=1e-6):
    """Validate a numerical candidate against the original model, including domains."""
    if len(point) != len(problem.variables):
        return False
    if any(not math.isfinite(float(v)) for v in point):
        return False
    if any(
        abs(float(value) - round(float(value))) > tolerance
        for v, value in zip(problem.variables, point)
        if v.is_integer is True
    ):
        return False
    for variable, value in zip(problem.variables, point):
        if (
            variable.is_even is True
            and abs(float(value) / 2 - round(float(value) / 2)) > tolerance
        ):
            return False
        if (
            variable.is_odd is True
            and abs((float(value) - 1) / 2 - round((float(value) - 1) / 2)) > tolerance
        ):
            return False
    substitution = dict(zip(problem.variables, point))
    for relation in problem.constraints:
        value = complex(sp.N((relation.lhs - relation.rhs).subs(substitution), 20))
        if abs(value.imag) > tolerance:
            return False
        if isinstance(relation, sp.Equality):
            if abs(value.real) > tolerance:
                return False
        elif isinstance(relation, sp.StrictGreaterThan):
            if value.real <= 0:
                return False
        elif isinstance(relation, sp.StrictLessThan):
            if value.real >= 0:
                return False
        elif isinstance(relation, sp.GreaterThan):
            if value.real < -tolerance:
                return False
        elif isinstance(relation, sp.LessThan):
            if value.real > tolerance:
                return False
        else:
            return False
    return True


def closed_scip_constraints(problem):
    """Close strict rational lattice rows by their exact integer spacing.

    Clearing denominators makes an integer polynomial residual. Its strict
    sign is equivalent to a closed bound of one unit; a continuous strict row
    has no such conversion and is refused.
    """
    constraints = []
    for relation in problem.constraints:
        if isinstance(relation, sp.Unequality):
            return None
        if not isinstance(relation, (sp.StrictGreaterThan, sp.StrictLessThan)):
            constraints.append(relation)
            continue
        polynomial = sp.Poly(relation.lhs - relation.rhs, *problem.variables)
        if any(c.is_Rational is not True for c in polynomial.coeffs()) or any(
            power and variable.is_integer is not True
            for powers, _ in polynomial.terms()
            for variable, power in zip(problem.variables, powers)
        ):
            return None
        denominator = sp.ilcm(1, *(c.q for c in polynomial.coeffs()))
        residual = denominator * polynomial.as_expr()
        constraints.append(
            sp.Ge(residual, 1, evaluate=False)
            if isinstance(relation, sp.StrictGreaterThan)
            else sp.Le(residual, -1, evaluate=False)
        )
    return constraints


def solve_scip(problem, controls):
    """Solve a fixed real closed quadratic model, preserving numerical proof scope.

    SCIP's dual bounds and optimal status use numerical tolerances. They are
    never promoted to exact certification here, even when a reported gap is zero.
    """
    constraints = closed_scip_constraints(problem)
    if constraints is None:
        return GlobalQuadraticResult(
            "unsupported",
            problem.variables,
            backend="scip",
            metadata={
                "reason": "SCIP requires closed rows or strict rational lattice rows"
            },
        )
    try:
        from pyscipopt import Model
    except (ImportError, OSError) as error:
        return GlobalQuadraticResult(
            "backend_unavailable",
            problem.variables,
            backend="scip",
            metadata={"reason": str(error)},
        )
    model = Model("symbopt quadratic")
    model.hideOutput()
    if controls.time_limit is not None:
        model.setRealParam("limits/time", controls.time_limit)
    if controls.max_nodes is not None:
        model.setLongintParam("limits/nodes", controls.max_nodes)
    model.setRealParam("limits/absgap", float(controls.absolute_gap))
    model.setRealParam("limits/gap", float(controls.relative_gap))
    variables = {
        v: model.addVar(
            name=f"x{i}", vtype="I" if v.is_integer is True else "C", lb=None, ub=None
        )
        for i, v in enumerate(problem.variables)
    }
    for i, variable in enumerate(problem.variables):
        if variable.is_even is True or variable.is_odd is True:
            quotient = model.addVar(name=f"parity{i}", vtype="I", lb=None, ub=None)
            model.addCons(
                variables[variable]
                == 2 * quotient + (1 if variable.is_odd is True else 0)
            )
    for relation in constraints:
        if not (relation.lhs - relation.rhs).free_symbols:
            if relation.func(*relation.args) is sp.false:
                return GlobalQuadraticResult(
                    "infeasible",
                    problem.variables,
                    backend="scip",
                    metadata={
                        "reason": "constant contradiction",
                        "numerical_bounds": True,
                    },
                )
            continue
        expression = _expression(
            relation.lhs - relation.rhs, problem.variables, variables
        )
        if isinstance(relation, sp.Equality):
            model.addCons(expression == 0)
        elif isinstance(relation, sp.GreaterThan):
            model.addCons(expression >= 0)
        elif isinstance(relation, sp.LessThan):
            model.addCons(expression <= 0)
    objective = model.addVar(name="objective", lb=None, ub=None)
    model.addCons(
        objective >= _expression(problem.objective, problem.variables, variables)
    )
    model.setObjective(objective, "minimize")
    model.optimize()
    raw = str(model.getStatus())
    statuses = {
        "optimal": "optimal",
        "infeasible": "infeasible",
        "unbounded": "unbounded",
        "inforunbd": "infeasible_or_unbounded",
        "timelimit": "time_limit",
        "nodelimit": "node_limit",
        "gaplimit": "gap_limit",
        "userinterrupt": "interrupted",
    }
    status = statuses.get(raw, raw)
    point = ()
    if model.getNSols():
        solution = model.getBestSol()
        candidate = tuple(
            sp.Float(model.getSolVal(solution, variables[v]), 17)
            for v in problem.variables
        )
        if numeric_feasible(problem, candidate):
            point = candidate
    value = problem.value(point) if point else None

    def bound(value):
        if model.isInfinity(abs(value)):
            return sp.oo if value > 0 else -sp.oo
        return sp.Float(value, 17)

    lower, upper = bound(model.getDualbound()), bound(model.getPrimalbound())
    if point:
        upper = max(upper, value)
    elif status not in {"infeasible", "unbounded", "infeasible_or_unbounded"}:
        upper = sp.oo
        if status == "optimal":
            status = "numerical_failure"
    metadata = {
        "solver_status": raw,
        "numerical_bounds": True,
        "requested_certification": controls.certified,
        "primal_exact": False,
        "numerical_gap": model.getGap(),
    }
    return GlobalQuadraticResult(
        status,
        problem.variables,
        value,
        point,
        lower,
        upper,
        "scip",
        model.getNNodes(),
        False,
        False,
        True if status == "optimal" and point else None,
        None,
        metadata,
    )
