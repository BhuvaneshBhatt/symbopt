"""Mathematical checks and process isolation for the reference corpus."""

import json
import multiprocessing
import traceback
import unittest
from functools import partial
from pathlib import Path

import sympy as sp

import symbopt


class ReferenceFailure(AssertionError):
    """A reproduced reference mismatch or bounded solver limitation."""


def load_cases():
    """Load mathematical examples stored directly as SymPy expressions."""
    return json.loads(Path(__file__).with_name("cases.json").read_text())


def decode_case(case):
    """Reconstruct the problem, retaining discrete assumptions on decision symbols."""
    objective = sp.sympify(case["objective"])
    constructors = {
        name: partial(constructor, evaluate=False)
        for name, constructor in (
            ("Equality", sp.Eq),
            ("Unequality", sp.Ne),
            ("StrictLessThan", sp.Lt),
            ("LessThan", sp.Le),
            ("StrictGreaterThan", sp.Gt),
            ("GreaterThan", sp.Ge),
        )
    }
    constraints = [
        sp.sympify(value, locals=constructors) for value in case["constraints"]
    ]
    variables = tuple(sp.sympify(value) for value in case["variables"])
    replacements = {
        variable: sp.Symbol(variable.name, integer=True)
        for variable in variables
        if variable.name in case["integer_variables"]
    }
    return (
        objective.xreplace(replacements),
        [
            value.func(
                *(arg.xreplace(replacements) for arg in value.args), evaluate=False
            )
            if isinstance(value, sp.core.relational.Relational)
            else value.xreplace(replacements)
            for value in constraints
        ],
        tuple(replacements.get(variable, variable) for variable in variables),
    )


def same_value(actual, expected, numeric=False):
    """Compare mathematical values, using tolerance only for numerical references."""
    if actual is None:
        return False
    actual, expected = sp.sympify(actual), sp.sympify(expected)
    if actual == expected:
        return True
    if actual in (sp.oo, -sp.oo) or expected in (sp.oo, -sp.oo):
        return False
    if numeric:
        actual, expected = complex(sp.N(actual, 40)), complex(sp.N(expected, 40))
        return abs(actual - expected) <= 1e-9 * (1 + abs(expected))
    difference = sp.simplify(actual - expected)
    if difference == 0:
        return True
    if difference.has(sp.Max, sp.Min, sp.Piecewise):
        return sp.simplify(difference.rewrite(sp.Piecewise)) == 0
    return False


def feasible(condition, point, numeric):
    """Check exact feasibility or numerical residuals, preserving strict bounds."""
    value = sp.simplify(condition.subs(point, simultaneous=True))
    if value is sp.true:
        return True
    if isinstance(condition, sp.And):
        return all(feasible(a, point, numeric) for a in condition.args)
    if isinstance(condition, sp.Or):
        return any(feasible(a, point, numeric) for a in condition.args)
    if not numeric or not hasattr(condition, "lhs"):
        return False
    residual = complex(sp.N((condition.lhs - condition.rhs).subs(point), 30))
    if abs(residual.imag) > 1e-10:
        return False
    residual = residual.real
    if isinstance(condition, sp.Equality):
        return abs(residual) <= 1e-8
    if isinstance(condition, sp.StrictGreaterThan):
        return residual > 0
    if isinstance(condition, sp.StrictLessThan):
        return residual < 0
    if isinstance(condition, sp.GreaterThan):
        return residual >= -1e-8
    if isinstance(condition, sp.LessThan):
        return residual <= 1e-8
    if isinstance(condition, sp.Unequality):
        return abs(residual) > 1e-10
    return False


def check_case(case):
    """Run the requested API and verify its value, attainment and real feasibility."""
    objective, constraints, variables = decode_case(case)
    expected = [sp.sympify(value) for value in case["expected_values"]]
    numeric = case["numeric"]
    kwargs = {
        "variables": variables,
        "use_sos": False,
        "interval_max_boxes": 12,
        "interval_tol": 1e-7,
        "max_active_sets": 16,
    }
    if case["working_precision"] is not None:
        kwargs["working_precision"] = case["working_precision"]
    if case["kind"] == "region_definition":
        for point, member in case["membership_points"]:
            point = {sp.Symbol(k, real=True): sp.sympify(v) for k, v in point.items()}
            assert all(feasible(r, point, False) for r in constraints) is member
        return
    if case["kind"] == "coordinate_bounds":
        for variable, pair in zip(variables, case["coordinate_values"]):
            for operation, value in zip((symbopt.minval, symbopt.maxval), pair):
                actual = operation(variable, constraints, **kwargs)
                assert same_value(actual, sp.sympify(value)), (variable, actual, value)
        return
    if case["operation"] == "minimize_local":
        for initial, target in zip(case["initial_points"], expected):
            initial = {
                sp.Symbol(k, real=True): sp.sympify(v) for k, v in initial.items()
            }
            result = symbopt.minimize_local(objective, variables, initial)
            assert result.success
            assert same_value(result.objective_value, target, True)
            assert all(
                abs(float(sp.N(sp.diff(objective, v).subs(result.point)))) < 1e-6
                for v in variables
            )
        return
    operation = getattr(symbopt, case["operation"])
    if case["kind"] == "unresolved":
        try:
            result = operation(objective, constraints, **kwargs)
        except (NotImplementedError, TypeError, ValueError):
            return
        assert not getattr(result, "certified", False)
        return
    result = operation(objective, constraints, **kwargs)
    if case["operation"] in {"minval", "maxval"}:
        assert any(same_value(result, value, numeric) for value in expected), (
            result,
            expected,
        )
        return
    if case["kind"] == "empty_arguments":
        assert result == ()
        detailed = symbopt.minimize_result(objective, constraints, **kwargs)
        assert detailed.certified
        assert (
            detailed.infeasible
            if case["expected_status"] == "infeasible"
            else detailed.unbounded
        )
        return
    if case["operation"] in {"argmin", "argmax"}:
        points = result
        assert any(
            all(feasible(r, p, numeric) for r in constraints)
            and any(same_value(objective.subs(p), value, numeric) for value in expected)
            for p in points
        ), (points, expected)
        return
    assert result.optimum_value is not None, result.explain()
    assert any(
        same_value(result.optimum_value, value, numeric) for value in expected
    ), (result.optimum_value, expected)
    if not numeric:
        assert result.certified, result.explain()
    if case["expected_status"] == "infeasible":
        assert result.infeasible and not result.unbounded
    if case["attained"] is not None:
        assert result.attained is case["attained"], (result.attained, case["attained"])
    if result.attained is True:
        candidate = result.best_candidate
        assert candidate is not None, result.explain()
        point = dict(candidate.original_variable_values or candidate.primal_values)
        active = objective.free_symbols.union(*(r.free_symbols for r in constraints))
        point.update((v, sp.S.Zero) for v in variables if v not in active)
        assert all(point.get(v, sp.nan).is_real is True for v in variables), point
        assert all(feasible(r, point, numeric) for r in constraints), point
        assert same_value(
            objective.subs(point, simultaneous=True), result.optimum_value, numeric
        )
        for definition in case["root_definitions"]:
            seed = sp.sympify(definition["seed"])
            equation = sp.sympify(definition["equation"])
            root_variable = sp.sympify(definition["variable"])
            for variable in variables:
                expected_coordinates = [
                    p.get(variable.name) for p in case["expected_points"]
                ]
                if not any(
                    v is not None and sp.simplify(sp.sympify(v) - seed) == 0
                    for v in expected_coordinates
                ):
                    continue
                terms = [
                    complex(sp.N(term.subs(root_variable, point[variable]), 40))
                    for term in sp.Add.make_args(equation)
                ]
                assert abs(sum(terms)) <= 1e-6 * (1 + sum(abs(term) for term in terms))
    if case["kind"] == "branch_domain":
        assert result.best_candidate.primal_values == {variables[0]: 0, variables[1]: 0}


class _Outcome(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.outcome = {"kind": "pass", "detail": ""}

    def addError(self, test, err):
        self.outcome = {
            "kind": err[0].__name__,
            "detail": "".join(traceback.format_exception(*err)),
        }

    def addFailure(self, test, err):
        self.addError(test, err)


def _worker(case, connection):
    # unittest preserves unexpected exception types across the process boundary.
    outcome = _Outcome()
    unittest.FunctionTestCase(lambda: check_case(case)).run(outcome)
    connection.send(outcome.outcome)
    connection.close()


def execute_case(case):
    """Bound a solver call in a process, including code that cannot handle signals."""
    methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in methods else "spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(case, child))
    process.start()
    child.close()
    try:
        if parent.poll(case["timeout"]):
            try:
                outcome = parent.recv()
            except EOFError:
                outcome = {
                    "kind": "WorkerExit",
                    "detail": f"Worker exited with code {process.exitcode}",
                }
        else:
            process.terminate()
            outcome = {
                "kind": "Timeout",
                "detail": f"Reference budget of {case['timeout']} seconds exceeded",
            }
    finally:
        parent.close()
        process.join(1)
        if process.is_alive():
            process.kill()
            process.join()
    return outcome
