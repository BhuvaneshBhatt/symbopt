from __future__ import annotations

import sympy as sp

from symbopt import plan_problem, semialg_available


def test_planner_small_sparse_polynomial_enables_exact_routes():
    x, y = sp.symbols("x y", real=True)
    plan = plan_problem(x**4 + y**4, [x + y >= 1], [x, y])
    assert plan.try_exact_kkt
    assert plan.try_numeric_kkt
    assert plan.try_sos
    assert plan.complexity.semialg_available is semialg_available()
    assert plan.try_semialg_early is semialg_available()
    assert "within" in plan.reasons["exact_kkt"]


def test_planner_backend_availability():
    x = sp.symbols("x", real=True)
    plan = plan_problem(x**4 - x + 1, [], [x])
    assert plan.complexity.semialg_available is semialg_available()
    if semialg_available():
        assert plan.try_semialg_early
        assert plan.reasons["semialg"] == "small exact semialgebraic problem"
    else:
        assert plan.try_semialg_early is False
        assert "unavailable" in plan.reasons["semialg"]


def test_planner_defers_active_set_explosion():
    variables = sp.symbols("x0:6", real=True)
    constraints = [variable >= 0 for variable in variables]
    plan = plan_problem(
        sum(variable**2 for variable in variables), constraints, variables
    )
    assert plan.complexity.estimated_active_sets == 64
    assert plan.try_exact_kkt

    constraints = [*constraints, *(variable <= 1 for variable in variables)]
    plan = plan_problem(
        sum(variable**2 for variable in variables), constraints, variables
    )
    assert plan.complexity.estimated_active_sets == 4096
    assert plan.try_exact_kkt is False
    assert plan.try_numeric_kkt is True

    plan = plan_problem(
        sum(variable**2 for variable in variables),
        [*constraints, sum(variables) <= 3],
        variables,
    )
    assert plan.complexity.estimated_active_sets == 8192
    assert plan.try_numeric_kkt is False


def test_planner_box_coverage():
    x, y = sp.symbols("x y", real=True)
    objective = sp.exp(x) + y**2
    partial = plan_problem(objective, [], [x, y], {x: (-1, 1)})
    finite = plan_problem(objective, [], [x, y], {x: (-1, 1), y: (-2, 2)})
    assert partial.complexity.finite_box is False
    assert partial.try_interval is False
    assert finite.complexity.finite_box is True
    assert finite.try_interval is True


def test_planner_density_changes_sos_budget_not_mathematics():
    xs = sp.symbols("x0:6", real=True)
    sparse = sum(x**8 for x in xs)
    dense = sp.expand(sum(xs) ** 8)
    sparse_plan = plan_problem(sparse, [], xs)
    dense_plan = plan_problem(dense, [], xs)
    assert sparse_plan.complexity.density < dense_plan.complexity.density
    assert (
        sparse_plan.complexity.estimated_sos_gram_dimension
        == dense_plan.complexity.estimated_sos_gram_dimension
    )
    assert sparse_plan.reasons["sos"] in {
        "estimated Gram/SDP size is within automatic budget",
        "SOS Gram estimate is unavailable or too large",
    }


def test_planner_explain_is_stable_and_user_facing():
    x = sp.symbols("x", real=True)
    plan = plan_problem((x - 1) ** 2, [], [x])
    explanation = plan.explain()
    assert explanation.startswith("auto planner:")
    assert "exact-kkt" in explanation
