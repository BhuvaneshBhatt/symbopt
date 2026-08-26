import sympy as sp

from symbopt import minimize
from symbopt.candidates import build_active_set_plan


def test_active_pattern_budget_includes_boundaries():
    x = sp.symbols("x", real=True)
    inequalities = [x - index for index in range(6)]
    plan = build_active_set_plan(
        inequalities,
        [x],
        max_active_sets=5,
        box_bounds=None,
    )
    assert plan.patterns[0] == (0, 0, 0, 0, 0, 0)
    assert 1 <= len(plan.patterns) <= 5
    assert all(len(pattern) == 6 for pattern in plan.patterns)
    assert len(plan.patterns) == 5


def test_active_pattern_prunes_strictly_inactive_constraint():
    x = sp.symbols("x", real=True)
    plan = build_active_set_plan(
        [x - 1, 3 - x],
        [x],
        max_active_sets=8,
        box_bounds={x: (2.0, 2.5)},
    )
    assert plan.patterns
    assert all(pattern[0] == 0 for pattern in plan.patterns)


def test_active_budget_keeps_boundary_solution():
    x = sp.symbols("x", real=True)
    constraints = [
        sp.Ge(x, 1),
        sp.Le(x, 2),
        sp.Le(x, 3),
        sp.Le(x, 4),
        sp.Le(x, 5),
        sp.Le(x, 6),
        sp.Le(x, 7),
    ]
    result = minimize(
        x**4,
        constraints,
        max_active_sets=3,
        use_sos=False,
        interval_max_boxes=20,
    )
    assert result.best_candidate is not None
    assert result.best_candidate.primal_values[x] == 1
