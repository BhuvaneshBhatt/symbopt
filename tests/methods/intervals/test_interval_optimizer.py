import math

import sympy as sp

from symbopt import infer_box_bounds, interval_optimize


def test_infer_box_bounds():
    x = sp.symbols("x", real=True)
    b = infer_box_bounds([x], [sp.Ge(x, -2), sp.Le(x, 3)])
    assert b[x] == (-2.0, 3.0)


def test_interval_quadratic():
    x = sp.symbols("x", real=True)
    r = interval_optimize(
        (x - 1) ** 2, [sp.Ge(x, 0), sp.Le(x, 2)], {x: (0, 2)}, max_boxes=80, tol=1e-2
    )
    assert r.status == "success"
    assert r.best_upper_bound < 1e-2


def test_interval_sine_bounded():
    x = sp.symbols("x", real=True)
    r = interval_optimize(
        sp.sin(x),
        [sp.Ge(x, 0), sp.Le(x, 2 * sp.pi)],
        {x: (0, 2 * math.pi)},
        max_boxes=100,
        tol=5e-2,
    )
    assert r.status == "success"
    assert r.best_upper_bound <= -0.9


def test_interval_infeasible_box():
    x = sp.symbols("x", real=True)
    r = interval_optimize(
        x**2, [sp.Ge(x, 2), sp.Le(x, 1)], {x: (0, 3)}, max_boxes=30, tol=0.1
    )
    assert r.status in {"no_feasible_point_found", "success"}
    if r.status == "success":
        assert not r.best_point


def test_interval_tangent_singularity_does_not_crash():
    x = sp.symbols("x", real=True)
    r = interval_optimize(
        sp.tan(x), [sp.Ge(x, 0), sp.Le(x, 1.4)], {x: (0, 1.4)}, max_boxes=20, tol=0.05
    )
    assert r.status in {"success", "no_feasible_point_found"}


def test_infer_box_bounds_ignores_one_sided_only():
    x = sp.symbols("x", real=True)
    b = infer_box_bounds([x], [sp.Ge(x, 0)])
    assert x not in b


def test_interval_rejects_reversed_box():
    import pytest
    import sympy as sp

    from symbopt import interval_optimize

    x = sp.symbols("x", real=True)
    with pytest.raises(ValueError):
        interval_optimize(x**2, [], {x: (1.0, -1.0)})


def test_interval_rejects_missing_variable_bound():
    import pytest
    import sympy as sp

    from symbopt import interval_optimize

    x, y = sp.symbols("x y", real=True)
    with pytest.raises(ValueError):
        interval_optimize(x + y, [], {x: (0.0, 1.0)})


def test_interval_rejects_invalid_search_controls():
    import pytest
    import sympy as sp

    from symbopt import interval_optimize

    x = sp.symbols("x", real=True)
    with pytest.raises(ValueError):
        interval_optimize(x, [], {x: (0.0, 1.0)}, max_boxes=0)
    with pytest.raises(ValueError):
        interval_optimize(x, [], {x: (0.0, 1.0)}, tol=0)


def test_interval_optimize_accepts_single_relation():
    x = sp.symbols("x", real=True)
    result = interval_optimize(x**2, sp.Ge(x, 0), {x: (-1.0, 1.0)}, max_boxes=8)
    assert result.status in {"success", "certified", "bounded", "max_boxes_reached"}
    assert result.best_lower_bound <= 0.0 <= result.best_upper_bound
