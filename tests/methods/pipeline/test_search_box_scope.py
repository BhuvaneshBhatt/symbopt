"""A search hint cannot restrict the mathematical global optimization domain."""

import sympy as sp

from symbopt import minimize_result


def test_local_box_lower_bound():
    x, y = sp.symbols("x y", real=True)
    objective = (x - sp.Float(2)) ** 4 + y**4 + x * y
    result = minimize_result(
        objective,
        variables=(x, y),
        box_bounds={x: (0, 1), y: (0, 1)},
        max_active_sets=0,
        interval_max_boxes=1,
        use_sos=False,
        semialg_certification="candidate",
        options={"structural": {"enabled": False}},
    )
    # This feasible point outside the hint box disproves its positive global bound.
    assert objective.subs({x: 2, y: 0}) == 0
    assert result.interval_result.best_lower_bound > 0
    assert not result.interval_result.lower_bound_rigorous
    assert result.certificate.lower_bound is None and not result.certified
