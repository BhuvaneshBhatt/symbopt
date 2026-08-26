import math

import sympy as sp

from symbopt import interval_optimize
from symbopt.intervals import contract_equalities
from symbopt.problem_models import IntervalBox


def test_interval_newton_contracts_scalar_root_without_losing_it():
    x = sp.symbols("x", real=True)
    box = IntervalBox({x: (1.0, 2.0)})
    result = contract_equalities(box, [x**2 - 2], [x])
    lo, hi = result.box.bounds[x]
    assert result.changed
    assert not result.infeasible
    assert "interval-newton" in result.method
    assert lo <= math.sqrt(2) <= hi
    assert hi - lo < 0.1


def test_interval_newton_can_prove_equality_box_infeasible():
    x = sp.symbols("x", real=True)
    result = contract_equalities(IntervalBox({x: (0.0, 1.0)}), [x - 2], [x])
    assert result.infeasible
    assert result.method == "interval-newton"


def test_krawczyk_contracts_coupled_square_system():
    x, y = sp.symbols("x y", real=True)
    result = contract_equalities(
        IntervalBox({x: (0.0, 4.0), y: (0.0, 4.0)}),
        [x + y - 3, x - y - 1],
        [x, y],
    )
    assert result.changed
    assert not result.infeasible
    assert "krawczyk" in result.method
    xlo, xhi = result.box.bounds[x]
    ylo, yhi = result.box.bounds[y]
    assert xlo <= 2 <= xhi
    assert ylo <= 1 <= yhi
    assert xhi - xlo < 1e-8
    assert yhi - ylo < 1e-8


def test_interval_optimizer_uses_equality_contractor():
    x = sp.symbols("x", real=True)
    result = interval_optimize(
        (x - 1) ** 2,
        [sp.Eq(x**2 - 1, 0), sp.Ge(x, 0)],
        {x: (0.0, 2.0)},
        max_boxes=20,
        tol=1e-5,
    )
    assert result.status == "success"
    assert result.contractions_applied >= 1
    assert "interval-newton" in result.contraction_method
    assert result.best_upper_bound < 1e-10
