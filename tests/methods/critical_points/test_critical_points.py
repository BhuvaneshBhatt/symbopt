import sympy as sp

from symbopt import critical_points
from symbopt.critical_points import check_second_order


def test_unconstrained_minimum():
    x = sp.symbols("x", real=True)
    r = critical_points((x - 2) ** 2, [x])
    assert any(
        cp.solution.get(x) == 2 and cp.classification == "LocalMinimum" for cp in r
    )


def test_unconstrained_saddle():
    x, y = sp.symbols("x y", real=True)
    r = critical_points(x**2 - y**2, [x, y])
    assert any(cp.classification == "SaddlePoint" for cp in r)


def test_degenerate_quartic_classified():
    x = sp.symbols("x", real=True)
    r = critical_points(x**4, [x])
    assert any(
        cp.classification
        in {
            "FlatMinimumCandidate",
            "Degenerate",
            "LocalMinimum",
            "FlatDirectionDetected",
        }
        for cp in r
    )


def test_second_order_condition_true_for_bowl():
    x = sp.symbols("x", real=True)
    assert check_second_order((x**2, []), [x], {x: 0}, target="Min")
