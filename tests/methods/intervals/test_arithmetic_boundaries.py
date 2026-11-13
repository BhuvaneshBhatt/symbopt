"""Exact arithmetic checks at floating-point proof and domain boundaries."""

import math
from fractions import Fraction

import pytest
import sympy as sp

from symbopt import interval_optimize
from symbopt.intervals import (
    BoxState,
    IntervalEvaluator,
    _idiv_point_by_interval,
    _split_choice,
)
from symbopt.precision import finite_midpoint
from symbopt.problem_models import IntervalBox

x, y = sp.symbols("x y", real=True)


@pytest.mark.parametrize(
    "numerator,denominator",
    [
        (4.1579655603036046e278, 5.949277003448074e302),
        (1.0, 3.0),
        (1e308, 1e308),
        (0.0, math.ulp(0.0)),
        (1e-300, 1e-10),
        (1.0, math.ulp(0.0)),
    ],
)
@pytest.mark.parametrize("signs", [(1, 1), (1, -1), (-1, 1), (-1, -1)])
def test_division_encloses_exact(numerator, denominator, signs):
    numerator *= signs[0]
    denominator *= signs[1]
    lo, hi = _idiv_point_by_interval((numerator, numerator), (denominator, denominator))
    exact = Fraction(numerator) / Fraction(denominator)
    assert lo == -math.inf or Fraction(lo) <= exact
    assert hi == math.inf or exact <= Fraction(hi)


def test_division_interval_endpoints():
    numerator, denominator = (-2.0, 3.0), (-7.0, -0.5)
    lo, hi = _idiv_point_by_interval(numerator, denominator)
    exact = [Fraction(n) / Fraction(d) for n in numerator for d in denominator]
    assert Fraction(lo) <= min(exact) <= max(exact) <= Fraction(hi)
    assert _idiv_point_by_interval(numerator, (-1.0, 1.0)) is None
    assert _idiv_point_by_interval(numerator, (1.0, math.inf)) is None


@pytest.mark.parametrize(
    "lo,hi",
    [
        (1e308, 1.5e308),
        (-1.5e308, -1e308),
        (-1e308, 1e308),
        (math.ulp(0.0), math.ulp(0.0)),
        (0.0, math.ulp(0.0)),
    ],
)
def test_finite_midpoint(lo, hi):
    midpoint = finite_midpoint(lo, hi)
    assert math.isfinite(midpoint) and lo <= midpoint <= hi
    assert IntervalBox({x: (lo, hi)}).midpoint_subs() == {x: midpoint}


def test_large_box_split():
    box = IntervalBox({x: (1e308, 1.5e308)})
    left, right = box.split()
    assert left.bounds[x][0] == box.bounds[x][0]
    assert right.bounds[x][1] == box.bounds[x][1]
    assert box.bounds[x][0] < left.bounds[x][1] == right.bounds[x][0] < box.bounds[x][1]


def test_split_ignores_fixed_coordinate():
    box = IntervalBox({x: (0.0, 0.0), y: (-1.0, 1.0)})
    objective = IntervalEvaluator.compile(y * y)
    state = BoxState(box, objective.evaluate(box), ())
    variable, split = _split_choice(state, [y * y], {}, {}, [])
    assert variable == y and split == 0


def test_adjacent_float_leaf():
    lo = 1e100
    hi = math.nextafter(lo, math.inf)
    coordinate = (x - sp.Rational(lo)) / sp.Rational(hi - lo) - sp.Rational(1, 2)
    result = interval_optimize(coordinate**2, [], {x: (lo, hi)}, max_boxes=8, tol=1e-10)
    assert result.boxes_processed == 1 and result.remaining_boxes == 0
    assert result.best_lower_bound <= 0 <= result.best_upper_bound
    assert result.best_upper_bound == 0.25


@pytest.mark.parametrize(
    "value", [sp.Rational(1, 3), -sp.Rational(2, 3), sp.Rational(1, 10)]
)
def test_point_upper_rounding(value):
    result = interval_optimize(value, [], {x: (0, 1)}, max_boxes=1)
    assert result.lower_bound_rigorous and result.upper_bound_rigorous
    assert (
        sp.Rational(result.best_lower_bound)
        <= value
        <= sp.Rational(result.best_upper_bound)
    )


def test_zero_coordinate_box():
    result = interval_optimize(1, [], {}, max_boxes=1)
    assert result.best_lower_bound == result.best_upper_bound == 1
    assert result.upper_bound_rigorous and result.best_point == {}


def test_correction_stays_in_source_box():
    result = interval_optimize(
        y, [sp.Eq(y, x * x)], {x: (-2, 2), y: (1, 4)}, max_boxes=8
    )
    point = {v: sp.Rational(value) for v, value in result.best_point.items()}
    assert -2 <= point[x] <= 2 and 1 <= point[y] <= 4
    assert point[y] == point[x] ** 2
    assert result.best_lower_bound <= 1 <= result.best_upper_bound


def test_exact_box_boundary():
    point = sp.Rational(1, 3)
    result = interval_optimize(x, [], {x: (point, point)}, max_boxes=1)
    assert result.best_lower_bound <= point
    assert result.best_point == {} and result.best_upper_bound == math.inf


def test_rational_outward_conversion():
    from random import Random

    from symbopt.precision import outward_float

    rng = Random(73)
    values = [Fraction(1, 2**1075), Fraction(-1, 2**1075), Fraction(2**1023, 3)]
    values += [
        Fraction(rng.getrandbits(1100), 1 + rng.getrandbits(1000)) for _ in range(64)
    ]
    for value in values:
        exact = sp.Rational(value.numerator, value.denominator)
        assert (
            Fraction(outward_float(exact, -1))
            <= value
            <= Fraction(outward_float(exact, 1))
        )
