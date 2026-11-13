"""Exact source endpoints must survive conversion to binary search boxes."""

import pytest
import sympy as sp

from symbopt import infer_box_bounds, interval_optimize
from symbopt.input_normalization import normalize_bounds_input
from symbopt.problem_models import IntervalBox

x = sp.Symbol("x", real=True)


@pytest.mark.parametrize(
    "endpoint",
    [sp.Rational(1, 10), -sp.Rational(1, 10), sp.sqrt(2), sp.pi, sp.Integer(2**60 + 1)],
)
def test_exact_endpoint_enclosure(endpoint):
    box = IntervalBox({x: (endpoint, endpoint)})
    lower, upper = box.bounds[x]
    assert (endpoint - sp.Rational(lower)).is_nonnegative is True
    assert (sp.Rational(upper) - endpoint).is_nonnegative is True
    inferred = infer_box_bounds(x, [sp.Eq(x, endpoint)])
    assert inferred == box.bounds
    _, normalized = normalize_bounds_input({x: (endpoint, endpoint)}, [x])
    assert normalized == box.bounds


def test_lower_bound_cancellation():
    lower = sp.Rational(1, 10)
    result = interval_optimize(x - lower, (), {x: (lower, 1)}, max_boxes=1)
    assert result.lower_bound_rigorous
    assert sp.Rational(result.best_lower_bound) <= 0
    assert result.best_upper_bound >= 0


def test_binary_endpoints_unchanged():
    bounds = {x: (0.1, 0.9)}
    assert IntervalBox(bounds).bounds == bounds
