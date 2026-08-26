import math

import pytest
import sympy as sp

from symbopt import (
    EvalConfig,
    arb_available,
    enclose_expression,
    interval_optimize,
)
from symbopt.verified_intervals import compile_arb


def test_polynomial_bound_is_rigorous():
    x = sp.symbols("x", real=True)
    bound = enclose_expression(
        x**2,
        (x,),
        ((-1.0, 1.0),),
        EvalConfig(),
    )
    assert bound.rigorous
    assert bound.lower <= 0 <= bound.upper
    assert bound.upper >= 1


def test_dependency_cancellation_exact():
    x = sp.symbols("x", real=True)
    bound = enclose_expression(
        x - x,
        (x,),
        ((-3.0, 5.0),),
    )
    assert bound.rigorous
    assert bound.lower == 0
    assert bound.upper == 0


def test_branch_bound_quadratic_certified():
    x = sp.symbols("x", real=True)
    result = interval_optimize(
        x**2,
        [sp.Ge(x, -1), sp.Le(x, 1)],
        {x: (-1.0, 1.0)},
        max_boxes=100,
        tol=1e-2,
    )
    assert result.lower_bound_rigorous
    assert result.upper_bound_rigorous
    assert abs(result.best_lower_bound) <= 1e-12
    assert abs(result.best_upper_bound) <= 1e-12


def test_reciprocal_crossing_zero_is_uncertain():
    x = sp.symbols("x", real=True)
    bound = enclose_expression(1 / x, (x,), ((-1.0, 1.0),))
    assert bound.rigorous
    assert bound.domain == "uncertain"
    assert math.isinf(bound.lower)
    assert math.isinf(bound.upper)


def test_log_negative_box_is_invalid():
    x = sp.symbols("x", real=True)
    bound = enclose_expression(sp.log(x), (x,), ((-2.0, -1.0),))
    assert bound.rigorous
    assert bound.domain == "invalid"


def test_compiled_arb_graph_structure():
    x = sp.symbols("x", real=True)
    node = compile_arb(sp.sin(x) + x**2)
    assert node.op == "add"
    assert len(node.children) == 2


def test_transcendental_without_arb_not_certifying():
    if arb_available():
        pytest.skip("Arb backend is installed")
    x = sp.symbols("x", real=True)
    result = interval_optimize(
        sp.sin(x),
        [sp.Ge(x, 0), sp.Le(x, 2 * sp.pi)],
        {x: (0.0, 2 * math.pi)},
        max_boxes=30,
        tol=0.05,
    )
    assert not result.lower_bound_rigorous
