"""Numerical proposals must remain finite and real before normalization."""

import pytest
import sympy as sp

from symbopt import algebraize, build_kkt_system, solve_convex_qp
from symbopt.candidates import _filter_candidate
from symbopt.numeric_utils import eval_float, is_numerically_real
from symbopt.relation_checks import relation_holds

x = sp.Symbol("x", real=True)


@pytest.mark.parametrize("value", [sp.oo, -sp.oo, sp.zoo, sp.nan, sp.I])
def test_numeric_conversion_rejects_invalid(value):
    with pytest.raises((ValueError, TypeError)):
        eval_float(value)
    assert is_numerically_real(value) is False


def test_relation_residuals_must_be_real():
    assert not relation_holds(sp.Eq(sp.I * x, 0, evaluate=False), {x: 1})
    assert not relation_holds(sp.Ge(x, 0, evaluate=False), {x: sp.oo})
    assert relation_holds(sp.Ge(x, 0, evaluate=False), {x: sp.Rational(1, 3)})


@pytest.mark.parametrize("value", [sp.Float(1) + sp.I * sp.Float(2), sp.oo, sp.nan])
def test_raw_candidate_rejection(value):
    problem = algebraize((x - 2) ** 2, [], variables=(x,))
    kkt = build_kkt_system(problem)
    assert _filter_candidate(problem, kkt, {x: value}, "numerical", 1e-7) is None


def test_real_candidate_is_preserved():
    problem = algebraize((x - 2) ** 2, [], variables=(x,))
    candidate = _filter_candidate(
        problem, build_kkt_system(problem), {x: sp.Float(2)}, "numerical", 1e-7
    )
    assert candidate.primal_values == {x: 2} and candidate.objective_value == 0


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf")])
def test_qp_control_validation(tolerance):
    with pytest.raises(ValueError):
        solve_convex_qp([[-1]], [0], convexity_tol=tolerance)
