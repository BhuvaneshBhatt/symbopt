"""Tolerance-screened proposals cannot establish exact primal proof claims."""

from dataclasses import replace

import sympy as sp

from symbopt import algebraize, minimize_result
from symbopt.problem_models import CandidateSolution
from symbopt.solver_pipeline import (
    _certify_search_result,
    _exact_source_candidate,
    _merge_piecewise_results,
    _optimizer_points,
)

x = sp.Symbol("x", real=True)


def proposal(point, value):
    return CandidateSolution(
        {x: point},
        {},
        value,
        float(value),
        True,
        "proposal",
        original_variable_values={x: point},
    )


def test_infeasible_zero_candidate():
    delta = sp.Rational(1, 10**12)
    problem = algebraize(x**4, [x >= delta], variables=(x,))
    candidate = proposal(sp.S.Zero, sp.S.Zero)
    assert not _exact_source_candidate(problem, candidate)
    # x**4 >= 0 is a valid global lower bound, but zero is infeasible here.
    certified, gap, *_ = _certify_search_result(
        problem, candidate, None, sp.S.Zero, True, (x,), [candidate], []
    )
    assert certified is False and gap is None


def test_unattained_branch_proposal():
    first = minimize_result(x, x > 0, variables=x)
    second = minimize_result(x, x > 1, variables=x)
    candidate = proposal(sp.S.Zero, sp.S.Zero)
    changed = replace(first, candidates=[candidate], best_candidate=candidate)
    merged = _merge_piecewise_results([changed, second])
    assert merged.certified and merged.optimum_value == 0
    assert merged.attained is False
    assert _optimizer_points(merged) == ()
    assert merged.certificate.primal_value is None


def test_branch_optimizer_source():
    objective = (x * x - 1) ** 2
    first = minimize_result(objective, x > sp.Rational(1, 2), variables=x)
    second = minimize_result(objective, x < -2, variables=x)
    candidate = proposal(-sp.S.One, sp.S.Zero)
    changed = replace(
        first, candidates=[candidate, *first.candidates], best_candidate=candidate
    )
    merged = _merge_piecewise_results([changed, second])
    assert merged.certified and merged.optimum_value == 0
    assert _optimizer_points(merged) == ({x: sp.S.One},)


def test_optimizer_source():
    result = minimize_result((x * x - 1) ** 2, x > sp.Rational(1, 2), variables=x)
    changed = replace(
        result, candidates=[proposal(-sp.S.One, sp.S.Zero), *result.candidates]
    )
    assert _optimizer_points(changed) == ({x: sp.S.One},)
