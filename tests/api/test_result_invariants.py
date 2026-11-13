from __future__ import annotations

import sympy as sp
from hypothesis import given, settings
from hypothesis import strategies as st

from symbopt import maximize_result, minimize_result


def _represented_optimizers(result):
    if result.best_candidate is not None:
        return [result.best_candidate.primal_values]
    return []


def test_certified_attained_result_evaluates_to_optimum():
    x, y = sp.symbols("x y", real=True)
    result = minimize_result((x - 2) ** 2 + (y + 1) ** 2, variables=[x, y])
    assert result.certified
    assert result.attained is True
    assert result.infeasible is False
    assert result.unbounded is False
    for optimizer in _represented_optimizers(result):
        assert (
            sp.simplify(
                result.lifted_problem.original_objective.subs(optimizer)
                - result.optimum_value
            )
            == 0
        )


def test_infeasible_and_unbounded_are_mutually_exclusive():
    x = sp.symbols("x", real=True)
    infeasible = minimize_result(x, [x >= 1, x <= 0], variables=x)
    unbounded = minimize_result(x, variables=x)
    assert infeasible.infeasible and not infeasible.unbounded
    assert unbounded.unbounded and not unbounded.infeasible


# Exact solver properties allow cold backend imports; benchmarks track timing.
@settings(deadline=None)
@given(st.integers(-5, 5))
def test_certified_quadratic_result_invariant(shift):
    x = sp.symbols("x", real=True)
    result = minimize_result((x - shift) ** 2 + 3, variables=x)
    assert result.certified
    assert result.attained is True
    assert result.optimum_value == 3
    assert sp.simplify(result.best_candidate.primal_values[x] - shift) == 0
    assert "certified" in result.explain()


@settings(deadline=None)
@given(st.integers(-4, 4))
def test_maximum_result_invariant(shift):
    x = sp.symbols("x", real=True)
    result = maximize_result(-((x - shift) ** 2) + 7, variables=x)
    assert result.certified
    assert result.attained is True
    assert result.optimum_value == 7
    assert sp.simplify(result.best_candidate.primal_values[x] - shift) == 0
