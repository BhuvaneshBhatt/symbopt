import pytest
import sympy as sp

from symbopt import (
    function_range_exact,
    minimize,
    prove_nonnegative_exact,
    semialg_available,
)
from symbopt.semialgebraic import (
    exact_feasible,
    prune_redundant,
    solve_exact_points,
)

pytestmark = pytest.mark.integration


def test_live_semialg_backend_when_installed():
    pytest.importorskip("semialg")
    x = sp.symbols("x", real=True)
    assert exact_feasible([sp.Lt(x**2, 0)], [x]) is False

    reduced = prune_redundant([sp.Ge(x, 1), sp.Ge(x, 0)], [x])
    assert reduced == [sp.Ge(x, 1)]

    points = solve_exact_points([x**2 - 2], [x])
    assert {sp.simplify(point[x]) for point in points} == {sp.sqrt(2), -sp.sqrt(2)}

    result = minimize(
        x**4,
        [sp.Ge(x, 1)],
        max_active_sets=0,
        use_sos=False,
    )
    assert result.certified
    assert result.best_candidate is not None
    assert result.best_candidate.objective_value == 1
    assert result.best_candidate.original_variable_values[x] == 1
    assert "semialg" in result.status


def test_live_semialg_function_range_when_installed():
    pytest.importorskip("semialg")
    x = sp.symbols("x", real=True)
    result = function_range_exact(
        x**2,
        [sp.Ge(x, -2), sp.Le(x, 2)],
        [x],
    )
    assert result is not None
    assert sp.simplify(result.infimum) == 0
    assert sp.simplify(result.supremum - 4) == 0
    assert result.infimum_attained is True
    assert result.supremum_attained is True


def test_live_semialg_disconnected_range_when_installed():
    pytest.importorskip("semialg")
    x = sp.symbols("x", real=True)
    constraints = sp.Or(
        sp.And(sp.Ge(x, -2), sp.Le(x, -1)),
        sp.And(sp.Ge(x, 1), sp.Le(x, 2)),
    )
    result = function_range_exact(x, constraints, [x])
    assert result is not None
    assert sp.simplify(result.infimum + 2) == 0
    assert sp.simplify(result.supremum - 2) == 0
    assert result.infimum_attained is True
    assert result.supremum_attained is True


def test_live_semialg_global_sign_proof_when_installed():
    pytest.importorskip("semialg")
    x = sp.symbols("x", real=True)
    proof = prove_nonnegative_exact(x**4 + x**2 + 1, [], [x])
    assert proof.proven is True


def test_live_semialg_constrained_sign_proof_when_installed():
    pytest.importorskip("semialg")
    x = sp.symbols("x", real=True)
    proof = prove_nonnegative_exact(
        x,
        [sp.Ge(x, 0)],
        [x],
    )
    assert proof.proven is True


def test_live_backend_reports_available():
    pytest.importorskip("semialg")
    assert semialg_available() is True
