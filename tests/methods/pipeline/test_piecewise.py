from types import SimpleNamespace

import sympy as sp

from symbopt.problem_models import (
    CandidateSolution,
    OptimalityCertificate,
)
from symbopt.solver_pipeline import _merge_piecewise_results


def _candidate(value):
    x = sp.symbols("x", real=True)
    return CandidateSolution(
        primal_values={x: sp.Integer(0)},
        multiplier_values={},
        objective_value=sp.Integer(value),
        objective_value_numeric=float(value),
        feasibility_checked=True,
        source="test",
        original_variable_values={x: sp.Integer(0)},
    )


def _branch(lower, candidate=None, status="proven_optimal"):
    certificate = OptimalityCertificate(
        lower_bound=sp.Integer(lower),
        upper_bound=(candidate.objective_value if candidate is not None else None),
        primal_value=(candidate.objective_value if candidate is not None else None),
        status=status,
    )
    return SimpleNamespace(
        candidates=[] if candidate is None else [candidate],
        best_candidate=candidate,
        certificate=certificate,
        certified=True,
        certification_gap=0.0 if candidate is not None else None,
        status=status,
    )


def test_piecewise_prefers_lower_unattained_infimum():
    unattained = _branch(-5, None, "proven_infimum")
    attained = _branch(-4, _candidate(-4), "proven_optimal")

    result = _merge_piecewise_results([unattained, attained])

    assert result.certified is True
    assert result.status == "global_piecewise_infimum"
    assert result.certificate.lower_bound == -5
    assert result.certificate.status == "proven_infimum"


def test_piecewise_downgrades_when_a_branch_is_unresolved():
    exact = _branch(-4, _candidate(-4), "proven_optimal")
    unresolved = _branch(-10, None, "bounded")
    unresolved.certified = False

    result = _merge_piecewise_results([exact, unresolved])

    assert result.certified is False
    assert result.certificate.status in {"bounded", "candidate_only"}
