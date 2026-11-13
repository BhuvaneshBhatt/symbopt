"""Branch assembly keeps exact infima separate from partial bounds."""

from dataclasses import replace

import sympy as sp

from symbopt import minimize_result
from symbopt.solver_pipeline import _merge_piecewise_results

x = sp.Symbol("x", real=True)


def test_lower_unattained_branch():
    unattained = minimize_result(x - 5, x > 0, variables=x)
    attained = minimize_result(x * x - 4, variables=x)
    result = _merge_piecewise_results([unattained, attained])
    assert result.certified and result.optimum_value == -5
    assert result.attained is False
    assert result.certificate.lower_bound == -5
    assert result.certificate.status == "proven_infimum"


def test_unresolved_branch():
    exact = minimize_result(x * x - 4, variables=x)
    candidate = minimize_result(x * x - 5, variables=x)
    unfinished = replace(
        candidate,
        certified=False,
        optimum_value=None,
        attained=None,
        certificate=replace(
            candidate.certificate, lower_bound=-sp.S(10), gap=sp.S(5), status="bounded"
        ),
    )
    result = _merge_piecewise_results([exact, unfinished])
    assert not result.certified and result.optimum_value is None
    assert result.certificate.status == "bounded"
    assert result.certificate.lower_bound == -10
    assert result.certificate.upper_bound == -5
    assert result.certificate.gap == 5
