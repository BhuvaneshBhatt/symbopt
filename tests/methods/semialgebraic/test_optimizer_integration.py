from __future__ import annotations

import pytest
import sympy as sp

from symbopt import minimize
from symbopt.semialgebraic import get_semialg_capabilities, minimize_exact


pytestmark = pytest.mark.integration


def _require_modern_semialg():
    pytest.importorskip("semialg")
    caps = get_semialg_capabilities()
    if not (
        caps.has_optimizer
        and caps.supports_cert_policy
        and caps.supports_range_limit
        and caps.result_has_certified
    ):
        pytest.skip("semialg optimizer certification API is unavailable")
    return caps


def test_semialg_adapter_certifies_three_variable_nonconvex_problem():
    _require_modern_semialg()
    x, y, z = sp.symbols("x y z", real=True)
    result = minimize_exact(
        (x**2 - 1) ** 2 + y**2 + z**2,
        [],
        [x, y, z],
        certification="complete",
        range_cost_limit=0,
    )
    assert result is not None
    assert result.value == 0
    assert result.attained is True
    assert result.certified is True
    assert len(result.points) == 2
    assert result.diagnostics["certification_policy"] == "complete"


def test_pipeline_uses_semialg_early_for_four_variable_nonconvex_problem():
    _require_modern_semialg()
    a, b, c, d = sp.symbols("a b c d", real=True)
    result = minimize(
        (a * b - 1) ** 2 + (b - c) ** 2 + (c - d) ** 2 + (d - 1) ** 2,
        [],
        use_sos=False,
        semialg_certification="auto",
    )
    assert result.certified
    assert result.status == "global_certified_by_semialg"
    assert result.optimum_value == 0
    assert result.attained is True
    assert result.semialg_result is not None
    assert result.semialg_result.certified is True
    assert result.sos_result is None
    assert result.interval_result is None
    assert all(candidate.source == "semialg" for candidate in result.candidates)


def test_pipeline_preserves_unattained_three_variable_infimum_from_semialg():
    _require_modern_semialg()
    x, y, z = sp.symbols("x y z", real=True)
    result = minimize(
        x,
        [x**2 + y**2 + z**2 < 1],
        use_sos=False,
        semialg_certification="complete",
        semialg_range_cost_limit=0,
    )
    assert result.certified
    assert result.status == "global_infimum_by_semialg"
    assert result.optimum_value == -1
    assert result.attained is False
    assert result.best_candidate is None
    assert result.semialg_result is not None
    assert result.semialg_result.attained is False
    assert "cad_range_certificate" in result.semialg_result.method


def test_positive_dimensional_kkt_locus_does_not_require_finite_candidates():
    _require_modern_semialg()
    x, y = sp.symbols("x y", real=True)
    result = minimize(
        x**2 + y**2,
        [sp.Eq(x**2 + y**2, 1)],
        use_sos=False,
    )
    assert result.certified
    assert result.optimum_value == 1
    assert result.attained is True
    assert result.semialg_result is not None
    assert result.semialg_result.certified is True
    # The entire circle is optimal.  symbopt must accept semialg's certified
    # value/attainment result even when no finite optimizer list is returned.
    assert result.best_candidate is None
    assert result.candidates == []
