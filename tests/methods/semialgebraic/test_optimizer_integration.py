from __future__ import annotations

import pytest
import sympy as sp

from symbopt import minimize_result
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


def test_semialgebraic_nonconvex_adapter():
    _require_modern_semialg()
    x, y, z = sp.symbols("x y z", real=True)
    result = minimize_exact(
        (x**2 - 1) ** 2 + y**2 + z**2,
        [],
        [x, y, z],
        certification="complete",
        range_cost_limit=0,
    )
    assert result.value == 0
    assert result.attained is True
    assert result.certified is True
    assert result.points
    assert all(
        sp.simplify((point[x] ** 2 - 1) ** 2 + point[y] ** 2 + point[z] ** 2) == 0
        for point in result.points
    )
    assert result.diagnostics["certification_policy"] == "complete"


def test_semialgebraic_nonconvex_routing():
    _require_modern_semialg()
    a, b, c, d = sp.symbols("a b c d", real=True)
    result = minimize_result(
        (a * b - 1) ** 2 + (b - c) ** 2 + (c - d) ** 2 + (d - 1) ** 2,
        [],
        use_sos=False,
        options={"structural": {"enabled": False}},
        semialg_certification="auto",
    )
    assert result.certified
    assert result.status == "global_certified_by_semialg"
    assert result.optimum_value == 0
    assert result.attained is True
    assert result.semialg_result.certified is True
    assert result.sos_result is None
    assert result.interval_result is None
    assert all(candidate.source == "semialg" for candidate in result.candidates)


def test_semialgebraic_unattained_infimum():
    _require_modern_semialg()
    x, y, z = sp.symbols("x y z", real=True)
    result = minimize_result(
        x,
        [x**2 + y**2 + z**2 < 1],
        use_sos=False,
        options={"structural": {"enabled": False}},
        semialg_certification="complete",
        semialg_range_cost_limit=0,
    )
    assert result.certified
    assert result.status == "global_infimum_by_semialg"
    assert result.optimum_value == -1
    assert result.attained is False
    assert result.best_candidate is None
    assert result.semialg_result.attained is False


def test_positive_dimensional_kkt_locus():
    _require_modern_semialg()
    x, y = sp.symbols("x y", real=True)
    result = minimize_result(
        x**2 + y**2,
        [sp.Eq(x**2 + y**2, 1)],
        use_sos=False,
        options={"structural": {"enabled": False}},
    )
    assert result.certified
    assert result.optimum_value == 1
    assert result.attained is True
    assert result.semialg_result.certified is True
    # The entire circle is optimal.  semialg may return either no finite
    # optimizer enumeration or any certified representative point on it.
    if result.best_candidate is not None:
        point = result.best_candidate.original_variable_values
        assert sp.simplify(point[x] ** 2 + point[y] ** 2 - 1) == 0
        assert result.best_candidate.objective_value == 1
