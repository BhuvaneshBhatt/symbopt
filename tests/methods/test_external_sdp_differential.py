from __future__ import annotations

import pytest
import sympy as sp

from symbopt import SDPSolverConfig, available_external_sdp_solvers, sos_decompose


@pytest.mark.integration
def test_sdp_exact_certificate_agreement():
    x, y = sp.symbols("x y", real=True)
    polynomial = (x + y) ** 2 + (x - y) ** 2
    results = {
        "internal": sos_decompose(
            polynomial,
            [x, y],
            solver_config=SDPSolverConfig(backend="internal", max_iter=5000, tol=1e-8),
        )
    }
    for solver in ("CLARABEL", "SCS"):
        if solver in available_external_sdp_solvers():
            results[solver] = sos_decompose(
                polynomial,
                [x, y],
                solver_config=SDPSolverConfig(
                    backend="cvxpy", external_solver=solver, max_iter=5000, tol=1e-8
                ),
            )
    assert results["internal"].certificate is not None
    for name, result in results.items():
        assert result.success, name
        assert result.certificate.verified, name
        assert sp.expand(result.exact_residual) == 0, name
