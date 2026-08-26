import sympy as sp

from symbopt import minimize


def test_qp_exact_postprocessing_certificate():
    x = sp.symbols("x", real=True)
    result = minimize((x + 2) ** 2)
    assert result.best_candidate.primal_values[x] == -2
    artifact = next(
        item for item in result.certificate.proof_objects if item.kind == "qp"
    )
    post = artifact.payload["exact_postprocessing"]
    assert post["exact_recovered"] is True
    assert all(value == "0" for value in post["stationarity_residuals"])
