import sympy as sp

from symbopt import minimize


def test_lp_exact_postprocessing_with_equality():
    x = sp.symbols("x", real=True)
    result = minimize(x, [sp.Eq(x, -2)])
    assert result.best_candidate.primal_values[x] == -2
    artifact = next(
        item for item in result.certificate.proof_objects if item.kind == "lp"
    )
    assert artifact.payload["exact_postprocessing"]["exact_recovered"] is True
