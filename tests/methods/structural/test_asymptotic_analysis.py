import sympy as sp

from symbopt import minimize_result


def _assert_asymptotic_or_monotone(result):
    if result.structural_result is not None:
        assert result.structural_result.kind in {
            "univariate_asymptotic",
            "univariate_polynomial",
            "transcendental_domain",
        }
    else:
        assert any(
            artifact.kind == "monotone_reduction"
            for artifact in result.certificate.proof_objects
        )


def test_exponential_infimum():
    x = sp.symbols("x", real=True)
    result = minimize_result(
        sp.exp(x),
        variables=[x],
        use_sos=False,
        options={"structural": {"enabled": False}},
    )
    assert result.certified is True
    assert result.optimum_value == 0
    assert result.attained is False
    _assert_asymptotic_or_monotone(result)


def test_logarithm_positive_domain_is_unbounded_below():
    x = sp.symbols("x", real=True)
    result = minimize_result(
        sp.log(x),
        [sp.Gt(x, 0)],
        variables=[x],
        use_sos=False,
        options={"structural": {"enabled": False}},
    )
    assert result.certified is True
    assert result.optimum_value is -sp.oo
    assert result.attained is False
    _assert_asymptotic_or_monotone(result)


def test_rational_singularity_analysis_on_real_domain():
    x = sp.symbols("x", real=True)
    result = minimize_result(
        1 / x, variables=[x], use_sos=False, options={"structural": {"enabled": False}}
    )
    assert result.certified is True
    assert result.optimum_value is -sp.oo
    assert result.attained is False
    assert result.structural_result.kind == "linear_fractional"
