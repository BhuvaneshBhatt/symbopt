import sympy as sp

from symbopt import maximize_result, minimize_result
from symbopt.mixed_periodic import decompose_mixed_periodic


def test_mixed_periodic_sum_has_certified_dense_range_infimum():
    x = sp.symbols("x", real=True)
    result = minimize_result(sp.sin(x) + sp.cos(sp.sqrt(2) * x), variables=x)
    assert result.certified is True
    assert result.optimum_value == -2
    assert result.method == "mixed_periodic"
    assert result.attained is None
    assert result.structural_result.metadata["inverse_periods_q_independent"] is True


def test_periodic_maximum():
    x = sp.symbols("x", real=True)
    result = maximize_result(sp.sin(x) + sp.cos(sp.sqrt(2) * x), variables=x)
    assert result.certified is True
    assert result.optimum_value == 2
    assert not result.infeasible and not result.unbounded
    assert result.attained in (False, None)


def test_commensurate_periodic_groups():
    x = sp.symbols("x", real=True)
    objective = sp.sin(x) + sp.sin(2 * x) + sp.cos(sp.sqrt(2) * x)
    decomposition = decompose_mixed_periodic(objective, x)
    assert len(decomposition.components) == 2
    assert any(
        component.has(sp.sin(x), sp.sin(2 * x))
        for component in decomposition.components
    )
    result = minimize_result(objective, variables=x)
    assert result.certified is True
    assert result.method == "mixed_periodic"
    cosine = (sp.sqrt(33) - 1) / 8
    expected = -1 - (1 + 2 * cosine) * sp.sqrt(1 - cosine * cosine)
    assert sp.simplify(result.optimum_value - expected) == 0


def test_mixed_periodic_supports_nonlinear_outer_expression():
    x = sp.symbols("x", real=True)
    objective = sp.exp(sp.sin(x) + sp.cos(sp.sqrt(2) * x))
    result = minimize_result(objective, variables=x)
    assert result.certified is True
    assert sp.simplify(result.optimum_value - sp.exp(-2)) == 0
    assert result.method == "mixed_periodic"


def test_periodic_product_certificate():
    x = sp.symbols("x", real=True)
    objective = sp.sin(x) * sp.cos(sp.sqrt(2) * x)
    result = minimize_result(objective, variables=x)
    assert result.certified is True
    assert result.optimum_value == -1
    assert result.method == "mixed_periodic"


def test_q_dependent_inverse_periods_are_rejected():
    x = sp.symbols("x", real=True)
    objective = sp.sin(x) + sp.sin(sp.sqrt(2) * x) + sp.sin((1 + sp.sqrt(2)) * x)
    assert decompose_mixed_periodic(objective, x) is None


def test_bounded_domain_does_not_use_dense_orbit_reduction():
    x = sp.symbols("x", real=True)
    result = minimize_result(
        sp.sin(x) + sp.cos(sp.sqrt(2) * x),
        constraints=[x >= 0, x <= 1],
        variables=x,
    )
    assert result.method != "mixed_periodic"
