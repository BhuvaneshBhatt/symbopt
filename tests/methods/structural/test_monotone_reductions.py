import sympy as sp

from symbopt import minimize


def _has_monotone_artifact(result):
    return any(
        artifact.kind == "monotone_reduction"
        for artifact in result.certificate.proof_objects
    )


def test_exponential_of_quadratic_reduces_to_quadratic():
    x = sp.symbols("x", real=True)
    result = minimize(sp.exp((x - 2) ** 2), variables=[x], use_sos=False)
    assert result.certified is True
    assert result.optimum_value == 1
    assert result.best_candidate.original_variable_values[x] == 2
    assert result.best_candidate.source.startswith("monotone_exp:")
    assert _has_monotone_artifact(result)


def test_positive_base_exponential_reduction():
    x = sp.symbols("x", real=True)
    result = minimize(2 ** ((x - 1) ** 2), variables=[x], use_sos=False)
    assert result.certified is True
    assert result.optimum_value == 1
    assert result.best_candidate.original_variable_values[x] == 1
    assert "positive_base_exp" in result.best_candidate.source


def test_log_reduction_adds_domain_and_preserves_exact_value():
    x = sp.symbols("x", real=True)
    result = minimize(sp.log(x), [sp.Ge(x, 1)], variables=[x], use_sos=False)
    assert result.certified is True
    assert result.optimum_value == 0
    assert result.attained is True
    assert result.best_candidate.original_variable_values[x] == 1


def test_square_root_monotone_reduction():
    x = sp.symbols("x", real=True)
    result = minimize(sp.sqrt(x**2 + 1), variables=[x], use_sos=False)
    assert result.certified is True
    assert result.optimum_value == 1
    assert result.best_candidate.original_variable_values[x] == 0


def test_even_integer_power_is_not_misclassified_as_monotone():
    x = sp.symbols("x", real=True)
    result = minimize((x + 2) ** 2, variables=[x], use_sos=False)
    assert result.certified is True
    assert result.optimum_value == 0
    assert result.best_candidate.original_variable_values[x] == -2
    assert not result.best_candidate.source.startswith("monotone_positive_power")


def test_decreasing_positive_base_exponential_reduction():
    x = sp.symbols("x", real=True)
    result = minimize(
        sp.Rational(1, 2) ** x,
        [sp.Ge(x, 0), sp.Le(x, 3)],
        variables=[x],
        use_sos=False,
    )
    assert result.certified is True
    assert result.optimum_value == sp.Rational(1, 8)
    assert result.best_candidate.original_variable_values[x] == 3


def test_positive_rational_power_reduction_on_real_domain():
    x = sp.symbols("x", real=True)
    result = minimize(
        (x**2 + 1) ** sp.Rational(3, 2),
        variables=[x],
        use_sos=False,
    )
    assert result.certified is True
    assert result.optimum_value == 1
    assert result.best_candidate.original_variable_values[x] == 0
