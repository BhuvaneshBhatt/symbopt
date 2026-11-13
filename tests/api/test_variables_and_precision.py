import pytest
import sympy as sp

from symbopt import minimize, minimize_result


def test_explicit_variables_keep_other_symbols_as_parameters():
    x = sp.symbols("x", real=True)
    a = sp.symbols("a", positive=True)
    result = minimize_result(a * x**2, variables=[x], use_sos=False)
    assert result.lifted_problem.original_variables == [x]
    assert result.lifted_problem.parameter_symbols == [a]
    assert result.certified and result.optimum_value == 0
    assert result.best_candidate.original_variable_values == {x: sp.S.Zero}


def test_variables_are_inferred_from_free_symbols_when_omitted():
    x, y = sp.symbols("x y", real=True)
    result = minimize_result(x**2 + y**2, use_sos=False)
    assert result.lifted_problem.original_variables == [x, y]


def test_explicit_variable_validation_and_free_coordinates():
    x, y = sp.symbols("x y", real=True)
    assert minimize(x**2, variables=x, use_sos=False) == [0, {x: 0}]
    with pytest.raises(TypeError):
        minimize(x**2, variables=[x, 3])
    with pytest.raises(ValueError):
        minimize(x**2, variables=[x, x])
    assert minimize(x**2, variables=[x, y], use_sos=False) == [0, {x: 0}]


def test_working_precision_and_input_precision_are_recorded():
    x = sp.symbols("x", real=True)
    objective = (x - sp.Float("1.25", 50)) ** 2
    result = minimize_result(
        objective,
        variables=[x],
        working_precision=80,
        use_sos=False,
    )
    provenance = result.precision_provenance
    assert result.working_precision == 80
    assert provenance["requested_precision"] == 80
    assert provenance["effective_precision"] == 80
    assert provenance["contains_inexact_input"] is True
    assert provenance["min_input_precision"] >= 49
    assert provenance["max_input_precision"] >= 49


def test_working_precision_validation():
    x = sp.symbols("x", real=True)
    with pytest.raises(ValueError):
        minimize(x**2, working_precision=0)
    with pytest.raises(TypeError):
        minimize(x**2, working_precision=20.5)
