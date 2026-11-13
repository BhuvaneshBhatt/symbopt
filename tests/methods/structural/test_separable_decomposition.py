import sympy as sp

from symbopt import argmin, minimize_result


def test_independent_quadratic_components_are_decomposed():
    x, y = sp.symbols("x y", real=True)
    result = minimize_result(x**2 + (y - 3) ** 2, variables=[x, y])
    assert result.certified is True
    assert result.optimum_value == 0
    assert result.structural_result.kind == "separable_decomposition"
    assert result.best_candidate.original_variable_values == {x: 0, y: 3}


def test_mixed_separable_components():
    x, y = sp.symbols("x y", real=True)
    objective = sp.exp(x) - 2 * x + (y - 1) ** 2
    result = minimize_result(objective, variables=[x, y])
    assert result.certified is True
    assert sp.simplify(result.optimum_value - (2 - 2 * sp.log(2))) == 0
    assert result.best_candidate.original_variable_values == {x: sp.log(2), y: 1}
    assert argmin(objective, variables=[x, y]) == ({x: sp.log(2), y: 1},)


def test_separate_constraints_stay_with_their_components():
    x, y = sp.symbols("x y", real=True)
    result = minimize_result(x**2 + y**2, [x >= 1, y <= -2], variables=[x, y])
    assert result.certified is True
    assert result.optimum_value == 5
    assert result.best_candidate.original_variable_values == {x: 1, y: -2}


def test_coupling_constraint_prevents_false_decomposition():
    x, y = sp.symbols("x y", real=True)
    result = minimize_result(x**2 + y**2, [x + y >= 1], variables=[x, y])
    assert (
        result.structural_result is None
        or result.structural_result.kind != "separable_decomposition"
    )
    assert result.certified is True
    assert sp.simplify(result.optimum_value - sp.Rational(1, 2)) == 0
