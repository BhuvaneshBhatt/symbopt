import pytest
import sympy as sp

from symbopt import (
    argmin,
    estimate_sos_cost,
    maximize,
    minimize,
    minimize_local,
    minimize_result,
    minval,
    search_infeasibility_certificate,
)


def test_bounds_mapping_and_sequence_forms():
    x = sp.symbols("x")
    assert minimize((x - 3) ** 2, bounds={x: (0, 1)}, variables=x, use_sos=False) == [
        4,
        {x: 1},
    ]
    assert minimize((x - 3) ** 2, bounds=[(0, 1)], variables=x, use_sos=False) == [
        4,
        {x: 1},
    ]


def test_or_constraints_are_optimized_by_branches():
    x = sp.symbols("x")
    result = minimize_result(x**2, sp.Or(x <= -1, x >= 2), variables=x, use_sos=False)
    assert result.optimum_value == 1
    assert result.best_candidate.original_variable_values == {x: -1}
    assert result.method == "branch"


def test_generic_iterable_constraints_and_false_constraint():
    x = sp.symbols("x")
    constraints = (item for item in [x >= 1, x <= 2])
    assert minimize(x**2, constraints, variables=x, use_sos=False) == [1, {x: 1}]
    with pytest.raises(TypeError, match="False is not a relational constraint"):
        minimize(x, False, variables=x, use_sos=False)


def test_symbol_assumptions_are_opt_in():
    x = sp.symbols("x", positive=True)
    result = minimize_result(x, variables=x, use_sos=False, use_symbol_assumptions=True)
    assert result.optimum_value == 0
    assert result.attained is False


def test_multiple_and_continuum_optimizers():
    x, y = sp.symbols("x y")
    result = minimize((x**2 - 1) ** 2, variables=x, use_sos=False)
    assert result == [0, ({x: -1}, {x: 1})]
    assert argmin((x**2 - 1) ** 2, variables=x, use_sos=False) == ({x: -1}, {x: 1})
    assert minimize(x**2, variables=[x, y], use_sos=False) == [0, {x: 0}]


def test_status_properties_method_and_explain():
    x = sp.symbols("x")
    infeasible = minimize_result(x, [x > 1, x < 0], variables=x, use_sos=False)
    assert infeasible.infeasible
    assert not infeasible.unbounded
    assert "feasible set is empty" in infeasible.explain()
    unbounded = minimize_result(x, variables=x, use_sos=False)
    assert unbounded.unbounded
    assert not unbounded.infeasible
    assert isinstance(unbounded.method, str)


def test_hierarchical_options_and_validation():
    x = sp.symbols("x")
    result = minimize_result(
        (x - 1) ** 2,
        variables=x,
        options={"sos": {"enabled": False}, "kkt": {"max_active_sets": 8}},
    )
    assert result.optimum_value == 0
    with pytest.raises(ValueError, match="unknown option section"):
        minimize_result(x**2, variables=x, options={"magic": {}})


def test_sos_input_conveniences():
    x = sp.symbols("x")
    estimate = estimate_sos_cost(x**2, variables=x, constraints=sp.And(x >= -1, x <= 1))
    assert estimate.applicable
    estimate2 = estimate_sos_cost(x**2, constraints=x >= 0)
    assert estimate2.variable_count == 1
    estimate3 = estimate_sos_cost(x**2, inequalities=[x >= 0])
    assert estimate3.applicable


def test_infeasibility_variable_inference():
    x = sp.symbols("x")
    result = search_infeasibility_certificate(
        constraints=sp.And(x >= 1, x <= 0),
        planner=True,
    )
    assert result.plan.estimate.variable_count == 1


def test_constraints_are_optional():
    x = sp.symbols("x", real=True)
    assert minimize((x - 2) ** 2, variables=[x]) == [0, {x: 2}]


def test_single_relational_constraint_is_accepted():
    x = sp.symbols("x", real=True)
    assert minimize(x**2, x >= 1, variables=[x]) == [1, {x: 1}]


def test_and_constraint_is_flattened():
    x = sp.symbols("x", real=True)
    constraints = sp.And(x >= 1, x <= 3)
    assert minimize((x - 2) ** 2, constraints, variables=[x]) == [0, {x: 2}]


def test_nested_and_inside_list_is_flattened():
    x, y = sp.symbols("x y", real=True)
    constraints = [x >= 0, sp.And(y >= 0, x + y >= 1)]
    result = minimize(x**2 + y**2, constraints, variables=[x, y])
    assert sp.simplify(result[0] - sp.Rational(1, 2)) == 0
    assert result[1] == {x: sp.Rational(1, 2), y: sp.Rational(1, 2)}


def test_true_constraint_is_ignored():
    x = sp.symbols("x", real=True)
    assert minval((x - 1) ** 2, sp.S.true, variables=[x]) == 0


def test_result_api_accepts_and():
    x = sp.symbols("x", real=True)
    result = minimize_result(x, sp.And(x >= -1, x <= 1), variables=[x])
    assert result.optimum_value == -1


def test_maximize_accepts_and():
    x = sp.symbols("x", real=True)
    assert maximize(x, sp.And(x >= -1, x <= 1), variables=[x]) == [1, {x: 1}]


def test_or_constraint_is_split_into_feasible_branches():
    x = sp.symbols("x", real=True)
    result = minimize(x**2, sp.Or(x <= -1, x >= 1), variables=[x])
    assert result == [1, ({x: -1}, {x: 1})]


def test_single_symbol_variables_form():
    x = sp.symbols("x")
    assert minimize((x - 3) ** 2, variables=x) == [0, {x: 3}]
    result = minimize_result(x**2, x >= 2, variables=x)
    assert result.optimum_value == 4


def test_single_symbol_variables_is_accepted_by_local_optimizer():
    x = sp.symbols("x")
    result = minimize_local((x - 4) ** 2, variables=x, initial_point={x: 0})
    assert result.success
    assert result.exact_point == {x: 4}
