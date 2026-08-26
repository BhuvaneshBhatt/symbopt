import sympy as sp

from symbopt import minimize


def test_same_frequency_sine_cosine_global_minimum_exact():
    x = sp.symbols("x", real=True)
    objective = 3 * sp.sin(2 * x) + 4 * sp.cos(2 * x) + 1
    result = minimize(objective, variables=[x], use_sos=False)
    assert result.certified is True
    assert result.optimum_value == -4
    assert result.attained is True
    assert result.structural_result.kind in {"periodic_trig", "rational_frequency_trig"}
    assert result.best_candidate is not None
    point = result.best_candidate.original_variable_values[x]
    assert sp.simplify(objective.subs(x, point) + 4) == 0


def test_periodic_trig_respects_closed_interval():
    x = sp.symbols("x", real=True)
    objective = 3 * sp.sin(2 * x) + 4 * sp.cos(2 * x) + 1
    result = minimize(
        objective,
        [sp.Ge(x, 0), sp.Le(x, sp.pi / 2)],
        variables=[x],
        use_sos=False,
    )
    assert result.certified is True
    assert result.optimum_value == -3
    assert result.attained is True
    assert result.best_candidate.original_variable_values[x] == sp.pi / 2
