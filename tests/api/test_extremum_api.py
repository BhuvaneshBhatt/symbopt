import sympy as sp

from symbopt import argmax, argmin, maximize_result, maxval, minval

x, y = sp.symbols("x y", real=True)


def test_minval_and_maxval_exact_quadratic_interval():
    constraints = [sp.Ge(x, -2), sp.Le(x, 3)]
    assert sp.simplify(minval((x - 1) ** 2, constraints, use_sos=False)) == 0
    assert sp.simplify(maxval(x, constraints, use_sos=False) - 3) == 0


def test_maximize_maps_full_result_to_original_objective():
    result = maximize_result(-((x - 2) ** 2) + 5, use_sos=False)
    assert result.certified
    assert sp.simplify(result.optimum_value - 5) == 0
    assert sp.simplify(result.best_candidate.objective_value - 5) == 0
    assert sp.simplify(result.best_candidate.original_variable_values[x] - 2) == 0
    assert sp.simplify(result.certificate.lower_bound - 5) == 0
    assert sp.simplify(result.certificate.upper_bound - 5) == 0


def test_argmin_and_argmax_return_optimizer_mappings():
    mins = argmin((x - 1) ** 2, use_sos=False)
    maxs = argmax(-((x + 2) ** 2), use_sos=False)
    assert any(sp.simplify(point[x] - 1) == 0 for point in mins)
    assert any(sp.simplify(point[x] + 2) == 0 for point in maxs)


def test_minval_unattained_infimum_and_argmin_empty():
    constraints = [sp.Gt(x, 0)]
    assert minval(1 / x, constraints, use_sos=False) == 0
    assert argmin(1 / x, constraints, use_sos=False) == ()


def test_maxval_unattained_supremum_and_argmax_empty():
    constraints = [sp.Gt(x, 0)]
    assert maxval(-1 / x, constraints, use_sos=False) == 0
    assert argmax(-1 / x, constraints, use_sos=False) == ()


def test_decision_parameter_roles():
    a = sp.symbols("a", positive=True)
    assert minval(a * x**2, variables=[x], use_sos=False) == 0
    assert argmin(a * x**2, variables=[x], use_sos=False) == ({x: sp.S.Zero},)


def test_minval_exact_periodic_cosine():
    assert minval(sp.cos(x), use_sos=False) == -1


def test_argmax_exact_periodic_sine():
    points = argmax(sp.sin(x), use_sos=False)
    assert any(sp.simplify(point[x] - sp.pi / 2) == 0 for point in points)


def test_maximize_preserves_normalized_kkt_provenance():
    objective = -(x**4) + 2 * x**2
    result = maximize_result(objective, use_sos=False)
    assert result.optimization_sense == "maximize"
    assert result.lifted_problem.original_objective == objective
    if result.kkt_system is None:
        # A direct exact backend may certify the maximization before KKT search.
        assert result.certified
        assert result.kkt_objective is None
    else:
        assert sp.simplify(result.kkt_objective + objective) == 0
        assert sp.simplify(result.kkt_system.lagrangian - result.kkt_objective) == 0
