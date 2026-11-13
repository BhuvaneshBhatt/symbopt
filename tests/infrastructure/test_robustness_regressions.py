import pytest
import sympy as sp

from symbopt import minimize_result
from symbopt.algebraize import algebraize
from symbopt.convex_routing import infer_box_bounds, try_direct_solver
from symbopt.planner import analyze_problem


def test_planner_prices_constrained_sos_relaxation():
    x, y = sp.symbols("x y", real=True)
    unconstrained = analyze_problem(x**4 + y**4, [], [x, y])
    constrained = analyze_problem(
        x**4 + y**4,
        [1 - x**2 >= 0, 1 - y**2 >= 0, x + y - 1 >= 0],
        [x, y],
    )
    assert (
        constrained.estimated_sos_scalar_variables
        > unconstrained.estimated_sos_scalar_variables
    )


def test_explicit_empty_variables_are_rejected():
    x = sp.symbols("x", real=True)
    with pytest.raises(ValueError, match="at least one optimization variable"):
        minimize_result(x**2, variables=[])


def test_foreign_bound_variable_is_rejected():
    x, y = sp.symbols("x y", real=True)
    with pytest.raises(ValueError, match="not an optimization variable"):
        minimize_result(x**2, variables=x, bounds={y: (0, 1)})
    with pytest.raises(ValueError, match="not optimization variables"):
        minimize_result(x**2, variables=x, box_bounds={y: (0, 1)})


def test_infer_box_bounds_ignores_foreign_relations_defensively():
    x, y = sp.symbols("x y", real=True)
    assert infer_box_bounds([x], [y >= 0, y <= 1]) == {}


def test_direct_symbolic_quadratic():
    x, a = sp.symbols("x a", real=True)
    exact = try_direct_solver((x - a) ** 2, [], [x])
    assert exact.point == {x: a} and exact.objective_value == 0
    assert exact.metadata["rigorous_global"] is True
    result = minimize_result((x - a) ** 2, variables=x)
    assert result.status


def test_strict_kkt_closure():
    x = sp.symbols("x", real=True)
    problem = algebraize(x**2, [x > 0], variables=[x])
    with pytest.raises(ValueError, match="Strict inequalities"):
        problem.kkt_residuals(close_strict=False)


def test_kkt_disequality_contract():
    x = sp.symbols("x", real=True)
    problem = algebraize(x**2, [sp.Ne(x, 0)], variables=[x])
    with pytest.raises(ValueError, match="disequality"):
        problem.kkt_residuals()
    result = minimize_result(x**2, sp.Ne(x, 0), variables=x)
    assert result.certified and result.optimum_value == 0
    assert result.attained is False and result.best_candidate is None


def test_explicit_flat_default_wins_over_grouped_option():
    x = sp.symbols("x", real=True)
    result = minimize_result(
        (x - 2) ** 2,
        variables=x,
        interval_tol=1e-3,
        options={"interval": {"tol": 0.2}},
    )
    # If precedence parsing succeeds, the ordinary QP result is unaffected.
    assert result.optimum_value == 0


def test_solver_options_validate_values_eagerly():
    x = sp.symbols("x", real=True)
    with pytest.raises(TypeError, match="boolean"):
        minimize_result(x**2, variables=x, options={"sos": {"enabled": "false"}})
    with pytest.raises(ValueError, match="positive finite"):
        minimize_result(x**2, variables=x, options={"interval": {"tol": -1}})
    with pytest.raises(ValueError, match="nonnegative integer"):
        minimize_result(x**2, variables=x, options={"kkt": {"max_active_sets": -1}})


def test_explain_distinguishes_unknown_attainment():
    x = sp.symbols("x", real=True)
    result = minimize_result((x - 1) ** 2, variables=x)
    result.attained = None
    text = result.explain()
    assert "attainment unknown" in text
    assert "not attained" not in text


def test_method_label_is_structured_and_not_status_dependent():
    x = sp.symbols("x", real=True)
    result = minimize_result((x - 1) ** 2, variables=x)
    assert result.method_label == "qp"
    result.status = "wording may change freely"
    assert result.method == "qp"


def test_solver_option_precedence():
    from symbopt.solver_options import normalize_solver_options

    normalized = normalize_solver_options(
        interval_tol=1e-3,
        max_active_sets=0,
        options={"interval": {"tol": 0.25}, "kkt": {"max_active_sets": 12}},
    )
    assert normalized.interval_tol == 1e-3
    assert normalized.max_active_sets == 0


def test_maximization_kkt_signs():
    from symbopt.kkt import build_kkt_system
    from symbopt.problem_models import CandidateSolution
    from symbopt.solver_pipeline import _mapped_candidate, _mapped_kkt_system

    x = sp.symbols("x", real=True)
    problem = algebraize(-(x**2), [sp.Eq(x, 1)], variables=[x])
    kkt = build_kkt_system(problem)
    lam = kkt.equality_multipliers[0]
    mapped = _mapped_kkt_system(kkt)
    assert sp.expand(mapped.lagrangian + kkt.lagrangian) == 0
    assert (
        sp.simplify(
            mapped.stationarity_equations[0].lhs + kkt.stationarity_equations[0].lhs
        )
        == 0
    )

    candidate = CandidateSolution(
        primal_values={x: 1},
        multiplier_values={lam: 2},
        objective_value=-1,
        objective_value_numeric=-1.0,
        feasibility_checked=True,
        source="kkt",
        local_classification="LocalMinimum",
        original_variable_values={x: 1},
    )
    mapped_candidate = _mapped_candidate(candidate, [lam])
    assert mapped_candidate.multiplier_values[lam] == -2
    assert mapped_candidate.objective_value == 1
    assert mapped_candidate.local_classification == "LocalMaximum"
