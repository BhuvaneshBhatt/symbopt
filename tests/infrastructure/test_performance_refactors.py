import numpy as np
import sympy as sp

from symbopt import algebraize, minimize_result
from symbopt.candidates import build_active_set_plan
from symbopt.intervals import EvalConfig, _compile_constraints
from symbopt.sdp import SDPProblem, _build_workspace, _penalty_value_gradient
from symbopt.sos_program import compile_sos_relaxation


def test_polynomial_algebraization_fast_path_records_metadata():
    x, y = sp.symbols("x y", real=True)
    problem = algebraize(x**4 + x * y + 1, [sp.Ge(1 - x**2 - y**2, 0)])
    assert problem.structural_metadata["algebraization_fast_path"] == "polynomial"
    assert problem.auxiliary_variables == []
    assert problem.is_polynomial_problem()


def test_active_set_plan_reports_budget_truncation():
    x = sp.symbols("x", real=True)
    inequalities = [sp.sin(x), sp.cos(x), sp.exp(x)]
    plan = build_active_set_plan(inequalities, [x], 2)
    assert len(plan.patterns) == 2
    assert not plan.complete


def test_sos_blocks_cache_exponent_tuple_basis_data():
    x, y = sp.symbols("x y", real=True)
    compilation = compile_sos_relaxation((x**2 + y**2 + 1) ** 2, [x, y], 2)
    assert compilation.gram_blocks
    assert all(block.basis_terms for block in compilation.gram_blocks)
    assert all(
        isinstance(powers, tuple)
        for block in compilation.gram_blocks
        for _, powers in block.basis_terms
    )


def test_sdp_penalty_gradient_matches_finite_difference():
    problem = SDPProblem.single_block(
        C=np.array([[1.0, 0.2], [0.2, 2.0]]),
        A=[np.array([[1.0, 0.0], [0.0, 0.0]]), np.eye(2)],
        b=np.array([0.8, 1.5]),
        B=np.array([[0.3], [0.1]]),
        c=np.array([0.4]),
    )
    workspace = _build_workspace(problem)
    theta = np.array([0.9, -0.2, 0.7, 0.6], dtype=float)
    value, gradient = _penalty_value_gradient(workspace, theta, 10.0, 1e-6)
    assert np.isfinite(value)
    step = 1e-6
    numerical = np.empty_like(theta)
    for index in range(len(theta)):
        plus = theta.copy()
        plus[index] += step
        minus = theta.copy()
        minus[index] -= step
        vp = _penalty_value_gradient(workspace, plus, 10.0, 1e-6)[0]
        vm = _penalty_value_gradient(workspace, minus, 10.0, 1e-6)[0]
        numerical[index] = (vp - vm) / (2 * step)
    assert np.allclose(gradient, numerical, rtol=2e-5, atol=2e-5)


def test_moment_bridge_not_built_when_sos_is_disabled():
    import symbopt.solver_pipeline as optimization

    x = sp.symbols("x", real=True)
    problem = algebraize(x**4 + x**2, [])
    result = optimization._solve_single_problem(
        problem,
        box_bounds=None,
        interval_max_boxes=5,
        interval_tol=0.1,
        max_active_sets=64,
        use_sos=False,
    )
    assert all(
        item.kind != "moment_bridge" for item in result.certificate.proof_objects
    )


def test_interval_constraints_store_equality_corrections():
    x, y = sp.symbols("x y", real=True)
    compiled = _compile_constraints([sp.Eq(y, x)], EvalConfig())
    assert len(compiled) == 1
    assert {item.symbol for item in compiled[0].equality_corrections} == {x, y}


def test_metadata_refresh_updates_structure_after_rewrite():
    x = sp.symbols("x", real=True)
    problem = algebraize(x**2, [sp.Ge(x, 0)])
    problem.lifted_constraints.append(sp.Eq(sp.sin(x), 0))
    problem.refresh_metadata()
    assert problem.structural_metadata["constraint_count"] == len(
        problem.all_constraints()
    )
    assert problem.structural_metadata["polynomial_problem"] is False
    assert problem.is_polynomial_problem() is False


def test_direct_global_route_does_not_construct_kkt_model():

    x = sp.symbols("x", real=True)
    result = minimize_result((x - 2) ** 2, use_sos=False)
    assert result.certified
    assert result.kkt_system is None


def test_kkt_stage_builds_model_when_candidate_search_needs_it():
    import symbopt.solver_pipeline as pipeline

    x, a = sp.symbols("x a", real=True)
    problem = algebraize(a * x**4 - x**2, [], variables=[x])
    candidates = []
    kkt, _, _, _ = pipeline._search_kkt_stage(
        problem,
        {x: (-2.0, 2.0)},
        64,
        [],
        candidates,
    )
    assert kkt.primal_variables == [x]


def test_cached_residual_groups_are_reused_after_refresh():
    x = sp.symbols("x", real=True)
    problem = algebraize(x**2, [sp.Ge(x, 0)])
    eqs, inequalities = problem.kkt_residuals()
    assert eqs == []
    assert inequalities == [x]
    assert problem.branch_residuals() == []
    assert problem.polynomial_residuals() == [x]
