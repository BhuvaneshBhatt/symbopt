import pytest
import sympy as sp

from symbopt import (
    SDPProblem,
    SDPSolverConfig,
    analyze_critical_cone,
    available_external_sdp_solvers,
    plan_problem,
    project_affine_exact,
    solve_sdp,
    sos_decompose,
)
from symbopt.verified_intervals import compile_arb


def test_planner_uses_finite_box_and_polynomial_metrics():
    x, y = sp.symbols("x y", real=True)
    plan = plan_problem(x**2 + y**2, [x + y >= 1], [x, y], {x: (-2, 2), y: (-2, 2)})
    assert plan.complexity.polynomial is True
    assert plan.complexity.max_degree == 2
    assert plan.complexity.finite_box is True
    assert plan.try_interval is True
    assert plan.try_exact_kkt is True


def test_exact_affine_projection_recovers_joint_solution():
    x, y = sp.symbols("x y")
    projected = project_affine_exact(
        [x + y - 1], [x, y], [sp.Rational(49, 100), sp.Rational(49, 100)]
    )
    assert projected == {x: sp.Rational(1, 2), y: sp.Rational(1, 2)}


def test_external_sdp_backend_is_capability_aware():
    problem = SDPProblem.single_block([[1.0]], [[[1.0]]], [1.0])
    solvers = available_external_sdp_solvers()
    if not solvers:
        with pytest.raises((ImportError, RuntimeError)):
            solve_sdp(problem, SDPSolverConfig(backend="cvxpy"))
        return

    result = solve_sdp(
        problem,
        SDPSolverConfig(backend="cvxpy", external_solver=solvers[0], tol=1e-8),
    )
    assert result.status in {"optimal", "optimal_inaccurate"}
    assert result.primal_residual_norm <= 1e-6
    assert "requires exact post-verification" in result.message


@pytest.mark.parametrize("solver", ["CLARABEL", "SCS"])
def test_external_sos_replay(solver):
    if solver not in available_external_sdp_solvers():
        pytest.skip(f"{solver} is not installed")

    x, y = sp.symbols("x y", real=True)
    result = sos_decompose(
        (x + y) ** 2 + (x - y) ** 2,
        [x, y],
        solver_config=SDPSolverConfig(
            backend="cvxpy",
            external_solver=solver,
            max_iter=5000,
            tol=1e-8,
        ),
    )
    assert result.success is True
    assert result.sdp_status in {"optimal", "optimal_inaccurate"}
    assert result.certificate.exact_residual_zero is True
    assert all(check is True for check in result.certificate.exact_psd_checks)
    assert result.certificate.verified is True


def test_critical_cone_subspace():
    x = sp.symbols("x", real=True)
    result = analyze_critical_cone(x**2, x**2, [x], {x: 0}, [], [], [])
    assert result.classification == "SecondOrderSufficientMinimum"
    assert result.negative_direction_exists is False


def test_compile_special_function_nodes():
    x = sp.symbols("x", positive=True)
    expressions = [
        sp.lowergamma(sp.Rational(3, 2), x),
        sp.uppergamma(sp.Rational(3, 2), x),
        sp.li(x),
        sp.chebyshevt(3, x),
        sp.legendre(4, x),
        sp.assoc_laguerre(3, 2, x),
    ]
    assert all(compile_arb(expr) is not None for expr in expressions)
