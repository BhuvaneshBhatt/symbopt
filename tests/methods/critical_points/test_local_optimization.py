import pytest
import sympy as sp

from symbopt import maximize_local, minimize_local


def test_local_quadratic_refinement():
    x, y = sp.symbols("x y", real=True)
    objective = (x - 2) ** 2 + 3 * (y + 1) ** 2

    result = minimize_local(objective, [x, y], [8.0, 5.0])

    assert result.success
    assert result.status == "exact_local_mode"
    assert result.point == {x: 2, y: -1}
    assert result.objective_value == 0
    assert result.gradient_at_point == (0, 0)
    assert result.curvature_matrix == sp.diag(2, 6)
    assert result.stationarity_verified
    assert result.curvature_verified
    assert result.exact_refined


def test_local_maximum_information():
    x = sp.symbols("x", real=True)
    log_posterior = -sp.Rational(1, 2) * (x - 3) ** 2

    result = maximize_local(log_posterior, [x], [0.0])

    assert result.success
    assert result.point == {x: 3}
    assert result.objective_hessian == sp.Matrix([[-1]])
    assert result.curvature_matrix == sp.Matrix([[1]])
    assert result.observed_information == sp.Matrix([[1]])
    assert result.curvature_verified


def test_local_boundary_refinement():
    x = sp.symbols("x", real=True)

    result = minimize_local((x - 4) ** 2, [x], [0.0], bounds={x: (0, 2)})

    assert result.success
    assert result.point == {x: 2}
    assert result.objective_value == 4
    assert result.curvature_matrix == sp.Matrix([[2]])
    assert result.exact_refined
    # A bound-active optimum need not have zero unconstrained gradient.
    assert not result.stationarity_verified


def test_nonconvex_local_mode():
    x = sp.symbols("x", real=True)

    result = minimize_local(-(x**2), [x], [0.0], method="BFGS")

    assert not result.success
    assert result.classification == "LocalMaximum"
    assert result.status == "stationary_or_degenerate"


def test_unresolved_parameters_are_rejected():
    x, scale = sp.symbols("x scale", real=True)
    with pytest.raises(ValueError, match="unresolved parameter"):
        minimize_local(scale * x**2, [x], [1.0])


def test_inexact_precision_preservation():
    x = sp.symbols("x", real=True)
    objective = (x - sp.Float("1.234567890123456789", 40)) ** 2

    with pytest.raises(ValueError, match="float64"):
        minimize_local(objective, [x], [0.0])


def test_public_result_has_probabilistic_mode_alias():
    x = sp.symbols("x", real=True)
    result = maximize_local(-((x - 1) ** 2), [x], [4.0])
    assert result.mode == result.point


def test_broader_exact_refinement_handles_rational_stationarity():
    x = sp.symbols("x", real=True)
    objective = (x - sp.sqrt(2)) ** 2 / (x**2 + 1)
    result = minimize_local(objective, [x], [1.4])
    assert result.success
    assert result.exact_refined
    assert sp.simplify(result.point[x] - sp.sqrt(2)) == 0
