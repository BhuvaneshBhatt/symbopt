from __future__ import annotations

import sympy as sp

from symbopt import analyze_critical_cone


def test_equality_only_positive_reduced_hessian():
    x, y = sp.symbols("x y", real=True)
    lam = sp.symbols("lam", real=True)
    objective = x**2 + y**2
    equality = x
    lagrangian = objective + lam * equality
    result = analyze_critical_cone(
        objective, lagrangian, [x, y], {x: 0, y: 0, lam: 0}, [equality], [], []
    )
    assert result.classification == "SecondOrderSufficientMinimum"
    assert result.negative_direction_exists is False


def test_flat_tangent_classification():
    x, y = sp.symbols("x y", real=True)
    objective = x**2
    result = analyze_critical_cone(
        objective, objective, [x, y], {x: 0, y: 0}, [], [], []
    )
    assert result.classification in {
        "SecondOrderNecessaryMinimum",
        "FlatCriticalCone",
        "Inconclusive",
    }
    assert result.classification != "SecondOrderSufficientMinimum"


def test_indefinite_tangent_space_is_detected():
    x, y = sp.symbols("x y", real=True)
    objective = x**2 - y**2
    result = analyze_critical_cone(
        objective, objective, [x, y], {x: 0, y: 0}, [], [], []
    )
    assert result.classification == "IndefiniteCriticalCone"
    assert result.positive_direction_exists is True
    assert result.negative_direction_exists is True


def test_strongly_active_inequality_becomes_tangent_equality():
    x = sp.symbols("x", real=True)
    mu = sp.symbols("mu", real=True)
    objective = x
    inequality = x
    lagrangian = objective - mu * inequality
    result = analyze_critical_cone(
        objective, lagrangian, [x], {x: 0, mu: 1}, [], [inequality], [mu]
    )
    # The normalized critical cone is empty because d=0 and d^2=1 conflict.
    # The important contract is that this degenerate case is not called indefinite.
    assert result.classification != "IndefiniteCriticalCone"
