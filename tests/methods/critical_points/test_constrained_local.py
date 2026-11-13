import sympy as sp

import symbopt


def test_exact_kkt_refinement_equality_constraint():
    x, y = sp.symbols("x y", real=True)
    result = symbopt.minimize_local_constrained(
        x**2 + y**2,
        [x, y],
        [0.2, 0.8],
        equalities=[x + y - 1],
    )
    assert result.kkt_verified
    assert result.exact_kkt_refined
    assert result.point == {x: sp.Rational(1, 2), y: sp.Rational(1, 2)}
    assert result.equality_multipliers == (sp.Integer(-1),)


def test_exact_kkt_refinement_active_inequality():
    x = sp.symbols("x", real=True)
    result = symbopt.minimize_local_constrained(
        (x + 1) ** 2,
        [x],
        [0.5],
        inequalities=[x],
    )
    assert result.kkt_verified
    assert result.exact_kkt_refined
    assert result.point[x] == 0
    assert result.inequality_multipliers == (sp.Integer(2),)
