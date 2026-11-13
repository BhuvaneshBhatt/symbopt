"""Exact optimization results must survive scale changes and coordinate renaming."""

import pytest
import sympy as sp

from symbopt import algebraize, build_kkt_system, critical_points, minimize_result
from symbopt.moment import verify_atom_measure

x = sp.Symbol("x", real=True)


@pytest.mark.parametrize("offset", [sp.Integer(10) ** 30, sp.Rational(1, 10**400)])
def test_piecewise_exact_order(offset):
    step = sp.Integer(1) if offset > 1 else offset / 2
    objective = sp.Piecewise((offset, x < 0), (offset - step, True))
    result = minimize_result(objective, variables=x, use_sos=False)
    assert result.certified
    assert result.optimum_value == offset - step
    assert result.attained is True


@pytest.mark.parametrize("name,relation", [("lam_1", "eq"), ("mu_1", "ge")])
def test_kkt_coordinate_names(name, relation):
    variable = sp.Symbol(name, real=True)
    constraint = sp.Eq(variable, 1) if relation == "eq" else variable >= 1
    problem = algebraize(variable**2, [constraint], variables=[variable])
    system = build_kkt_system(problem)
    multipliers = (*system.equality_multipliers, *system.inequality_multipliers)
    assert not set(system.primal_variables) & set(multipliers)
    multiplier = multipliers[0]
    expected = (
        2 * variable + multiplier if relation == "eq" else 2 * variable - multiplier
    )
    assert sp.expand(system.stationarity_equations[0].lhs - expected) == 0


def test_auxiliary_names():
    variable = sp.Symbol("aux_1", real=True)
    problem = algebraize(sp.sqrt(variable), [], variables=[variable])
    assert variable not in problem.auxiliary_variables
    root = problem.substitution_map[sp.sqrt(variable)]
    assert any(
        sp.expand((equation.lhs - equation.rhs).subs(variable, root**2)) == 0
        for equation in problem.defining_equalities
    )


@pytest.mark.parametrize(
    "matrix,weights,atoms",
    [
        (sp.Matrix([[1, 0], [100, 0]]), [1], [{x: 0}]),
        (sp.zeros(0), [1], [{x: 0}]),
        (sp.Matrix([[1]]), [1], [{x: 0}]),
        (sp.Matrix([[1, 2], [2, 2]]), [-1, 2], [{x: 0}, {x: 1}]),
        (sp.Matrix([[sp.Float("1.000000000000001"), 0], [0, 0]]), [1], [{x: 0}]),
        (sp.Matrix([[1, sp.I], [sp.I, -1]]), [1], [{x: sp.I}]),
    ],
)
def test_moment_rejections(matrix, weights, atoms):
    verified, _ = verify_atom_measure(atoms, weights, [sp.S.One, x], matrix, [x])
    assert verified is False


def test_positive_atomic_measure():
    verified, residuals = verify_atom_measure(
        [{x: 0}, {x: 1}],
        [sp.Rational(1, 3), sp.Rational(2, 3)],
        [sp.S.One, x],
        sp.Matrix([[1, sp.Rational(2, 3)], [sp.Rational(2, 3), sp.Rational(2, 3)]]),
        [x],
    )
    assert verified is True
    assert set(residuals.values()) == {sp.S.Zero}


def test_constrained_coordinate_names():
    variable = sp.Symbol("lam_1", real=True)
    results = critical_points((variable**2, [sp.Eq(variable, 1)]), [variable])
    assert [result.solution[variable] for result in results] == [sp.S.One]


def test_multivariate_flatness():
    from symbopt import build_moment_matrix, extract_moment_atoms
    from symbopt.monomials import exponent_tuples

    y = sp.Symbol("y", real=True)
    atoms = [(0, 0), (0, 1), (1, 2), (1, 3)]
    moments = {
        powers: sum(
            sp.Rational(1, 4) * a ** powers[0] * b ** powers[1] for a, b in atoms
        )
        for powers in exponent_tuples(2, 4)
    }
    data = build_moment_matrix((x, y), 2, moments)
    result = extract_moment_atoms(data.matrix, data.basis, (x, y))
    assert data.matrix.rank() == 4
    assert data.matrix[:3, :3].rank() == 3
    assert result.rank_m == 4 and result.rank_m_minus_1 == 3
    assert result.flat is False and result.verified is False


def test_fractional_small_coefficients():
    y = sp.Symbol("y", real=True)
    result = minimize_result(
        x / (y + 1), [x / 10**12 >= 0, y >= 0, y <= 1], variables=(x, y), use_sos=False
    )
    assert result.certified is True
    assert result.optimum_value == 0
    assert result.attained is True and result.unbounded is False


def test_fractional_empty_open_domain():
    from symbopt.linear_fractional import try_linear_fractional

    y = sp.Symbol("y", real=True)
    result = try_linear_fractional(y / (x + 1), [x > 0, x < 0], (x, y))
    assert result.certified and result.value is sp.oo
    assert result.points == () and result.attained is False


def test_fractional_unresolved_sign_branch():
    from symbopt.linear_fractional import try_linear_fractional

    assert (
        try_linear_fractional(1 / (sp.sqrt(2) * x + 1), [x >= -2, x <= 0], (x,)) is None
    )
