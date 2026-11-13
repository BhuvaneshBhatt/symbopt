import pytest
import sympy as sp

from symbopt import build_moment_matrix, extract_moment_atoms
from symbopt.monomials import exponent_tuples

x, y = sp.symbols("x y", real=True)


def moment_data(points, weights, order):
    values = {
        powers: sum(
            weight * point[0] ** powers[0] * point[1] ** powers[1]
            for point, weight in zip(points, weights)
        )
        for powers in exponent_tuples(2, 2 * order)
    }
    return build_moment_matrix((x, y), order, values)


def recovered_measure(result):
    return {
        tuple(sp.simplify(atom[v]) for v in (x, y)): weight
        for atom, weight in zip(result.atoms, result.weights)
    }


@pytest.mark.parametrize(
    "points,weights,order",
    [
        ([(0, 0), (1, 0)], [sp.Rational(1, 3), sp.Rational(2, 3)], 2),
        (
            [(0, 0), (0, 1), (0, 3)],
            [sp.Rational(1, 5), sp.Rational(3, 10), sp.Rational(1, 2)],
            3,
        ),
        ([(0, 0), (0, 1), (1, 0), (1, 1)], [sp.Rational(1, 4)] * 4, 3),
        (
            [(-2, 3), (0, -1), (sp.Rational(1, 3), 2)],
            [sp.Rational(1, 5), sp.Rational(3, 10), sp.Rational(1, 2)],
            2,
        ),
        (
            [(sp.sqrt(2), 0), (-sp.sqrt(2), 1)],
            [sp.Rational(1, 3), sp.Rational(2, 3)],
            2,
        ),
    ],
)
def test_joint_measure(points, weights, order):
    data = moment_data(points, weights, order)
    result = extract_moment_atoms(data.matrix, data.basis, (x, y))
    assert result.flat and result.verified
    assert recovered_measure(result) == dict(zip(map(tuple, points), weights))
    assert result.rank_m == result.rank_m_minus_1 == len(points)
    assert set(result.residuals.values()) == {sp.S.Zero}


def test_reordered_scaled_basis():
    points, weights = [(0, 0), (1, 0)], [sp.Rational(1, 3), sp.Rational(2, 3)]
    data = moment_data(points, weights, 2)
    order = tuple(reversed(range(len(data.basis))))
    scaling = sp.diag(*range(1, len(order) + 1))
    basis = [scaling[i, i] * data.basis[j] for i, j in enumerate(order)]
    matrix = scaling * data.matrix.extract(order, order) * scaling
    result = extract_moment_atoms(matrix, basis, (x, y))
    assert result.verified
    assert recovered_measure(result) == dict(zip(points, weights))


def test_separator_budget():
    data = moment_data([(0, 0), (0, 1), (1, 0), (1, 1)], [sp.Rational(1, 4)] * 4, 3)
    limited = extract_moment_atoms(
        data.matrix, data.basis, (x, y), max_separating_attempts=1
    )
    assert limited.flat and limited.verified is False
    assert limited.atoms == []
    complete = extract_moment_atoms(
        data.matrix, data.basis, (x, y), max_separating_attempts=2
    )
    assert complete.verified and len(complete.atoms) == 4


def test_nonpositive_data():
    data = moment_data([(1, 2)], [-1], 2)
    result = extract_moment_atoms(data.matrix, data.basis, (x, y))
    assert result.flat and result.verified is False


def test_budget_validation():
    data = moment_data([(1, 2)], [1], 2)
    with pytest.raises(ValueError, match="positive integer"):
        extract_moment_atoms(data.matrix, data.basis, (x, y), max_separating_attempts=0)


def test_three_coordinates():
    z = sp.Symbol("z", real=True)
    variables = (z, x, y)
    points = [(0, 1, 0), (0, 1, 2), (1, 1, 2)]
    weights = [sp.Rational(1, 7), sp.Rational(2, 7), sp.Rational(4, 7)]
    values = {
        powers: sum(
            weight * sp.prod(v**d for v, d in zip(point, powers))
            for point, weight in zip(points, weights)
        )
        for powers in exponent_tuples(3, 4)
    }
    data = build_moment_matrix(variables, 2, values)
    result = extract_moment_atoms(data.matrix, data.basis, variables)
    assert result.verified and result.rank_m == 3
    actual = {
        tuple(atom[v] for v in variables): weight
        for atom, weight in zip(result.atoms, result.weights)
    }
    assert actual == dict(zip(points, weights))
