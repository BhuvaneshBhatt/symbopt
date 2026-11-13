from itertools import product
from math import comb

import pytest
import sympy as sp

from symbopt import project_affine_exact, refine_exact_system
from symbopt.monomials import exponent_tuples
from symbopt.symbolic_cache import affine_coefficients

x, y, a = sp.symbols("x y a", real=True)


@pytest.mark.parametrize(
    "size,degree", [(0, 4), (1, 8), (2, 4), (4, 4), (5, 4), (8, 4), (12, 2), (16, 1)]
)
def test_exponent_order(size, degree):
    expected = tuple(
        sorted(
            (m for m in product(range(degree + 1), repeat=size) if sum(m) <= degree),
            key=lambda m: (sum(m), m),
        )
    )
    actual = exponent_tuples(size, degree)
    assert actual == expected
    assert len(actual) == comb(size + degree, degree)


@pytest.mark.parametrize(
    "expression,expected",
    [
        (3 * x + 2 * y + 1, ((3, 2), 1)),
        (sp.sqrt(2) * x + a * y + a, ((sp.sqrt(2), a), a)),
        ((sp.sin(a) ** 2 + sp.cos(a) ** 2) * x, ((1, 0), 0)),
        (sp.S.Zero, ((0, 0), 0)),
        (x * y, None),
        (sp.sin(x), None),
    ],
)
def test_affine_coefficients(expression, expected):
    assert affine_coefficients(expression, (x, y)) == expected


def test_affine_projection():
    expected = {x: sp.Rational(1, 5), y: sp.Rational(3, 5)}
    assert project_affine_exact([2 * x + y - 1], (x, y), (1, 1)) == expected
    assert (
        project_affine_exact([2 * x + y - 1, 4 * x + 2 * y - 2], (x, y), (1, 1))
        == expected
    )
    assert (
        project_affine_exact([2 * x + y - 1, 4 * x + 2 * y - 3], (x, y), (1, 1)) is None
    )
    assert project_affine_exact([x * y - 1], (x, y), (1, 1)) is None


def test_refined_affine_point():
    result = refine_exact_system([x + y - 1], (x, y), [0.55, 0.44])
    assert result.verified
    assert result.point == {x: sp.Rational(111, 200), y: sp.Rational(89, 200)}
