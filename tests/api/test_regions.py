"""Region membership, coordinate ownership and extrema agree with the source set."""

import pytest
import sympy as sp

from symbopt import (
    argmax,
    argmin,
    maximize,
    maxval,
    minimize,
    minimize_quadratic,
    minimize_result,
    minval,
    verify_quadratic_certificate,
)
from symbopt.region_inputs import normalize_region

x, y, t = sp.symbols("x y t", real=True)


@pytest.fixture
def geometry():
    return pytest.importorskip("semialg")


@pytest.mark.parametrize(
    "function,objective,expected",
    [
        (minimize, (x - 2) ** 2, [1, {x: 1}]),
        (maximize, x, [1, {x: 1}]),
        (minval, (x - 2) ** 2, 1),
        (maxval, x, 1),
        (argmin, (x - 2) ** 2, ({x: 1},)),
        (argmax, x, ({x: 1},)),
    ],
)
def test_global_region_wrappers(function, objective, expected):
    assert function(objective, variables=x, region=sp.Interval(0, 1)) == expected


def test_constraints_and_bounds_intersect():
    result = minimize_result(
        -x, x >= 0, region=sp.Interval(-2, 2), bounds={x: (None, 1)}, variables=x
    )
    assert result.certified and result.optimum_value == -1
    assert result.best_candidate.original_variable_values == {x: 1}


@pytest.mark.parametrize("region", [sp.false, sp.S.EmptySet])
def test_empty_region(region):
    result = minimize_result(x, region=region, variables=x)
    assert result.infeasible and result.certified and result.optimum_value == sp.oo
    assert argmin(x, region=region, variables=x) == ()


def test_full_region():
    assert minimize((x - 2) ** 2, region=sp.true, variables=x) == [0, {x: 2}]


@pytest.mark.parametrize(
    "region",
    [sp.Interval.open(0, 1), sp.Complement(sp.Interval(0, 1), sp.FiniteSet(0))],
)
def test_open_region_infimum(region):
    result = minimize_result(x, region=region, variables=x)
    assert result.certified and result.optimum_value == 0 and result.attained is False
    assert argmin(x, region=region, variables=x) == ()


def test_open_fractional_infimum():
    result = minimize_result(
        (x + 1) / (x + 2), region=sp.Interval.open(0, 1), variables=x
    )
    assert result.certified and result.optimum_value == sp.Rational(1, 2)
    assert result.attained is False


def test_open_constant_attainment():
    result = minimize_result(sp.Integer(3), region=sp.Interval.open(0, 1), variables=x)
    assert result.certified and result.optimum_value == 3 and result.attained is True
    point = result.best_candidate.original_variable_values
    assert 0 < point[x] < 1


def test_disconnected_region():
    region = sp.Union(sp.Interval(-2, -1), sp.Interval(1, 2))
    result = minimize_result(x * x, region=region, variables=x)
    assert result.certified and result.optimum_value == 1
    assert argmin(x * x, region=region, variables=x) == ({x: -1}, {x: 1})


def test_product_region():
    region = sp.ProductSet(sp.Interval(0, 1), sp.Interval(2, 3))
    assert minimize(x + y, region=region, variables=(x, y)) == [2, {x: 0, y: 2}]


def test_finite_region():
    region = sp.FiniteSet(sp.Tuple(1, 2), sp.Tuple(-2, 3))
    assert minimize(x * x + y * y, region=region, variables=(x, y)) == [5, {x: 1, y: 2}]


def test_unassumed_coordinates():
    coordinate = sp.Symbol("coordinate")
    assert minimize(coordinate**2, region=sp.S.Reals, variables=coordinate) == [
        0,
        {coordinate: 0},
    ]


def test_coordinate_validation():
    with pytest.raises(ValueError):
        minimize(x * x, region=sp.Interval(0, 1))
    with pytest.raises(ValueError):
        minimize(x + y, region=sp.Interval(0, 1), variables=(x, y))
    with pytest.raises(ValueError):
        normalize_region(sp.ProductSet(sp.S.Reals, sp.S.Reals), (x,))
    with pytest.raises(ValueError):
        normalize_region(sp.S.Reals, (sp.Symbol("z", imaginary=True),))
    with pytest.raises(ValueError):
        normalize_region(sp.FiniteSet(sp.Tuple(0, 1)), x)


def test_formula_complement():
    region = sp.Not(sp.And(x >= 0, x <= 1), evaluate=False)
    formula, coordinates = normalize_region(region, x)
    assert coordinates == [x]
    assert sp.simplify(formula.subs(x, -1)) is sp.true
    assert sp.simplify(formula.subs(x, sp.Rational(1, 2))) is sp.false
    assert sp.simplify(formula.subs(x, 1)) is sp.false


def test_quadratic_empty_region():
    result = minimize_quadratic(
        x * x, region=sp.S.EmptySet, variables=x, certified=True
    )
    assert result.status == "infeasible" and result.certified
    assert result.lower_bound == result.upper_bound == sp.oo


def test_quadratic_union_scope():
    region = sp.Union(sp.Interval(-2, -1), sp.Interval(1, 2))
    with pytest.raises(NotImplementedError):
        minimize_quadratic(x * x, region=region, variables=x)


def test_unresolved_set_is_not_empty():
    region = sp.ImageSet(sp.Lambda(t, sp.sin(t)), sp.S.Reals)
    with pytest.raises(NotImplementedError):
        normalize_region(region, x)


def test_ball_and_sphere(geometry):
    ball, sphere = geometry.Ball((0, 0), 1), geometry.Sphere((0, 0), 1)
    result = minimize_result(x + y, region=ball, variables=(x, y))
    assert result.certified and result.optimum_value == -sp.sqrt(2)
    assert result.best_candidate.original_variable_values == {
        x: -sp.sqrt(2) / 2,
        y: -sp.sqrt(2) / 2,
    }
    assert minval(x * x + y * y, region=ball, variables=(x, y)) == 0
    assert minval(x * x + y * y, region=sphere, variables=(x, y)) == 1


def test_point_region(geometry):
    assert minimize(x + y, region=geometry.Point((1, 2)), variables=(x, y)) == [
        3,
        {x: 1, y: 2},
    ]
    assert minimize(x + y, region=geometry.Ball((1, 2), 0), variables=(x, y)) == [
        3,
        {x: 1, y: 2},
    ]


def test_owned_coordinates(geometry):
    region = geometry.SemialgebraicRegion(
        sp.And(y >= 0, y <= 1, x >= 2, x <= 3), (y, x)
    )
    result = minimize_result(x + 10 * y, region=region)
    assert result.certified and result.optimum_value == 2
    assert result.best_candidate.original_variable_values == {x: 2, y: 0}
    assert result.lifted_problem.original_variables == [y, x]
    with pytest.raises(ValueError):
        minimize(x + 10 * y, region=region, variables=(x, y))


def test_region_parameter_stays_parameter(geometry):
    radius = sp.Symbol("radius", positive=True)
    region = geometry.SemialgebraicRegion(sp.And(x >= 0, x <= radius), (x,))
    result = minimize_result(-x, region=region)
    assert result.certified and result.optimum_value == -radius
    assert result.lifted_problem.original_variables == [x]
    assert result.lifted_problem.parameter_symbols == [radius]


def test_empty_owned_region(geometry):
    result = minimize_result(x, region=geometry.SemialgebraicRegion(False, (x,)))
    assert result.infeasible and result.certified and result.optimum_value == sp.oo


def test_cad_region(geometry):
    region = geometry.as_cad_region(sp.And(x >= 0, x <= 1), (x,))
    result = minimize_result(-x, region=region)
    assert result.certified and result.optimum_value == -1
    assert result.best_candidate.original_variable_values == {x: 1}


def test_open_geometry(geometry):
    result = minimize_result(
        x, region=geometry.Interval(0, 1, lower_closed=False), variables=x
    )
    assert result.certified and result.optimum_value == 0 and result.attained is False


def test_geometry_coordinate_count(geometry):
    region = geometry.Ball((0, 0), 1)
    with pytest.raises(ValueError):
        minimize(x * x, region=region)
    with pytest.raises(ValueError):
        minimize(x * x, region=region, variables=x)


def test_embedded_simplex(geometry):
    region = geometry.Simplex(((0, 0), (1, 1)))
    assert minimize(-x - y, region=region, variables=(x, y)) == [-2, {x: 1, y: 1}]


def test_parametric_image(geometry):
    region = geometry.ParametricRegion((t,), ((t, 0, 1),), (t, 2 * t))
    assert minimize(-x - y, region=region, variables=(x, y)) == [-3, {x: 1, y: 2}]


def test_curved_image_membership(geometry):
    region = geometry.ParametricRegion((t,), ((t, 0, 1),), (t, t * t))
    formula, coordinates = normalize_region(region, (x, y))
    assert coordinates == [x, y]
    for a, b in [(0, 0), (sp.Rational(1, 2), sp.Rational(1, 4)), (1, 1)]:
        assert sp.simplify(formula.subs({x: a, y: b}, simultaneous=True)) is sp.true
    for a, b in [(-1, 1), (sp.Rational(1, 2), sp.Rational(1, 2)), (2, 4)]:
        assert sp.simplify(formula.subs({x: a, y: b}, simultaneous=True)) is sp.false


def test_boolean_geometry(geometry):
    left = geometry.Interval(-2, -1)
    right = geometry.Interval(1, 2)
    region = geometry.BooleanRegion("union", (left, right))
    assert argmin(x * x, region=region, variables=x) == ({x: -1}, {x: 1})
    punctured = geometry.BooleanRegion(
        "difference", (geometry.Interval(0, 1), geometry.Point((0,)))
    )
    result = minimize_result(x, region=punctured, variables=x)
    assert result.certified and result.optimum_value == 0 and result.attained is False


def test_quadratic_region_replay(geometry):
    region = geometry.Box(((-1, 1), (-1, 1)))
    result = minimize_quadratic(
        x * x - y * y, region=region, variables=(x, y), backend="faces", certified=True
    )
    assert result.objective_value == result.lower_bound == result.upper_bound == -1
    assert verify_quadratic_certificate(
        x * x - y * y, result.certificate, region=region, variables=(x, y)
    )
    assert not verify_quadratic_certificate(
        x * x - y * y,
        result.certificate,
        region=geometry.Box(((-1, 1), (-2, 2))),
        variables=(x, y),
    )
