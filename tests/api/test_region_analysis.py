"""Prepared covers, exact loci, topology transitions and rigorous branch pruning."""

import pytest
import sympy as sp

import symbopt as so

x, y, t, a, value = sp.symbols("x y t a value", real=True)


@pytest.fixture
def geometry():
    return pytest.importorskip("semialg")


def truth(formula, point):
    return sp.simplify(formula.subs(point, simultaneous=True))


def test_prepared_core():
    region = so.prepare_region(sp.Interval.open(0, 1), x)
    assert region.outer_bounds == {x: (0, 1)}
    assert region.provenance["membership_ready"] is False
    assert so.minval(x, region=region) == 0
    assert so.maxval(x, region=region) == 1
    assert so.minval(x, region=region, variables=x) == 0
    assert region.variables == (x,)
    assert region.provenance["membership_ready"] is True
    assert so.prepare_region(region) is region
    with pytest.raises(ValueError):
        so.prepare_region(region, y)


def test_admission_budget():
    region = so.prepare_region(sp.Interval(0, 1), x, max_operations=0)
    with pytest.raises(so.RegionConversionError) as error:
        so.minval(x, region=region)
    assert error.value.stage == "admission"
    assert (
        so.minval(
            x, region=so.prepare_region(sp.Interval(0, 1), x, max_operations=None)
        )
        == 0
    )


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_budget_validation(limit):
    with pytest.raises(ValueError):
        so.prepare_region(sp.Interval(0, 1), x, max_quantifiers=limit)


def test_component_pruning():
    domain = sp.Or(sp.And(x >= 0, x <= 1), sp.And(x >= 10, x <= 11))
    result = so.minimize_result(x * x, domain, variables=x)
    assert result.certified and result.optimum_value == 0
    proof = next(p for p in result.certificate.proof_objects if p.kind == "piecewise")
    assert proof.payload["pruned_component_count"] == 1
    lower = proof.payload["branch_bound_proofs"][1].lower_bound
    assert 0 < lower <= 100
    assert truth(result.optimizer_locus, {x: 0}) is sp.true
    assert truth(result.optimizer_locus, {x: 10}) is sp.false


def test_component_ties():
    region = sp.Union(sp.Interval(-2, -1), sp.Interval(1, 2))
    assert so.argmin(x * x, region=region, variables=x) == ({x: -1}, {x: 1})


def test_unknown_locus():
    result = so.minimize_result(x * x, variables=x)
    result.certified = False
    assert result.optimizer_locus is None


def test_locus_extended_values():
    assert (
        so.minimize_result(x, region=sp.EmptySet, variables=x).optimizer_locus
        is sp.false
    )
    assert so.minimize_result(x, variables=x).optimizer_locus is sp.false
    assert (
        so.minimize_result(
            x, region=sp.Interval.open(0, 1), variables=x
        ).optimizer_locus
        is sp.false
    )


def test_prepared_geometry(geometry):
    region = so.prepare_region(geometry.Ball((1, 2), 3), (x, y))
    assert region.outer_bounds == {x: (-2, 4), y: (-1, 5)}
    assert so.minval(x + y, region=region) == 3 - 3 * sp.sqrt(2)
    assert region.provenance["membership_ready"] is True


def test_curve_pullback(geometry):
    source = geometry.ParametricRegion((t,), ((t, 0, 1),), (t, t * t))
    region = so.prepare_region(source, (x, y), max_quantifiers=0)
    result = so.minimize_result(y - x, region=region)
    assert result.certified and result.optimum_value == -sp.Rational(1, 4)
    assert result.best_candidate.original_variable_values == {
        x: sp.Rational(1, 2),
        y: sp.Rational(1, 4),
    }
    assert result.method == "region_cover"
    assert region.provenance["membership_ready"] is False
    assert so.maximize(y - x, region=region) == [0, ({x: 0, y: 0}, {x: 1, y: 1})]


def test_pullback_constraints(geometry):
    source = geometry.ParametricRegion((t,), ((t, 0, 1),), (t, t * t))
    result = so.minimize_result(
        y - x, x >= sp.Rational(3, 4), region=source, variables=(x, y)
    )
    assert result.optimum_value == -sp.Rational(3, 16) and result.certified
    assert result.best_candidate.original_variable_values == {
        x: sp.Rational(3, 4),
        y: sp.Rational(9, 16),
    }
    assert so.minval(
        y - x, region=source, variables=(x, y), bounds={x: (sp.Rational(3, 4), None)}
    ) == -sp.Rational(3, 16)


def test_singular_image(geometry):
    source = geometry.ParametricRegion((t,), ((t, -1, 1),), (t * t, t**3))
    assert so.minimize(x + y * y, region=source, variables=(x, y)) == [0, {x: 0, y: 0}]


def test_noninjective_image(geometry):
    source = geometry.ParametricRegion((t,), ((t, -1, 1),), (t * t, t**4))
    assert so.minimize(-x, region=source, variables=(x, y)) == [-1, {x: 1, y: 1}]


def test_parameter_capture(geometry):
    source = geometry.ParametricRegion((x,), ((x, 0, 1),), (x, x * x))
    assert so.minimize(y - x, region=source, variables=(x, y)) == [
        -sp.Rational(1, 4),
        {x: sp.Rational(1, 2), y: sp.Rational(1, 4)},
    ]


def test_transformed_image(geometry):
    source = geometry.TransformedRegion(geometry.Box(((0, 1),)), (t, t * t), (t,))
    assert so.minimize(y - x, region=source, variables=(x, y)) == [
        -sp.Rational(1, 4),
        {x: sp.Rational(1, 2), y: sp.Rational(1, 4)},
    ]


def test_union_point_stratum(geometry):
    line = geometry.Simplex(((0, 0), (1, 1)))
    point = geometry.Point((-2, 0))
    region = geometry.BooleanRegion("union", (line, point))
    assert so.minimize(x + y, region=region, variables=(x, y)) == [-2, {x: -2, y: 0}]


def test_vertex_route(geometry):
    source = geometry.Simplex(((0, 0), (2, 0), (0, 1)))
    result = so.minimize_result(-x - y, region=source, variables=(x, y))
    assert result.method == "region_vertices" and result.certified
    assert result.optimum_value == -2
    assert result.best_candidate.original_variable_values == {x: 2, y: 0}


def test_continuum_locus(geometry):
    region = geometry.Box(((-1, 1), (-2, 2)))
    formula = so.optimizer_locus(x * x, region=region, variables=(x, y))
    for ordinate in [-2, 0, 2]:
        assert truth(formula, {x: 0, y: ordinate}) is sp.true
    assert truth(formula, {x: 1, y: 0}) is sp.false
    assert truth(formula, {x: 0, y: 3}) is sp.false


def test_constant_locus(geometry):
    region = geometry.Sphere((0, 0), 1)
    formula = so.optimizer_locus(x * x + y * y, region=region, variables=(x, y))
    assert truth(formula, {x: 1, y: 0}) is sp.true
    assert truth(formula, {x: 0, y: 0}) is sp.false


def test_family_attainment(geometry):
    region = geometry.SemialgebraicRegion(sp.And(x > 0, x < a), (x,))
    family = so.optimization_family(
        x, region=region, parameters=(a,), value_symbol=value
    )
    assert family.quantifier_free
    for parameter in [-1, 0, 1, 2]:
        assert truth(family.feasible_condition, {a: parameter}) is sp.sympify(
            parameter > 0
        )
        assert truth(family.attained_condition, {a: parameter}) is sp.false
        assert truth(family.value_relation, {a: parameter, value: 0}) is sp.sympify(
            parameter > 0
        )
        assert truth(family.value_relation, {a: parameter, value: 1}) is sp.false
    assert truth(family.topology.empty_condition, {a: -1}) is sp.true


def test_family_closed_boundary(geometry):
    region = geometry.SemialgebraicRegion(sp.And(x >= 0, x <= a), (x,))
    family = so.optimization_family(
        -x, region=region, parameters=(a,), value_symbol=value
    )
    for parameter in [-1, 0, 1, 2]:
        feasible = sp.sympify(parameter >= 0)
        assert truth(family.feasible_condition, {a: parameter}) is feasible
        assert truth(family.attained_condition, {a: parameter}) is feasible
        assert (
            truth(family.value_relation, {a: parameter, value: -parameter}) is feasible
        )
        assert truth(family.optimizer_formula, {a: parameter, x: parameter}) is feasible


def test_family_unbounded(geometry):
    region = geometry.SemialgebraicRegion(x >= 0, (x,))
    family = so.optimization_family(
        a * x,
        region=region,
        parameters=(a,),
        value_symbol=value,
        analyze_topology=False,
    )
    for parameter in [-1, 0, 1]:
        assert truth(family.unbounded_condition, {a: parameter}) is sp.sympify(
            parameter < 0
        )
        assert truth(family.attained_condition, {a: parameter}) is sp.sympify(
            parameter >= 0
        )


def test_family_definitions(geometry):
    family = so.optimization_family(
        a * x,
        [x >= 0, x <= 1],
        variables=x,
        parameters=(a,),
        eliminate=False,
        analyze_topology=False,
    )
    assert family.quantifier_free is False
    assert family.value_relation.free_symbols == {a, family.value_symbol}
    assert family.optimizer_formula.free_symbols == {a, x}
    with pytest.raises(ValueError):
        so.optimization_family(a * x, variables=x, parameters=(x,))


def test_rational_chart_domain(geometry):
    source = geometry.ParametricRegion((t,), ((t, 0, 1),), (t, 1 / t))
    result = so.minimize_result(x, region=source, variables=(x, y))
    assert result.certified and result.optimum_value == 0 and result.attained is False
    assert result.optimizer_locus is sp.false


def test_cover_tied_charts(geometry):
    left = geometry.ParametricRegion((t,), ((t, 0, 1),), (t, t * t))
    right = geometry.ParametricRegion((t,), ((t, 0, 1),), (t, -t * t))
    source = geometry.BooleanRegion("union", (left, right))
    assert so.argmax(x * x + y * y, region=source, variables=(x, y)) == (
        {x: 1, y: -1},
        {x: 1, y: 1},
    )


def test_chart_integer_refusal(geometry):
    integer = sp.Symbol("integer", integer=True)
    source = geometry.ParametricRegion((t,), ((t, 0, 1),), (t,))
    with pytest.raises(NotImplementedError):
        so.minval(integer, region=source, variables=integer)


def test_preparation_immutable():
    from dataclasses import FrozenInstanceError

    region = so.prepare_region(sp.Interval(0, 1), x)
    with pytest.raises(FrozenInstanceError):
        region.variables = (y,)
    with pytest.raises(TypeError):
        region.outer_bounds[x] = (2, 3)
    assert so.minval(x, region=region) == 0


def test_quantified_locus(geometry):
    from semialg import SemialgebraicRegion

    formula = so.optimizer_locus((x - 1) ** 2, x >= 0, variables=x, eliminate=False)
    assert formula.free_symbols == {x}
    eliminated = SemialgebraicRegion(formula, (x,)).quantifier_free_formula()
    assert truth(eliminated, {x: 1}) is sp.true
    assert truth(eliminated, {x: 0}) is sp.false
    assert truth(eliminated, {x: -1}) is sp.false


def test_family_maximum(geometry):
    family = so.optimization_family(
        a * x,
        [x >= 0, x <= 1],
        variables=x,
        parameters=(a,),
        value_symbol=value,
        sense="maximize",
        analyze_topology=False,
    )
    for parameter in [-1, 0, 1]:
        optimum = max(0, parameter)
        assert truth(family.value_relation, {a: parameter, value: optimum}) is sp.true
        assert (
            truth(family.value_relation, {a: parameter, value: optimum + 1}) is sp.false
        )
    assert truth(family.optimizer_formula, {a: 0, x: sp.Rational(1, 2)}) is sp.true
    assert truth(family.optimizer_formula, {a: 1, x: 0}) is sp.false


def test_assumed_parameter(geometry):
    slope = sp.Symbol("slope", positive=True)
    family = so.optimization_family(
        slope * x, x >= 0, variables=x, parameters=(slope,), value_symbol=value
    )
    assert truth(family.value_relation, {slope: 2, value: 0}) is sp.true
    assert truth(family.value_relation, {slope: 2, value: -1}) is sp.false
    assert truth(family.unbounded_condition, {slope: 2}) is sp.false
    assert truth(family.optimizer_formula, {slope: 2, x: 0}) is sp.true
    assert truth(family.optimizer_formula, {slope: 2, x: 1}) is sp.false


def test_coordinate_coefficient_alias(geometry):
    source = geometry.ParametricRegion((t,), ((t, 0, 1),), (t + x, t * t))
    result = so.minimize_result(x, region=source, variables=(x, y))
    assert result.certified and result.optimum_value is -sp.oo and result.unbounded


def test_vertex_coefficient_alias(geometry):
    source = geometry.Simplex(((0, 0), (x, 1)))
    result = so.minimize_result(x, region=source, variables=(x, y))
    assert result.certified and result.optimum_value is -sp.oo and result.unbounded


def test_inexact_vertex_bound(geometry):
    coefficient = sp.Float("0.1", 80)
    source = geometry.Simplex(((sp.Rational(1, 3), 0), (1, 0)))
    result = so.minimize_result(
        coefficient * x,
        region=source,
        variables=(x, y),
        working_precision=80,
        max_active_sets=0,
        interval_max_boxes=1,
        use_sos=False,
        semialg_certification="candidate",
    )
    expected = sp.Rational(coefficient) / 3
    if result.certified:
        assert sp.Rational(result.optimum_value) == expected
    assert result.working_precision >= 80


def test_inexact_query_scope():
    with pytest.raises(so.RegionConversionError):
        so.optimizer_locus(sp.Float("0.1") * x, variables=x)
    with pytest.raises(so.RegionConversionError):
        so.optimization_family(sp.Float("0.1") * x + a, variables=x, parameters=(a,))
