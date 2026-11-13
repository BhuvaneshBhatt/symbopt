import pytest
import sympy as sp

from symbopt import argmin, maximize_result, minimize_result
from symbopt.convex_rules import convex_kkt_certificate, curvature
from symbopt.rule_utils import RuleContext

x, y, z = sp.symbols("x y z", real=True)


def check(objective, constraints, variables, value, *, attained=True, kind=None):
    result = minimize_result(objective, constraints, variables=variables, use_sos=False)
    assert result.certified and result.attained is attained
    assert (
        sp.simplify(result.optimum_value - value) == 0
        if value not in (-sp.oo, sp.oo)
        else result.optimum_value == value
    )
    if kind is not None:
        assert result.structural_result.kind == kind
    if attained:
        assert result.best_candidate is not None
        point = result.best_candidate.original_variable_values
        assert sp.simplify(objective.subs(point, simultaneous=True) - value) == 0
        assert all(
            sp.simplify(c.subs(point, simultaneous=True)) is sp.true
            for c in constraints
        )
    return result


def test_leading_degree():
    check(x**101 - 3 * x**20, [], x, -sp.oo, attained=False, kind="leading_ray")
    check(-(x**100), [], x, -sp.oo, attained=False, kind="leading_ray")
    check(-x * x + y**4, [y >= 0], (x, y), -sp.oo, attained=False, kind="leading_ray")


def test_ray_must_be_feasible():
    context = RuleContext(-(x**4), (x >= 0, x <= 1), (x,))
    from symbopt.elementary_rules import try_unbounded_ray

    assert try_unbounded_ray(context) is None


def test_visible_nonnegative_sum():
    check(5 + (x + y - 1) ** 2 + (x - y) ** 4, [], (x, y), 5, kind="nonnegative_sum")
    p = sp.Symbol("p", positive=True)
    c = sp.Symbol("c", real=True)
    check(
        c + p * (x + y - 1) ** 6 + (x - y) ** 4, [], (x, y), c, kind="nonnegative_sum"
    )


def test_lower_bound_needs_witness():
    from symbopt.elementary_rules import try_nonnegative_sum

    context = RuleContext(x * x + (x * y - 1) ** 2, (), (x, y))
    assert try_nonnegative_sum(context) is None


def test_affine_elimination():
    result = check(
        x**4 + y**4 + x * y, [sp.Eq(x + y, 2)], (x, y), 3, kind="affine_elimination"
    )
    assert result.best_candidate.primal_values == {x: 1, y: 1}


def test_parameter_rank_change():
    from symbopt.shape_reductions import affine_reduction

    a = sp.Symbol("a", real=True)
    assert (
        affine_reduction(RuleContext(x * x + y * y, (sp.Eq(a * x, 0),), (x, y))) is None
    )


def test_power_substitution():
    result = check(x**20 - 2 * x**10 + 3, [], x, 2, kind="power_substitution")
    assert {p[x] for p in result.structural_result.points} == {-1, 1}
    assert argmin(x**20 - 2 * x**10 + 3, variables=x, use_sos=False) == (
        {x: -1},
        {x: 1},
    )


def test_negative_power_branch():
    result = check(x**20 - 2 * x**10 + 3, [x < 0], x, 2, kind="power_substitution")
    assert result.best_candidate.primal_values[x] == -1


def test_affine_ridge():
    check((x + y) ** 4 - 2 * (x + y) ** 2 + 3, [], (x, y), 2, kind="affine_ridge")


def test_radial_reduction():
    result = check((x * x + y * y - 3) ** 2 + 2, [], (x, y), 2, kind="radial_reduction")
    assert (
        sp.simplify(
            sum(v * v for v in result.best_candidate.primal_values.values()) - 3
        )
        == 0
    )


def test_monotone_box():
    check(
        sp.exp(x) + y**3,
        [x > 0, x <= 1, y >= 1, y <= 2],
        (x, y),
        2,
        attained=False,
        kind="coordinate_monotone",
    )


def test_multiaffine_box():
    check(
        x * y + x - 2 * y,
        [x >= -1, x <= 2, y >= 1, y <= 3],
        (x, y),
        -10,
        kind="multiaffine_box",
    )


def test_fractional_box():
    objective = (x * y + x + 2) / (x + y + 5)
    constraints = [x >= 0, x <= 1, y >= 0, y <= 1]
    check(objective, constraints, (x, y), sp.Rational(1, 3), kind="multiaffine_box")


def test_poles_inside_box_refused():
    from symbopt.box_rules import try_monotone_box, try_multiaffine_box

    context = RuleContext(
        1 / (x + y), (x >= -1, x <= 1, y >= -1, y <= 1, sp.Ne(x + y, 0)), (x, y)
    )
    assert try_multiaffine_box(context) is None
    assert try_monotone_box(RuleContext(sp.tan(x), (x >= -2, x <= 2), (x,))) is None


def test_weighted_median():
    result = check(abs(x - 1) + abs(x - 3), [], x, 2, kind="weighted_median")
    assert result.structural_result.metadata["median_interval"] == (1, 3)


def test_linear_epigraph():
    check(
        abs(x + y) + abs(x - y - 2), [x >= 0, y >= 0], (x, y), 2, kind="linear_epigraph"
    )


def test_maximum_epigraph():
    check(sp.Max(x + y, 2 - x - y), [x >= 0, y >= 0], (x, y), 1, kind="linear_epigraph")


def test_exact_quadratic():
    result = check((x + 2 * y - 1) ** 2 + 3 * (x - y) ** 2, [], (x, y), 0)
    assert result.method == "qp"
    artifact = next(a for a in result.certificate.proof_objects if a.kind == "qp")
    assert artifact.payload["exact_postprocessing"]["method"] == "exact_linear_algebra"


def test_rayleigh():
    check(
        (2 * x * x + 2 * x * y + 2 * y * y) / (x * x + y * y),
        [],
        (x, y),
        1,
        kind="rayleigh_quotient",
    )
    check(
        2 * x * x + 2 * x * y + 2 * y * y,
        [sp.Eq(x * x + y * y, 1)],
        (x, y),
        1,
        kind="rayleigh_quotient",
    )


def test_linear_ellipsoid():
    check(3 * x + 4 * y, [x * x + y * y <= 1], (x, y), -5, kind="linear_ellipsoid")
    check(
        3 * x + 4 * y,
        [x * x + y * y < 1],
        (x, y),
        -5,
        attained=False,
        kind="linear_ellipsoid",
    )


def test_weighted_amgm():
    a, b = sp.symbols("a b", positive=True)
    check(a * x + b / x, [x > 0], x, 2 * sp.sqrt(a * b), kind="weighted_amgm")
    check(
        x * x + 1 / x**4, [x > 0], x, 3 / 2 ** sp.Rational(2, 3), kind="weighted_amgm"
    )


def test_cauchy_schwarz():
    check(
        1 / x + 4 / y, [x > 0, y > 0, sp.Eq(x + y, 3)], (x, y), 3, kind="cauchy_schwarz"
    )


def test_convex_stationary():
    check(sp.exp(x) + sp.exp(-x), [], x, 2, kind="convex_kkt")


def test_complex_composition_refused():
    objective = sp.exp(sp.I * x) + sp.exp(-sp.I * x)
    assert curvature(objective, RuleContext(objective, (), (x,))) is None
    assert convex_kkt_certificate(RuleContext(objective, (), (x,)), {x: 0}) is None


def test_concave_minimum():
    check(-sp.exp(x), [x >= 0, x <= 1], x, -sp.E)


def test_coupled_constraints_prevent_corner_rule():
    from symbopt.box_rules import try_multiaffine_box

    context = RuleContext(
        x * y, (x >= 0, x <= 1, y >= 0, y <= 1, sp.Eq(x + y, 1)), (x, y)
    )
    assert try_multiaffine_box(context) is None


def test_disabled_rules():
    result = minimize_result(
        x**6 - 2 * x * x + 3,
        variables=x,
        use_sos=False,
        options={"structural": {"enabled": False}},
    )
    assert result.certified
    with pytest.raises(TypeError, match="structural enabled"):
        minimize_result(x * x, options={"structural": {"enabled": 1}})


def test_maximize_mapping():
    result = maximize_result(
        -((x * x + y * y - 3) ** 2) + 2, variables=(x, y), use_sos=False
    )
    assert result.certified and result.attained and result.optimum_value == 2


def test_nonnegative_products():
    check(x**4 + y**4 + x * x * y * y, [], (x, y), 0, kind="nonnegative_sum")


def test_power_odd_inverse():
    check(x**6 - 2 * x**3 + 3, [], x, 2, kind="power_substitution")


def test_ridge_box_image():
    result = check(
        (x + y - 3) ** 4 + (x + y - 3) ** 2, [x >= 0, x <= 2, y >= 0, y <= 2], (x, y), 0
    )
    assert sp.simplify(sum(result.best_candidate.primal_values.values()) - 3) == 0


def test_radial_rational_infimum():
    result = minimize_result(1 / (1 + x * x + y * y), variables=(x, y), use_sos=False)
    assert result.certified and result.optimum_value == 0 and result.attained is False


def test_concave_endpoints():
    check(-x * x, [x >= -1, x <= 1], x, -1, kind="concave_vertices")


def test_generalized_rayleigh():
    result = check(
        (2 * x * x + 8 * y * y) / (x * x + 2 * y * y),
        [],
        (x, y),
        2,
        kind="rayleigh_quotient",
    )
    assert result.best_candidate.primal_values[y] == 0


def test_ellipsoid_parameters():
    radius = sp.Symbol("radius", positive=True)
    check(
        3 * x + 4 * y,
        [x * x + y * y <= radius**2],
        (x, y),
        -5 * radius,
        kind="linear_ellipsoid",
    )


def test_convex_kkt_sufficiency():
    context = RuleContext(sp.exp(x), (x >= 0,), (x,))
    certificate = convex_kkt_certificate(context, {x: 0})
    assert certificate.value == 1 and certificate.points == ({x: 0},)
    assert set(certificate.metadata["multipliers"].values()) == {1}


def test_nonconvex_kkt_refused():
    context = RuleContext(-(x**4), (), (x,))
    assert convex_kkt_certificate(context, {x: 0}) is None


def test_zero_witness_respects_hole():
    objective = sp.Mul(x * x, sp.Pow(x, -1, evaluate=False), evaluate=False)
    result = minimize_result(objective, [x >= 0], variables=x, use_sos=False)
    assert result.certified and result.optimum_value == 0 and result.attained is False


def test_open_box_constant_attainment():
    from symbopt.box_rules import try_multiaffine_box

    result = try_multiaffine_box(RuleContext(sp.S(3), (x > 0, x < 1), (x,)))
    assert (
        result.value == 3
        and result.attained is True
        and result.points == ({x: sp.Rational(1, 2)},)
    )


def test_unknown_sign_does_not_choose_corner():
    from symbopt.box_rules import try_multiaffine_box

    a = sp.Symbol("a", real=True)
    assert try_multiaffine_box(RuleContext(a * x, (x >= 0, x <= 1), (x,))) is None


def test_large_sparse_power():
    result = check(x**2000 - 2 * x**1000 + 3, [], x, 2, kind="power_substitution")
    assert {p[x] for p in result.structural_result.points} == {-1, 1}


def test_large_even_power_zero():
    result = check((x + y - 1) ** 2000, [], (x, y), 0, kind="nonnegative_sum")
    assert sp.simplify(sum(result.best_candidate.primal_values.values()) - 1) == 0


def test_floating_kkt_not_promoted():
    objective = (3 * x - sp.Float("0.1")) ** 2
    assert (
        convex_kkt_certificate(
            RuleContext(objective, (), (x,)), {x: sp.Float("0.1") / 3}
        )
        is None
    )


def test_power_sign_intersection():
    from symbopt.rule_utils import RuleContext
    from symbopt.shape_reductions import power_reduction

    objective = x**4 - 2 * x**2 + 3
    check(objective, [x >= 0, x <= 0], x, 3)
    check(objective, [x < 0], x, 2)
    context = RuleContext(objective, (x > 0, x < 0), (x,))
    assert power_reduction(context) is None


def test_rule_cache_isolation():
    from symbopt.symbolic_cache import cache_info, clear_symbolic_caches

    clear_symbolic_caches()
    objective = abs(x - 1) + 2 * abs(x - 2) + abs(x - 5)
    first = minimize_result(objective, variables=x)
    first.structural_result.metadata["median_interval"] = (-99, 99)
    first.structural_result.points[0][x] = -99
    second = minimize_result(objective, variables=x)
    assert second.optimum_value == 4
    assert second.structural_result.metadata["median_interval"] == (2, 2)
    assert second.structural_result.points == ({x: 2},)
    assert cache_info()["structural_recognition"].hits >= 1
    clear_symbolic_caches()
    assert cache_info()["structural_recognition"].currsize == 0


def test_scalar_quadratic_requires_real_coefficients():
    from symbopt.quadratic_optimization import _scalar_quadratic

    assert _scalar_quadratic(sp.Poly(x * x + sp.I, x), (x >= 0,), x) is None


def test_reduction_cache_isolation():
    objective = x**4 + y**4 + x * y
    first = minimize_result(objective, sp.Eq(x + y, 2), variables=(x, y), use_sos=False)
    first.structural_result.metadata["parameterization"][x] = -99
    second = minimize_result(
        objective, sp.Eq(x + y, 2), variables=(x, y), use_sos=False
    )
    assert second.optimum_value == 3
    assert second.structural_result.metadata["parameterization"][x] != -99
    assert second.best_candidate.primal_values == {x: 1, y: 1}


def test_nonsmooth_kkt_requires_subgradients():
    objective = sp.Max(2 * x, 3 * x, 4 * x) - sp.Rational(9, 2) * x
    # Default tie derivatives give zero here, although the function is unbounded below.
    assert convex_kkt_certificate(RuleContext(objective, (), (x,)), {x: 0}) is None
    certificate = convex_kkt_certificate(
        RuleContext(objective, (x <= -1,), (x,)), {x: -1}
    )
    assert certificate.value == sp.Rational(5, 2)
    assert certificate.points == ({x: -1},)


def test_absolute_value_constraints_define_exact_box():
    from symbopt.rule_utils import RuleContext

    x, y = sp.symbols("x y", real=True)
    context = RuleContext(
        x**4 - y**4,
        (sp.Le(sp.Abs(x), 2), sp.Lt(sp.Abs(y), 1)),
        (x, y),
    )
    rectangle = context.rectangle
    assert rectangle[x].left == -2
    assert rectangle[x].right == 2
    assert not rectangle[x].left_open and not rectangle[x].right_open
    assert rectangle[y].left == -1
    assert rectangle[y].right == 1
    assert rectangle[y].left_open and rectangle[y].right_open

    assert RuleContext(x**2, (sp.Ge(sp.Abs(x), 2),), (x,)).rectangle is None
    complex_symbol = sp.Symbol("z")
    assert (
        RuleContext(
            complex_symbol**4,
            (sp.Le(sp.Abs(complex_symbol), 1),),
            (complex_symbol,),
        ).rectangle
        is None
    )
    assert (
        RuleContext(
            x**4 - y**4,
            (sp.Le(sp.Abs(x), 2), sp.Eq(x + y, 1)),
            (x, y),
        ).rectangle
        is None
    )
