"""Structural reductions and candidate slices preserve their mathematical data."""

import pytest
import sympy as sp

from symbopt import algebraize
from symbopt.constraint_reduction import reduce_lifted_problem, reduce_relations
from symbopt.intervals import _precondition_forms
from symbopt.quadratic_candidates import _equality_points

x, y = sp.symbols("x y", real=True)


def test_relation_reduction_equivalence():
    relations = [sp.Eq(2 * x, 2), sp.Eq(x, 1), sp.Ge(y * y, 0, evaluate=False)] * 3
    reduced, duplicates, trivial = reduce_relations(relations)
    assert reduced == [sp.Eq(x, 1)]
    assert duplicates == 5 and trivial == 3
    for point in [-2, 0, 1, sp.Rational(3, 2)]:
        assert sp.simplify(sp.And(*relations).subs(x, point)) == sp.And(*reduced).subs(
            x, point
        )


def test_contradiction_is_retained():
    contradiction = sp.Eq(0, 1, evaluate=False)
    reduced, duplicates, trivial = reduce_relations([contradiction] * 2)
    assert reduced == [contradiction] and duplicates == 1 and trivial == 0
    assert sp.simplify(sp.And(*reduced)) is sp.false


def test_lifted_relations_keep_domains():
    problem = algebraize(sp.sqrt(x) + y, [x >= 0, sp.Eq(y, x * x)], variables=(x, y))
    original = sp.And(*problem.all_constraints())
    expected_vars = set(problem.all_variables)
    problem.graph_equalities += problem.defining_equalities
    problem.lifted_constraints *= 3
    result, report = reduce_lifted_problem(problem)
    assert report.duplicate_rels > 0
    assert set(result.all_variables) == expected_vars
    assert sp.And(*result.all_constraints()) == original
    aux = result.auxiliary_variables[0]
    assert sp.Ge(aux, 0) in result.all_constraints()
    assert sp.Eq(aux**2 - x, 0) in result.all_constraints()


@pytest.mark.parametrize(
    "equation,expected",
    [
        (x * x - 2, {(-sp.sqrt(2),), (sp.sqrt(2),)}),
        ((x - 1) ** 2, {(sp.S.One,)}),
        (2 * x - 1, {(sp.Rational(1, 2),)}),
        (x * x + 1, set()),
        (sp.S.One, set()),
        (sp.S.Zero, set()),
    ],
)
def test_coordinate_slice_roots(equation, expected):
    roots = tuple(_equality_points((x,), (equation,), ((sp.S.Zero,),)))
    assert set(roots) == expected
    assert len(roots) == len(expected)
    assert all(sp.simplify(equation.subs(x, point[0])) == 0 for point in roots)


def test_preconditioned_forms_agree():
    expression = (x * x - 1) / (x - 1) + sp.sin(x) ** 2 + sp.cos(x) ** 2
    forms = _precondition_forms(expression)
    assert forms[0] == expression
    assert len(forms) == len(set(forms))
    for point in [-2, 0, sp.Rational(1, 2), 2]:
        expected = sp.simplify(expression.subs(x, point))
        assert all(sp.simplify(form.subs(x, point) - expected) == 0 for form in forms)


def test_repeated_slice_seeds():
    points = tuple(
        _equality_points(
            (x,), (x * x - 2,), ((sp.S.Zero,), (sp.S.One,), (sp.Integer(2),))
        )
    )
    assert set(points) == {(-sp.sqrt(2),), (sp.sqrt(2),)}
    assert len(points) == 2
