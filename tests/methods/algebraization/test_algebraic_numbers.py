import sympy as sp

from symbopt import algebraize
from symbopt.algebraic_numbers import (
    algebraic_compare,
    isolating_interval,
    exactify,
    is_fixed_real_algebraic,
)


def test_high_degree_fixed_rootof_uses_exact_isolating_interval():
    z = sp.symbols("z")
    root = sp.CRootOf(z**5 - z - 1, 0)
    interval = isolating_interval(root)
    assert interval is not None
    lo, hi = interval
    assert lo.is_Rational and hi.is_Rational and lo < hi

    problem = algebraize(root + 1, [])
    records = [record for record in problem.lift_records if record.kind == "rootof"]
    assert len(records) == 1
    record = records[0]
    assert record.metadata["exact_branch"] is True
    meta_lo, meta_hi = map(sp.Rational, record.metadata["isolating_interval"])
    assert meta_lo < meta_hi
    assert algebraic_compare(root, meta_lo) == 1
    assert algebraic_compare(root, meta_hi) == -1
    assert len(record.domain_rels) >= 2


def test_repeated_rootof_reuses_single_exact_branch_lift():
    z = sp.symbols("z")
    root = sp.CRootOf(z**5 - z - 1, 0)
    problem = algebraize(root**2 + 2 * root, [])
    records = [record for record in problem.lift_records if record.kind == "rootof"]
    assert len(records) == 1


def test_algebraic_number_input_is_preserved_as_exact_algebraic_value():
    value = sp.AlgebraicNumber(sp.sqrt(2))
    assert exactify(value) == sp.sqrt(2)
    assert is_fixed_real_algebraic(value)
    problem = algebraize(value + 1, [])
    assert not problem.lifted_objective.has(sp.Float)


def test_algebraic_compare_orders_rootof_values_exactly():
    z = sp.symbols("z")
    plastic = sp.CRootOf(z**5 - z - 1, 0)
    cube_root_two = sp.CRootOf(z**3 - 2, 0)
    assert algebraic_compare(plastic, cube_root_two) == -1
    assert algebraic_compare(cube_root_two, plastic) == 1
    assert algebraic_compare(plastic, plastic) == 0
