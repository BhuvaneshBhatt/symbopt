import sympy as sp
import pytest
from symbopt import algebraize, algebraize_cases


def test_abs_and_sqrt_lifting():
    x = sp.symbols("x", real=True)
    p = algebraize(sp.Abs(x - 1) + sp.sqrt(x + 2), [sp.Ge(x, -2)])
    assert any(r.kind == "abs" for r in p.lift_records)
    assert any(r.kind == "root" for r in p.lift_records)


def test_inverse_lift_records_nonzero_domain():
    x = sp.symbols("x", real=True)
    p = algebraize(1 / (x + 1), [])
    assert any(r.kind == "inverse" for r in p.lift_records)
    assert any(isinstance(r, sp.Unequality) for r in p.graph_inequalities)


def test_repeated_root_and_inverse_are_cached():
    x = sp.symbols("x", real=True)
    p = algebraize(2 * sp.sqrt(x + 1) + 3 / (x + 2), [sp.Ge(x, -1)])
    kinds = [r.kind for r in p.lift_records]
    assert kinds.count("root") <= 1
    assert kinds.count("inverse") <= 1


def test_piecewise_splits_into_cases():
    x = sp.symbols("x", real=True)
    cases = algebraize_cases(sp.Piecewise((x**2, x >= 0), (-x, True)), [sp.Le(x, 2)])
    assert len(cases) == 2


def test_duplicate_constraints_removed():
    x = sp.symbols("x", real=True)
    p = algebraize(x**2, [sp.Ge(x, 0), sp.Ge(x, 0), sp.Eq(x - x, 0)])
    keys = [sp.srepr(r) for r in p.all_constraints()]
    assert len(keys) == len(set(keys))


def test_unsupported_function_raises():
    x = sp.symbols("x", real=True)
    with pytest.raises(NotImplementedError):
        algebraize(sp.gamma(x), [])


def test_strict_inequality_preserved_and_closure_available():
    x = sp.symbols("x", real=True)
    p = algebraize(x**2, [sp.Gt(x, 0)])
    assert p.has_strict_constraints()
    assert any(isinstance(r, sp.GreaterThan) for r in p.closure_constraints())


def test_piecewise_default_branch_preserved_regression():
    x = sp.symbols("x", real=True)
    cases = algebraize_cases(sp.Piecewise((1, x > 0), (2, True)), [])
    values = {sp.simplify(c.original_objective) for c in cases}
    assert values == {1, 2}


def test_rootof_real_branch_lifts_when_supported():
    z = sp.symbols("z")
    r = sp.CRootOf(z**2 - 2, 0)
    p = algebraize(r + 1, [])
    assert any(rec.kind == "rootof" for rec in p.lift_records)
