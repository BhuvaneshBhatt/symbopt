import sympy as sp
from symbopt import clear_symbolic_caches, cache_info, algebraize
from symbopt.polynomial_structure import build_sparsity_graph, build_symmetry_report
from symbopt.sos_program import monomial_basis, sparse_monomial_basis


def test_correlative_components():
    x, y, z = sp.symbols("x y z")
    c = build_sparsity_graph([x * y, z**2], [x, y, z]).connected_components()
    assert any(set(v) == {x, y} for v in c)


def test_even_symmetry_detected():
    x, y = sp.symbols("x y", real=True)
    r = build_symmetry_report(x**2 + y**2, [], [x, y])
    assert set(r.even_variables) == {x, y}


def test_permutation_symmetry_detected():
    x, y = sp.symbols("x y")
    r = build_symmetry_report(x**2 + y**2, [], [x, y])
    assert (x, y) in r.permutation_pairs or (y, x) in r.permutation_pairs


def test_monomial_basis_cache_hit():
    x, y = sp.symbols("x y")
    clear_symbolic_caches()
    monomial_basis([x, y], 4)
    a = cache_info()["monomial_basis"]
    monomial_basis([x, y], 4)
    b = cache_info()["monomial_basis"]
    assert b.hits > a.hits


def test_sparse_basis_not_larger():
    x, y, z = sp.symbols("x y z")
    assert len(sparse_monomial_basis([x**4 + y**4 + z**2], [x, y, z], 2)) <= len(
        monomial_basis([x, y, z], 2)
    )


def test_reduction_note_on_duplicates():
    x = sp.symbols("x", real=True)
    p = algebraize(x**2, [sp.Ge(x, 0), sp.Ge(x, 0)])
    assert any("reduction" in n for n in p.preprocessing_notes)


def test_sparsity_graph_accepts_single_expression_and_symbol():
    x, y = sp.symbols("x y")
    graph = build_sparsity_graph(x * y + x**2, [x, y])
    assert graph.edges == {(x, y)}
    one_var = build_sparsity_graph(x**2 + 1, x)
    assert one_var.variables == [x]


def test_symmetry_report_accepts_unconstrained_shorthand_and_single_constraint():
    x, y = sp.symbols("x y", real=True)
    report = build_symmetry_report(x**2 + y**2, [x, y])
    assert set(report.even_variables) == {x, y}
    constrained = build_symmetry_report(x**2 + y**2, sp.Ge(x, 0), [x, y])
    assert y in constrained.even_variables


def test_infer_box_bounds_accepts_single_symbol_and_relation():
    from symbopt import infer_box_bounds

    x = sp.symbols("x", real=True)
    assert infer_box_bounds(x, sp.Eq(x, 2)) == {x: (2.0, 2.0)}
