import sympy as sp
from symbopt import (
    build_moment_matrix,
    build_localizing_matrix,
    build_moment_relaxation,
    extract_moment_atoms,
    solve_moment_dual,
    link_sos_moments,
)


def test_moment_matrix_rank1_data():
    x = sp.symbols("x", real=True)
    mm = build_moment_matrix([x], 1, {(0,): 1, (1,): 2, (2,): 4})
    assert mm.matrix.det() == 0


def test_localizing_matrix_builds():
    x = sp.symbols("x", real=True)
    lm = build_localizing_matrix(x, [x], 1, {(0,): 1, (1,): 2, (2,): 4, (3,): 8})
    assert lm.matrix.rows >= 1


def test_compile_moment_relaxation():
    x = sp.symbols("x", real=True)
    r = build_moment_relaxation(x**2, [x], [x], 1)
    assert r.moment_matrix.matrix.rows >= 1
    assert len(r.localizing_matrices) == 1


def test_atom_extraction_rank1():
    x = sp.symbols("x", real=True)
    mm = build_moment_matrix([x], 1, {(0,): 1, (1,): 2, (2,): 4})
    a = extract_moment_atoms(mm.matrix, mm.basis, [x])
    assert a.flat
    assert a.atoms == [{x: 2}]


def test_atom_extraction_multiatom_smoke():
    x = sp.symbols("x", real=True)
    mm = build_moment_matrix([x], 2, {(0,): 1, (1,): 1, (2,): 2, (3,): 4, (4,): 8})
    a = extract_moment_atoms(mm.matrix, mm.basis, [x])
    assert a.flat
    assert len(a.atoms) >= 1


def test_moments_dual_without_values_compiles_only():
    x = sp.symbols("x", real=True)
    r = solve_moment_dual(x**2, [], [x], 1)
    assert not r.flat_extension_detected


def test_moments_dual_with_values_extracts():
    x = sp.symbols("x", real=True)
    r = solve_moment_dual(x**2, [], [x], 1, moment_values={(0,): 1, (1,): 2, (2,): 4})
    assert r.flat_extension_detected
    assert r.candidate_points == [{x: 2}]


def test_moment_bridge_consistency_hint():
    x = sp.symbols("x", real=True)
    r = link_sos_moments(x**2, [], [x], [{x: 0}], order=1)
    assert len(r.candidate_hints) == 1
    assert r.candidate_hints[0].objective_value == 0


def test_moment_bridge_no_candidates_is_graceful():
    x = sp.symbols("x", real=True)
    r = link_sos_moments(x**2, [], [x], [], order=1)
    assert r.candidate_hints == []
    assert r.dual_result is None


def test_moment_bridge_candidate_objective_exact():
    x = sp.symbols("x", real=True)
    r = link_sos_moments((x - 3) ** 2, [], [x], [{x: 3}], order=1)
    assert r.candidate_hints[0].objective_value == 0


def test_atom_verification_checks_off_diagonal_moments():
    from symbopt.moment import verify_atom_measure

    x = sp.symbols("x", real=True)
    basis = [sp.Integer(1), x]
    matrix = sp.Matrix([[1, 0], [0, 1]])
    verified, residuals = verify_atom_measure(
        [{x: sp.Integer(0)}],
        [sp.Integer(1)],
        basis,
        matrix,
        [x],
    )
    assert verified is False
    assert any(value != 0 for value in residuals.values())


def test_moment_builders_accept_single_symbol_and_inequality():
    x = sp.symbols("x", real=True)
    matrix = build_moment_matrix(x, 2)
    assert matrix.matrix.rows == 3
    localizing = build_localizing_matrix(1 - x**2, x, 2)
    assert localizing.matrix.rows >= 1
    relaxation = build_moment_relaxation(x**2, 1 - x**2, x, 2)
    assert relaxation.variables == [x]
    assert len(relaxation.localizing_matrices) == 1
