import sympy as sp

from symbopt import sos_decompose
from symbopt.sos_program import compile_sos_relaxation


def test_disconnected_sos_structure_reaches_sdp_as_blocks():
    x, y = sp.symbols("x y", real=True)
    compilation = compile_sos_relaxation(x**2 + y**2, [x, y], 1)
    assert compilation.main_gram_count == 2
    assert compilation.sdp_problem.num_blocks == 2
    assert {frozenset(block) for block in compilation.variable_blocks} == {
        frozenset([x]),
        frozenset([y]),
    }


def test_connected_sos_structure_remains_one_block():
    x, y = sp.symbols("x y", real=True)
    compilation = compile_sos_relaxation((x + y) ** 2, [x, y], 1)
    assert compilation.main_gram_count == 1
    assert compilation.sdp_problem.num_blocks == 1


def test_sparse_multiblock_exact_certificate_recovery():
    x, y = sp.symbols("x y", real=True)
    result = sos_decompose(x**2 + y**2, [x, y])
    assert result.success
    assert result.certificate.exact_residual_zero
    assert result.certificate.exact_psd_checks == [True, True]


def test_exact_recovery_handles_large_rational_denominators():
    x, y = sp.symbols("x y", real=True)
    polynomial = (x + sp.Rational(1, 997) * y) ** 2 + (y - sp.Rational(1, 991) * x) ** 2
    result = sos_decompose(polynomial, [x, y])
    assert result.success
    assert result.certificate.exact_residual_zero
    assert all(check is True for check in result.certificate.exact_psd_checks)


def test_overlapping_chordal_cliques_reach_sdp_as_psd_blocks():
    x, y, z = sp.symbols("x y z", real=True)
    polynomial = (x + y) ** 2 + (y + z) ** 2
    compilation = compile_sos_relaxation(polynomial, [x, y, z], 1)
    assert compilation.sparsity_mode == "chordal-overlap"
    assert compilation.main_gram_count == 2
    assert compilation.sdp_problem.num_blocks == 2
    assert {frozenset(block) for block in compilation.variable_blocks} == {
        frozenset([x, y]),
        frozenset([y, z]),
    }
    assert any(set(separator) == {y} for separator in compilation.clique_separators)


def test_overlapping_chordal_exact_certificate_reconstruction():
    x, y, z = sp.symbols("x y z", real=True)
    polynomial = (x + y) ** 2 + (y + z) ** 2
    result = sos_decompose(polynomial, [x, y, z])
    assert result.success
    assert result.certificate.exact_residual_zero
    assert result.certificate.exact_psd_checks == [True, True]


def test_chordal_fill_handles_overlapping_cycle_cliques():
    x, y, z, w = sp.symbols("x y z w", real=True)
    polynomial = (x + y) ** 2 + (y + z) ** 2 + (z + w) ** 2 + (w + x) ** 2
    compilation = compile_sos_relaxation(polynomial, [x, y, z, w], 1)
    assert compilation.sparsity_mode == "chordal-overlap"
    assert compilation.main_gram_count >= 2
    assert (
        max(
            len(block.variables)
            for block in compilation.gram_blocks[: compilation.main_gram_count]
        )
        < 4
    )
    # Every original interaction edge must be covered by a clique block.
    blocks = [set(block) for block in compilation.variable_blocks]
    for edge in ({x, y}, {y, z}, {z, w}, {w, x}):
        assert any(edge.issubset(block) for block in blocks)
