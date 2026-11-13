from __future__ import annotations

import sympy as sp

from symbopt.exact_refinement import reconstruct_exact_vector


def test_large_algebraic_vectors_skip_common_number_field():
    values = [sp.N(sp.sqrt(2), 30)] * 8
    methods = [method for _candidate, method in reconstruct_exact_vector(values)]
    assert methods
    assert "common_number_field_reconstruction" not in methods


def test_small_algebraic_vector_reconstruction_remains_exact():
    values = [sp.sqrt(2), 1 + sp.sqrt(2)]
    candidates = [candidate for candidate, _method in reconstruct_exact_vector(values)]
    assert candidates
    assert any(candidate == tuple(values) for candidate in candidates)
