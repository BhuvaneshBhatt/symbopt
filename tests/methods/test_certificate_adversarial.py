"""SOS replay reconstructs the theorem instead of trusting cached flags."""

from dataclasses import replace

import pytest
import sympy as sp

from symbopt import sos_decompose

x = sp.Symbol("x", real=True)


@pytest.fixture(scope="module")
def certificate():
    result = sos_decompose((x - 1) ** 2, [x])
    assert result.certificate.verify(polynomial=(x - 1) ** 2, lower_bound=0)
    return result.certificate


@pytest.mark.parametrize(
    "changes",
    [
        {"gram_matrices": [sp.diag(-1, -1)]},
        {"gram_matrices": []},
        {"gram_bases": ()},
        {"variables": (x, x)},
        {"gram_bases": ((sp.S.One,),)},
        {"gram_multipliers": (-sp.S.One,)},
        {"polynomial": x * x - 2 * x},
        {"lower_bound": sp.S.One},
        {"normalization_scale": -sp.S.One},
        {"equalities": (x,)},
        {"eq_multiplier_polys": [sp.S.One]},
        {"residual": sp.Rational(1, 10**12)},
        {"exact_residual_zero": False},
        {"exact_psd_checks": [None]},
    ],
)
def test_altered_sos_theorem(certificate, changes):
    assert not replace(certificate, **changes).verified


def test_unknown_gram_sign(certificate):
    a = sp.Symbol("a", real=True)
    bad = replace(
        certificate,
        polynomial=a * x * x,
        gram_matrices=[sp.Matrix([[a]])],
        gram_bases=((x,),),
        exact_psd_checks=[True],
    )
    assert not bad.verified


def test_source_binding(certificate):
    assert not certificate.verify(polynomial=x * x)
    assert not certificate.verify(lower_bound=1)


def test_multiplier_binding(certificate):
    constrained = replace(
        certificate, polynomial=x * (x - 1) ** 2, gram_multipliers=(x,)
    )
    assert constrained.verify(inequalities=(x,))
    assert not constrained.verify(inequalities=(-x,))


def test_nonsymmetric_gram(certificate):
    bad = replace(certificate, gram_matrices=[sp.Matrix([[1, -2], [0, 1]])])
    assert not bad.verified


def test_coherent_negative_gram(certificate):
    bad = replace(
        certificate,
        polynomial=-certificate.polynomial,
        gram_matrices=[-matrix for matrix in certificate.gram_matrices],
    )
    basis = sp.Matrix(bad.gram_bases[0])
    assert sp.expand(bad.polynomial - (basis.T * bad.gram_matrices[0] * basis)[0]) == 0
    assert bad.exact_residual_zero and bad.exact_psd_checks == [True]
    assert not bad.verified


def test_changed_matrix_after_replay(certificate):
    bad = replace(
        certificate,
        gram_matrices=[
            sp.MutableDenseMatrix(matrix) for matrix in certificate.gram_matrices
        ],
    )
    assert bad.verified
    bad.gram_matrices[0][0, 0] += 1
    assert not bad.verified
