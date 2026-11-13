import sympy as sp

import symbopt


def test_preordering_products_include_pair_product():
    x = sp.symbols("x")
    products = symbopt.preordering_generators([x, 1 - x])
    assert [item.indices for item in products] == [(0,), (1,), (0, 1)]
    assert sp.expand(products[-1].polynomial - x * (1 - x)) == 0


def test_infeasibility_certificate_with_equality_ideal():
    x = sp.symbols("x", real=True)
    result = symbopt.search_infeasibility_certificate(
        [x], [-x - 1], [x], force=True, multiplier_degree=0
    )
    assert result.certified
    assert result.certificate.verified
    assert result.certificate.ideal_remainder == 0
    assert symbopt.verify_infeasibility_certificate(result.certificate, [x])


def test_infeasibility_certificate_simple_inequalities():
    x = sp.symbols("x", real=True)
    result = symbopt.search_infeasibility_certificate(
        [], [x, -x - 1], [x], force=True, multiplier_degree=0
    )
    assert result.certified
