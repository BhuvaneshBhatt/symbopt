import pytest
import sympy as sp

from symbopt import sos_decompose


pytestmark = pytest.mark.property


@pytest.mark.parametrize("a", range(-6, 7))
@pytest.mark.parametrize("b", range(-6, 7))
def test_affine_square_is_sos(a, b):
    """Every affine square in the bounded integer test grid is exactly SOS."""
    x = sp.symbols("x", real=True)
    poly = (a * x + b) ** 2
    result = sos_decompose(poly, [x])

    assert result.success
    assert result.certificate is not None
    assert result.certificate.exact_residual_zero
    assert all(check is True for check in result.certificate.exact_psd_checks)
