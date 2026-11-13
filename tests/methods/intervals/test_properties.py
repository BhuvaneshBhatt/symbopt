import pytest
import sympy as sp

from symbopt import enclose_expression

hypothesis = pytest.importorskip("hypothesis")
st = pytest.importorskip("hypothesis.strategies")
given = hypothesis.given
settings = hypothesis.settings

pytestmark = pytest.mark.property

SMALL_INT = st.integers(min_value=-6, max_value=6)
NONZERO_INT = SMALL_INT.filter(lambda value: value != 0)


@given(
    a=SMALL_INT,
    b=SMALL_INT,
    c=SMALL_INT,
    low=st.integers(min_value=-4, max_value=1),
    width=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=30, deadline=None)
def test_property_polynomial_verified_bound_contains_samples(a, b, c, low, width):
    x = sp.symbols("x", real=True)
    high = low + width
    poly = a * x**2 + b * x + c
    bound = enclose_expression(poly, (x,), ((float(low), float(high)),))
    assert bound.rigorous
    for value in (low, (low + high) / 2, high):
        exact = float(sp.N(poly.subs(x, value), 30))
        assert bound.lower - 1e-12 <= exact <= bound.upper + 1e-12
