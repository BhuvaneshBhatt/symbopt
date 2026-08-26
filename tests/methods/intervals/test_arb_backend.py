import math

import pytest
import sympy as sp

from symbopt.verified_intervals import (
    EvalConfig,
    arb_available,
    enclose_expression,
)

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not arb_available(), reason="python-flint not installed")
def test_arb_transcendental_bound():
    x = sp.symbols("x", real=True)
    bound = enclose_expression(
        sp.sin(x),
        (x,),
        ((0.0, 2 * math.pi),),
        EvalConfig(precision=192),
    )
    assert bound.rigorous
    assert bound.domain == "valid"
    assert bound.lower <= -1
    assert bound.upper >= 1
    assert "natural" in bound.method


@pytest.mark.skipif(not arb_available(), reason="python-flint not installed")
def test_broader_arb_live_enclosures():
    x = sp.symbols("x", real=True)
    for expr, bounds in [
        (sp.zeta(x), ((2.0, 3.0),)),
        (sp.LambertW(x), ((0.1, 1.0),)),
        (sp.besselj(0, x), ((0.0, 1.0),)),
        (sp.Ei(x), ((1.0, 2.0),)),
    ]:
        bound = enclose_expression(expr, (x,), bounds)
        assert bound.rigorous
        assert bound.domain == "valid"
        assert bound.lower <= bound.upper
