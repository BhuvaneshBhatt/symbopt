from __future__ import annotations

import pytest
import sympy as sp

from symbopt import minimize

pytestmark = [
    pytest.mark.reference,
    pytest.mark.capability_probe,
]


@pytest.mark.timeout(10)
def test_capability_probe_parameterized_quadratic():
    x, a = sp.symbols("x a", real=True)
    assert minimize((x - a) ** 2, variables=x) == [0, {x: a}]


@pytest.mark.timeout(10)
@pytest.mark.unsupported
@pytest.mark.xfail(
    strict=True,
    reason="parameter-stratified feasible-set optimization is not implemented",
)
def test_capability_probe_unit_circle_with_unfixed_parameter():
    x, y = sp.symbols("x y", real=True)
    result = minimize(x, sp.Eq(x**2 + y**2, 1), variables=x)
    assert result == [-sp.sqrt(1 - y**2), {x: -sp.sqrt(1 - y**2)}]
