"""Extremum wrappers on the unit circle."""

import pytest
import sympy as sp

from symbopt import argmax, argmin, maxval, minval

semialg = pytest.importorskip("semialg")

x, y = sp.symbols("x y", real=True)


@pytest.mark.reference
def test_unit_circle_wrappers():
    constraints = [sp.Eq(x**2 + y**2, 1)]
    assert minval(x, constraints, variables=[x, y], use_sos=False) == -1
    points = argmin(x, constraints, variables=[x, y], use_sos=False)
    assert points
    assert any(
        sp.simplify(point[x] + 1) == 0 and sp.simplify(point[y]) == 0
        for point in points
    )

    assert maxval(x, constraints, variables=[x, y], use_sos=False) == 1
    assert any(
        point == {x: 1, y: 0}
        for point in argmax(x, constraints, variables=[x, y], use_sos=False)
    )
