"""Reference wrapper cases promoted after the public extremum API was added."""

import pytest
import sympy as sp

from symbopt import argmin, minval

semialg = pytest.importorskip("semialg")

x, y = sp.symbols("x y", real=True)


@pytest.mark.reference
def test_wrapper_reference_unit_circle_cases_245_to_249():
    constraints = [sp.Eq(x**2 + y**2, 1)]
    assert minval(x, constraints, variables=[x, y], use_sos=False) == -1
    points = argmin(x, constraints, variables=[x, y], use_sos=False)
    assert points
    assert any(
        sp.simplify(point[x] + 1) == 0 and sp.simplify(point[y]) == 0
        for point in points
    )
