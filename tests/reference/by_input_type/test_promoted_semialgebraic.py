"""Exact semialgebraic optimization regression cases."""

import pytest
import sympy as sp

from symbopt import minimize

semialg = pytest.importorskip("semialg")

x, y, z, a = sp.symbols("x y z a", real=True)


@pytest.mark.reference
@pytest.mark.parametrize(
    "case_no,objective,constraints,variables,expected,attained",
    [
        (1, x, [sp.Le(x**2 + y**2, 1)], [x, y], -1, True),
        (2, x, [sp.Lt(x**2 + y**2, 1)], [x, y], -1, False),
        (4, x, [sp.Le(x**2 + y**2, 0)], [x, y], 0, True),
        (
            6,
            x**2 + 2 * y**2 + 3 * z**2,
            [sp.Le(x**2 + y**2, 1), sp.Le(y**2 + z**2, 1)],
            [x, y, z],
            0,
            True,
        ),
        (
            24,
            x + y,
            [sp.Eq(x * y, 1), sp.Ge(x, 0), sp.Ge(y, 0)],
            [x, y],
            2,
            True,
        ),
        (25, x**4, [sp.Ge(x, 3), sp.Le(x, 4)], [x], 81, True),
        (235, a * x**2, [sp.Gt(a, 0)], [a, x], 0, True),
    ],
)
def test_promoted_semialgebraic_reference_cases(
    case_no, objective, constraints, variables, expected, attained
):
    result = minimize(
        objective,
        constraints,
        variables=variables,
        use_sos=False,
        semialg_certification="auto",
    )
    assert result.certified is True, case_no
    assert sp.simplify(result.optimum_value - expected) == 0, case_no
    assert result.attained is attained, case_no
    assert result.semialg_result is not None, case_no
