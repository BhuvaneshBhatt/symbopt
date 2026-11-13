"""Exact semialgebraic optimization regression cases."""

import pytest
import sympy as sp

from symbopt import minimize_result

semialg = pytest.importorskip("semialg")

x, y, z, a = sp.symbols("x y z a", real=True)


@pytest.mark.reference
@pytest.mark.parametrize(
    "objective,constraints,variables,expected,attained",
    [
        pytest.param(
            x,
            [sp.Le(x**2 + y**2, 1)],
            [x, y],
            -1,
            True,
            id="unit_disk_boundary_minimum",
        ),
        pytest.param(
            x,
            [sp.Lt(x**2 + y**2, 1)],
            [x, y],
            -1,
            False,
            id="open_disk_unattained_infimum",
        ),
        pytest.param(
            x,
            [sp.Le(x**2 + y**2, 0)],
            [x, y],
            0,
            True,
            id="zero_radius_disk",
        ),
        pytest.param(
            x**2 + 2 * y**2 + 3 * z**2,
            [sp.Ge(x**3 - 2 * x * y + 3 * z**3, 0), sp.Ge(x - y + z**2, 0)],
            [x, y, z],
            0,
            True,
            id="constrained_positive_quadratic",
        ),
        pytest.param(
            x + y,
            [sp.Eq(x * y, 1), sp.Ge(x, 0), sp.Ge(y, 0)],
            [x, y],
            2,
            True,
            id="positive_hyperbola_minimum",
        ),
        pytest.param(
            x**4,
            [sp.Ge(x, 3), sp.Le(x, 4)],
            [x],
            81,
            True,
            id="quartic_on_closed_interval",
        ),
        pytest.param(
            a * x**2,
            [sp.Gt(a, 0)],
            [a, x],
            0,
            True,
            id="positive_parameter_times_square",
        ),
    ],
)
def test_semialgebraic_cases(objective, constraints, variables, expected, attained):
    result = minimize_result(
        objective,
        constraints,
        variables=variables,
        use_sos=False,
        semialg_certification="auto",
    )
    assert result.certified is True
    assert sp.simplify(result.optimum_value - expected) == 0
    assert result.attained is attained
