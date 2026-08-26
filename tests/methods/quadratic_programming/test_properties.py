import pytest
import sympy as sp
from symbopt import minimize

hypothesis = pytest.importorskip("hypothesis")
st = pytest.importorskip("hypothesis.strategies")
given = hypothesis.given
settings = hypothesis.settings

pytestmark = pytest.mark.property

SMALL_INT = st.integers(min_value=-6, max_value=6)
NONZERO_INT = SMALL_INT.filter(lambda value: value != 0)


@given(center=SMALL_INT)
@settings(max_examples=20, deadline=None)
def test_property_unconstrained_qp_center(center):
    x = sp.symbols("x", real=True)
    result = minimize((x - center) ** 2)
    assert result.best_candidate is not None
    point = result.best_candidate.original_variable_values[x]
    assert sp.simplify(point - center) == 0
    assert sp.simplify(result.best_candidate.objective_value) == 0
