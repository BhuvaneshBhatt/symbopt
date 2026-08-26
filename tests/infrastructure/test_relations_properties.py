import pytest
import sympy as sp
from symbopt.relation_checks import normalize_relational

hypothesis = pytest.importorskip("hypothesis")
st = pytest.importorskip("hypothesis.strategies")
given = hypothesis.given
settings = hypothesis.settings

pytestmark = pytest.mark.property

SMALL_INT = st.integers(min_value=-6, max_value=6)
NONZERO_INT = SMALL_INT.filter(lambda value: value != 0)


@given(a=NONZERO_INT, b=SMALL_INT, x_value=SMALL_INT)
@settings(max_examples=30, deadline=None)
def test_property_relational_normalization(a, b, x_value):
    x = sp.symbols("x", real=True)
    relation = sp.Ge(a * x + b, 0)
    kind, residual = normalize_relational(relation)
    assert kind == "ge"
    exact = bool(relation.subs(x, x_value))
    normalized = bool(sp.Ge(residual.subs(x, x_value), 0))
    assert exact == normalized
