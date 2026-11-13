from __future__ import annotations

import sympy as sp
from hypothesis import given, settings
from hypothesis import strategies as st

from symbopt import minimize


# Backend imports make cold timings variable; performance is measured separately.
@settings(deadline=None)
@given(st.integers(-3, 3), st.integers(1, 5))
def test_equivalent_constraint_containers_have_same_result(center, radius):
    x = sp.symbols("x", real=True)
    lower = center - radius
    upper = center + radius
    objective = (x - center) ** 2
    forms = [
        [x >= lower, x <= upper],
        (x >= lower, x <= upper),
        sp.And(x >= lower, x <= upper),
        {x >= lower, x <= upper},
    ]
    results = [minimize(objective, constraints, variables=x) for constraints in forms]
    assert all(result == results[0] for result in results[1:])
    assert minimize(objective, variables=x, bounds={x: (lower, upper)}) == results[0]


@settings(deadline=None)
@given(st.integers(-4, 4))
def test_scalar_and_sequence_variable_forms_are_equivalent(shift):
    x = sp.symbols("x", real=True)
    objective = (x - shift) ** 2
    assert minimize(objective, variables=x) == minimize(objective, variables=[x])


def test_coordinate_and_row_permutations():
    x, y = sp.symbols("x y", real=True)
    objective = (x - 1) ** 2 + (y - 2) ** 2
    constraints_a = [x >= 0, y >= 0, x + y <= 4]
    constraints_b = [x + y <= 4, y >= 0, x >= 0]
    first = minimize(objective, constraints_a, variables=[x, y])
    second = minimize(objective, constraints_b, variables=[y, x])
    assert first[0] == second[0]
    assert first[1] == second[1]


def test_disjunctive_optimizer_order():
    x = sp.symbols("x", real=True)
    first = minimize(x**2, sp.Or(x <= -1, x >= 1), variables=x)
    second = minimize(x**2, sp.Or(x >= 1, x <= -1), variables=x)
    assert first == second
