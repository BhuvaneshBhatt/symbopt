import pytest
import sympy as sp

from symbopt import minimize_result

x, y = sp.symbols("x y", real=True)


def test_multivariate_linear_fractional_charnes_cooper_exact():
    x, y = sp.symbols("x y", real=True)
    result = minimize_result(
        (x + 2 * y + 1) / (x + y + 3),
        [sp.Ge(x, 0), sp.Le(x, 1), sp.Ge(y, 0), sp.Le(y, 1)],
        variables=[x, y],
        use_sos=False,
        options={"structural": {"enabled": False}},
    )
    assert result.certified is True
    assert result.optimum_value == sp.Rational(1, 3)
    assert result.attained is True
    assert result.structural_result.kind == "linear_fractional"
    assert result.best_candidate.original_variable_values == {x: 0, y: 0}
    assert result.structural_result.metadata["transformation"] == "Charnes-Cooper"


def test_linear_fractional_negative_denominator_region():
    x = sp.symbols("x", real=True)
    result = minimize_result(
        (x + 1) / (x - 3),
        [sp.Ge(x, 0), sp.Le(x, 1)],
        variables=[x],
        use_sos=False,
        options={"structural": {"enabled": False}},
    )
    assert result.certified is True
    assert result.optimum_value == -1
    assert result.attained is True
    assert result.best_candidate.original_variable_values[x] == 1
    assert result.structural_result.metadata["denominator_sign"] == -1


def test_linear_fractional_recession_infimum_is_unattained():
    x = sp.symbols("x", real=True)
    result = minimize_result(
        1 / x,
        [sp.Ge(x, 1)],
        variables=[x],
        use_sos=False,
        options={"structural": {"enabled": False}},
    )
    assert result.certified is True
    assert result.optimum_value == 0
    assert result.attained is False
    assert result.best_candidate is None
    assert result.structural_result.kind == "linear_fractional"
    assert result.structural_result.metadata["asymptotic"] is True


def test_fractional_denominator_signs():
    x = sp.symbols("x", real=True)
    # x/(x-1) has denominator of either sign on [-2, 3], with a pole at 1.
    # The positive branch tends to -infinity as x -> 1+.
    result = minimize_result(
        x / (x - 1),
        [sp.Ge(x, -2), sp.Le(x, 3)],
        use_sos=False,
        options={"structural": {"enabled": False}},
    )
    assert result.structural_result.kind == "linear_fractional"
    assert result.structural_result.metadata.get("denominator_sign_split") is True
    assert result.optimum_value is -sp.oo
    assert result.attained is False


def test_linear_fractional_sign_split_finite_negative_branch():
    x = sp.symbols("x", real=True)
    result = minimize_result(
        (x + 2) / (x - 1),
        [sp.Ge(x, -2), sp.Le(x, 0)],
        use_sos=False,
        options={"structural": {"enabled": False}},
    )
    assert result.certified
    assert result.structural_result is not None
    assert sp.simplify(result.optimum_value + 2) == 0
    assert result.attained is True
    assert sp.simplify(result.best_candidate.original_variable_values[x]) == 0


def test_strict_fractional_infimum():
    x = sp.symbols("x", real=True)
    result = minimize_result(
        (x + 1) / (x + 2),
        [sp.Gt(x, 0)],
        use_sos=False,
        options={"structural": {"enabled": False}},
    )
    assert result.certified
    assert sp.simplify(result.optimum_value - sp.Rational(1, 2)) == 0
    assert result.attained is False


@pytest.mark.parametrize(
    "objective,constraints,expected",
    [
        (1 + y / (1 + x + y), [sp.Gt(x, 1), sp.Gt(y, 1)], 1),
        (1 + y / (1 + x + y), [sp.Lt(y, -2), sp.Lt(y, 0)], -sp.oo),
        (1 + y / (1 + x + y), [sp.Lt(x, -2), sp.Lt(y, 0)], 1),
        (1 + y / (1 + x + y), [sp.Lt(x + y, -2), sp.Lt(x + y, 0)], -sp.oo),
        (1 + y / (1 + x + y), [sp.Gt(x + y, 1)], -sp.oo),
        (1 + y / (1 + x + y), [sp.Gt(x + y, 1), sp.Gt(x + 2 * y, 2)], 0),
        (1 + 2 / (x + 2 * y), [sp.Gt(x + 2 * y, 1)], 1),
        (1 + 2 / (x + 2 * y), [sp.Lt(x + 2 * y, 1), sp.Gt(x + 2 * y, -100)], -sp.oo),
        (1 + y / (x + 2 * y), [sp.Gt(x, 1), sp.Lt(x, 2)], -sp.oo),
    ],
)
def test_fractional_limit_cases(objective, constraints, expected):
    result = minimize_result(
        objective,
        constraints,
        use_sos=False,
        options={"structural": {"enabled": False}},
    )
    assert result.certified
    if expected is -sp.oo:
        assert result.optimum_value is -sp.oo
    else:
        assert sp.simplify(result.optimum_value - expected) == 0
    assert result.attained is False
    assert result.structural_result.kind == "linear_fractional"


def test_linear_fractional_strict_equality_infimum():
    result = minimize_result(
        1 + y / (1 + x + y),
        [sp.Eq(x + y, 1), sp.Gt(x + 2 * y, 2)],
        use_sos=False,
        options={"structural": {"enabled": False}},
    )
    assert result.certified
    assert sp.simplify(result.optimum_value - sp.Rational(3, 2)) == 0
    assert result.attained is False


def test_linear_fractional_equality_unbounded_below():
    result = minimize_result(
        1 + y / (1 + x + y),
        [sp.Eq(x + y, 1)],
        use_sos=False,
        options={"structural": {"enabled": False}},
    )
    assert result.certified
    assert result.optimum_value is -sp.oo
    assert result.attained is False


def test_fractional_branch_ordering():
    from symbopt.linear_fractional import _smaller_structural
    from symbopt.structural_results import StructuralGlobalResult

    left = StructuralGlobalResult(
        kind="linear_fractional",
        value=sp.sqrt(2),
        points=(),
        attained=False,
        certified=True,
        metadata={"branch": "left"},
    )
    right = StructuralGlobalResult(
        kind="linear_fractional",
        value=sp.sqrt(3),
        points=(),
        attained=False,
        certified=True,
        metadata={"branch": "right"},
    )
    assert _smaller_structural(left, right) is left
