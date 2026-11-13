import pytest
import sympy as sp

from symbopt import (
    argmax,
    argmin,
    maximize,
    maximize_result,
    maxval,
    minimize,
    minimize_result,
    minval,
)


@pytest.mark.parametrize(
    "optimize",
    [
        minimize_result,
        maximize_result,
        minimize,
        maximize,
        minval,
        maxval,
        argmin,
        argmax,
    ],
)
@pytest.mark.parametrize("explicit", [True, False])
def test_integer_domain_rejected(optimize, explicit):
    n = sp.Symbol("n", integer=True)
    kwargs = {"variables": n} if explicit else {}
    with pytest.raises(NotImplementedError, match="integer optimization"):
        optimize((2 * n - 1) ** 2, **kwargs)


@pytest.mark.parametrize(
    "assumptions", [{"integer": True}, {"even": True}, {"odd": True}]
)
@pytest.mark.parametrize("use_assumptions", [False, True])
def test_implied_integrality_rejected(assumptions, use_assumptions):
    n = sp.Symbol("n", **assumptions)
    x = sp.Symbol("x", real=True)
    with pytest.raises(NotImplementedError, match="integer optimization"):
        minimize_result(
            (x - n) ** 2, variables=(x, n), use_symbol_assumptions=use_assumptions
        )


def test_unused_integer_variable_rejected():
    n = sp.Symbol("n", integer=True)
    x = sp.Symbol("x", real=True)
    with pytest.raises(NotImplementedError, match="integer optimization"):
        minimize_result(x * x, variables=(x, n))
    with pytest.raises(NotImplementedError, match="integer optimization"):
        minimize_result(sp.S(3), variables=n)


def test_integer_constraint_variable_rejected():
    n = sp.Symbol("n", integer=True)
    with pytest.raises(NotImplementedError, match="integer optimization"):
        minimize_result(sp.S(3), n >= 0)


def test_integer_parameter_allowed():
    n = sp.Symbol("n", integer=True)
    x = sp.Symbol("x", real=True)
    result = minimize_result((x - n) ** 2, variables=x)
    assert result.certified and result.attained is True
    assert result.optimum_value == 0
    assert result.best_candidate.primal_values == {x: n}


@pytest.mark.parametrize(
    "name",
    [
        "minimize_local",
        "maximize_local",
        "minimize_local_constrained",
        "maximize_local_constrained",
    ],
)
def test_local_integer_domain_rejected(name):
    import symbopt

    n = sp.Symbol("n", integer=True)
    with pytest.raises(NotImplementedError, match="integer optimization"):
        getattr(symbopt, name)((2 * n - 1) ** 2, n, {n: 0})
