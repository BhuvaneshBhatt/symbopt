import sympy as sp

from symbopt import build_kkt_system, critical_points
from symbopt.candidates import solve_exact_candidates, solve_kkt_candidates
from symbopt.algebraize import algebraize


def test_constrained_critical_point():
    x, y = sp.symbols("x y", real=True)
    r = critical_points((x**2 + y**2, [sp.Eq(x + y, 1)]), [x, y])
    assert r
    assert any(sp.simplify(cp.solution.get(x, 0) - sp.Rational(1, 2)) == 0 for cp in r)


def test_kkt_system_has_complementarity():
    x = sp.symbols("x", real=True)
    p = algebraize(x**2, [sp.Ge(x, 1)])
    k = build_kkt_system(p)
    assert len(k.complementarity_eqs) == len(k.inequality_multipliers)


def test_symbolic_candidate_generation():
    x = sp.symbols("x", real=True)
    p = algebraize((x - 1) ** 2, [])
    c = solve_exact_candidates(p)
    assert any(sp.simplify(v.primal_values[x] - 1) == 0 for v in c)


def test_nsolve_candidate_generation_bounded():
    x = sp.symbols("x", real=True)
    p = algebraize((x - 1) ** 2, [sp.Ge(x, 0), sp.Le(x, 2)])
    c = solve_kkt_candidates(p, box_bounds={x: (0, 2)}, max_active_sets=8)
    assert isinstance(c, list)


def test_kkt_inequality_sign_boundary_minimum():
    x = sp.symbols("x", real=True)
    r = critical_points((x, [sp.Ge(x, 1)]), [x])
    assert any(sp.simplify(cp.solution.get(x, 0) - 1) == 0 for cp in r)


def test_constrained_reduced_hessian_classifies_minimum():
    x, y = sp.symbols("x y", real=True)
    r = critical_points((-(x**2) + y**2, [sp.Eq(x, 0)]), [x, y])
    assert any(
        cp.solution.get(x) == 0
        and cp.solution.get(y) == 0
        and cp.classification == "LocalMinimum"
        for cp in r
    )


def test_karush_john_multipliers_are_nontrivial():
    x = sp.symbols("x", real=True)
    r = critical_points(
        (x**2, [sp.Ge(x, 0)]),
        [x],
        karush_john_multiplier="rho",
    )
    assert r
    for cp in r:
        rho = next(symbol for symbol in cp.solution if str(symbol) == "rho")
        multipliers = [rho] + [
            symbol for symbol in cp.solution if str(symbol).startswith(("lam_", "mu_"))
        ]
        total = sp.simplify(sum(cp.solution[symbol] ** 2 for symbol in multipliers))
        assert total == 1
