import sympy as sp

from symbopt.semialgebraic import (
    _truth_value,
    certify_global_min,
    exact_feasible,
    make_formula,
    minimize_exact,
    prune_redundant,
    semialg_available,
    solve_exact_points,
)


class _SatResult:
    def __init__(self, value, method="fake_qe", witness=None):
        self.satisfiable = value
        self.method = method
        self.witness = witness


class _ImpResult:
    def __init__(self, valid):
        self.valid = valid


class _SolveResult:
    def __init__(self, samples):
        self.samples = tuple(samples)


class _OptResult:
    def __init__(self, value, points, attained=True):
        self.value = value
        self.points = tuple(points)
        self.attained = attained
        self.method = "fake_exact"
        self.diagnostics = {"exact": True}


class _FakeSemialg:
    def __init__(self, sat_answers=None):
        self.sat_answers = list(sat_answers or [True])

    def is_satisfiable(self, formula, variables, return_result=True):
        value = self.sat_answers.pop(0)
        witness = {var: sp.Integer(0) for var in variables} if value else None
        return _SatResult(value, witness=witness)

    def implies(self, assumptions, conclusion, variables, return_result=True):
        x = variables[0]
        valid = sp.simplify(conclusion.lhs - x) == 0 and conclusion == sp.Ge(x, 0)
        return _ImpResult(valid)

    def solve_semialgebraic(self, formula, variables, method="auto", count=1):
        x = variables[0]
        return _SolveResult([{x: sp.sqrt(2)}, {x: -sp.sqrt(2)}])

    def semialgebraic_minimize(
        self, objective, constraints, variables, return_result=True
    ):
        x = variables[0]
        return _OptResult(sp.Integer(0), [{x: sp.Integer(2)}])


class _SignResult:
    def __init__(self, proven, counterexample=None):
        self.proven = proven
        self.method = "fake_sign"
        self.counterexample = counterexample


class _RangeBackend:
    def prove_nonnegative(self, expr, variables, return_result=True):
        return _SignResult(sp.expand(expr).is_nonnegative is not False)

    def implies(self, assumptions, conclusion, variables, return_result=True):
        return _ImpResult(True)

    def function_range(self, expr, constraints, variables, value_symbol=None):
        return sp.And(sp.Ge(value_symbol, 0), sp.Le(value_symbol, 4))

    def semialgebraic_minimize(
        self, objective, constraints, variables, return_result=True
    ):
        return _OptResult(sp.Integer(0), [{variables[0]: sp.Integer(0)}])

    def semialgebraic_maximize(
        self, objective, constraints, variables, return_result=True
    ):
        return _OptResult(sp.Integer(4), [{variables[0]: sp.Integer(4)}])


class _NoRURBackend:
    """Backend stub without a zero-dimensional solve entry point."""


def test_formula_and_exact_feasible_with_injected_backend():
    x = sp.symbols("x", real=True)
    formula = make_formula([sp.Ge(x, 0), sp.Le(x, 2)])
    assert isinstance(formula, sp.And)
    backend = _FakeSemialg([True])
    assert exact_feasible(formula, [x], backend=backend) is True


def test_redundancy_uses_exact_implication():
    x = sp.symbols("x", real=True)
    backend = _FakeSemialg()
    reduced = prune_redundant(
        [sp.Ge(x, 1), sp.Ge(x, 0)],
        [x],
        backend=backend,
    )
    assert reduced == [sp.Ge(x, 1)]


def test_rur_fallback_converts_exact_samples():
    x = sp.symbols("x", real=True)
    points = solve_exact_points([x**2 - 2], [x], backend=_FakeSemialg())
    assert {sp.simplify(point[x]) for point in points} == {sp.sqrt(2), -sp.sqrt(2)}


def test_exact_point_solver_falls_back_when_rur_api_is_missing():
    x = sp.symbols("x", real=True)
    points = solve_exact_points([x**2 - 2], [x], backend=_NoRURBackend())
    assert {sp.simplify(point[x]) for point in points} == {sp.sqrt(2), -sp.sqrt(2)}


def test_global_min_query_requires_candidate_feasibility_then_unsat():
    x = sp.symbols("x", real=True)
    backend = _FakeSemialg([True, False])
    proof = certify_global_min(
        x**2,
        [sp.Ge(x, 0)],
        [x],
        {x: sp.Integer(0)},
        sp.Integer(0),
        backend=backend,
    )
    assert proof.proven is True
    assert proof.method == "fake_qe"


def test_exact_minimizer_result_conversion():
    x = sp.symbols("x", real=True)
    result = minimize_exact(
        (x - 2) ** 2,
        [sp.Ge(x, 0)],
        [x],
        backend=_FakeSemialg(),
    )
    assert result is not None
    assert result.value == 0
    assert result.points == ({x: 2},)
    assert result.attained


def test_sign_proof_adapter_uses_semialg_sign_backend():
    from symbopt.semialgebraic import prove_nonnegative_exact

    x = sp.symbols("x", real=True)
    proof = prove_nonnegative_exact(
        x**2 + 1,
        [],
        [x],
        backend=_RangeBackend(),
    )
    assert proof.proven is True
    assert proof.method == "fake_sign"


def test_constrained_sign_proof_uses_exact_implication():
    from symbopt.semialgebraic import prove_nonnegative_exact

    x = sp.symbols("x", real=True)
    proof = prove_nonnegative_exact(
        x,
        [sp.Ge(x, 0)],
        [x],
        backend=_RangeBackend(),
    )
    assert proof.proven is True


def test_function_range_extracts_exact_endpoints():
    from symbopt.semialgebraic import function_range_exact

    x = sp.symbols("x", real=True)
    result = function_range_exact(
        x**2,
        [sp.Ge(x, -2), sp.Le(x, 2)],
        [x],
        backend=_RangeBackend(),
    )
    assert result is not None
    assert result.infimum == 0
    assert result.supremum == 4
    assert result.infimum_attained is True
    assert result.supremum_attained is True
    assert result.formula.has(result.value_symbol)


def test_truth_value_preserves_false_and_unknown():
    assert _truth_value(False) is False

    class Unknown:
        satisfiable = None

    assert _truth_value(Unknown()) is None


def test_exact_feasible_preserves_false_and_unknown():
    x = sp.symbols("x", real=True)
    assert (
        exact_feasible(
            [sp.Ge(x, 0)],
            [x],
            backend=_FakeSemialg([False]),
        )
        is False
    )

    class UnknownSemialg(_FakeSemialg):
        def is_satisfiable(self, formula, variables, return_result=True):
            return _SatResult(None)

    assert (
        exact_feasible(
            [sp.Ge(x, 0)],
            [x],
            backend=UnknownSemialg(),
        )
        is None
    )


def test_global_proof_does_not_treat_unknown_as_unsat():
    x = sp.symbols("x", real=True)

    class UnknownSemialg(_FakeSemialg):
        def __init__(self):
            self.calls = 0

        def is_satisfiable(self, formula, variables, return_result=True):
            self.calls += 1
            if self.calls == 1:
                return _SatResult(True)
            return _SatResult(None)

    proof = certify_global_min(
        x**2,
        [sp.Ge(x, 0)],
        [x],
        {x: sp.Integer(0)},
        sp.Integer(0),
        backend=UnknownSemialg(),
    )
    assert proof.proven is None


def test_semialg_available_bool():
    first = semialg_available()
    second = semialg_available()
    assert isinstance(first, bool)
    assert second is first


def test_capability_detection_and_policy_forwarding_for_modern_backend():
    from symbopt.semialgebraic import get_semialg_capabilities

    x, y, z = sp.symbols("x y z", real=True)

    class ModernResult(_OptResult):
        def __init__(self):
            super().__init__(sp.Integer(0), [{x: 0, y: 0, z: 0}], attained=True)
            self.certified = True

    class ModernSemialg:
        __version__ = "0.2.0b1"

        def __init__(self):
            self.received = None

        def semialgebraic_minimize(
            self,
            objective,
            constraints,
            variables,
            *,
            return_result=True,
            certification="auto",
            range_cost_limit=2500,
        ):
            self.received = (tuple(variables), certification, range_cost_limit)
            return ModernResult()

    backend = ModernSemialg()
    caps = get_semialg_capabilities(backend)
    assert caps.version == "0.2.0b1"
    assert caps.has_optimizer
    assert caps.supports_cert_policy
    assert caps.supports_range_limit

    result = minimize_exact(
        x**4 + y**2 + z**2,
        [],
        [x, y, z],
        backend=backend,
        certification="complete",
        range_cost_limit=17,
    )
    assert result is not None
    assert result.certified is True
    assert result.backend_version == "0.2.0b1"
    assert backend.received == ((x, y, z), "complete", 17)


def test_minimize_exact_has_no_symbopt_dimension_ceiling():
    a, b, c, d = sp.symbols("a b c d", real=True)

    class HighDimResult(_OptResult):
        def __init__(self):
            super().__init__(
                sp.Integer(0),
                [{a: 0, b: 0, c: 0, d: 0}],
                attained=True,
            )
            self.certified = True

    class HighDimBackend:
        def semialgebraic_minimize(
            self,
            objective,
            constraints,
            variables,
            *,
            return_result=True,
            certification="auto",
            range_cost_limit=2500,
        ):
            assert tuple(variables) == (a, b, c, d)
            return HighDimResult()

    result = minimize_exact(
        a**4 + b**2 + c**2 + d**2,
        [],
        [a, b, c, d],
        backend=HighDimBackend(),
    )
    assert result is not None
    assert result.value == 0
    assert result.certified is True


def test_uncertified_semialg_candidate_is_not_a_rigorous_lower_bound():
    from symbopt.certification import build_certificate
    from symbopt.semialgebraic import SemialgOptResult

    x = sp.symbols("x", real=True)
    result = SemialgOptResult(
        value=sp.Integer(1),
        points=({x: sp.Integer(1)},),
        attained=True,
        method="candidate_only",
        diagnostics={},
        certified=False,
    )
    certificate = build_certificate(semialg_result=result)
    assert certificate.lower_bound is None
    assert certificate.upper_bound == 1
    assert certificate.status != "proven_optimal"
