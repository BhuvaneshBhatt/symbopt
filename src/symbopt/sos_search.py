from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import comb

import sympy as sp
from sympy.core.relational import Relational
from sympy.polys.polyerrors import PolynomialError

from .input_normalization import normalize_constraint_branches, normalize_variables
from .semidefinite import SDPSolverConfig, solve_sdp
from .sos import SOSCertificate, SOSResult, _decode_certificate, _main_block_data
from .sos_program import SOSCompilation, compile_sos_relaxation


def _normalize_sos_domain(
    polynomial, variables, inequalities, equalities, constraints=None
):
    """Normalize relational/domain inputs for public SOS helpers."""
    polynomial = sp.expand(sp.sympify(polynomial))
    ineqs = []
    for expr in inequalities:
        expr = sp.sympify(expr)
        if isinstance(expr, (sp.GreaterThan, sp.StrictGreaterThan)):
            expr = expr.lhs - expr.rhs
        elif isinstance(expr, (sp.LessThan, sp.StrictLessThan)):
            expr = expr.rhs - expr.lhs
        elif isinstance(expr, sp.Equality):
            raise TypeError(
                "an equality was supplied in inequalities; use equalities= or constraints="
            )
        ineqs.append(sp.expand(expr))
    eqs = []
    for expr in equalities:
        expr = sp.sympify(expr)
        if isinstance(expr, sp.Equality):
            expr = expr.lhs - expr.rhs
        elif isinstance(expr, Relational):
            raise TypeError(
                "a non-equality relation was supplied in equalities; use inequalities= or constraints="
            )
        eqs.append(sp.expand(expr))
    if constraints is not None:
        branches = normalize_constraint_branches(constraints)
        if len(branches) != 1:
            raise ValueError(
                "SOS certificates require one conjunctive feasible region; split Or branches explicitly"
            )
        for relation in branches[0]:
            if isinstance(relation, sp.Equality):
                eqs.append(sp.expand(relation.lhs - relation.rhs))
            elif isinstance(relation, (sp.GreaterThan, sp.StrictGreaterThan)):
                ineqs.append(sp.expand(relation.lhs - relation.rhs))
            elif isinstance(relation, (sp.LessThan, sp.StrictLessThan)):
                ineqs.append(sp.expand(relation.rhs - relation.lhs))
            else:
                raise TypeError(f"unsupported SOS constraint: {relation!s}")
    normalized = normalize_variables(variables)
    if normalized is None:
        symbols = set(polynomial.free_symbols)
        for expr in (*ineqs, *eqs):
            symbols.update(expr.free_symbols)
        normalized = sorted(symbols, key=lambda symbol: symbol.sort_key())
    return polynomial, normalized, tuple(ineqs), tuple(eqs)


@dataclass(frozen=True)
class SOSCostEstimate:
    """Cheap upper-bound estimate for an SOS/SDP certificate search."""

    applicable: bool
    variable_count: int
    polynomial_degree: int | None
    relaxation_order: int | None
    main_gram_dimension: int
    estimated_psd_blocks: int
    scalar_decision_vars: int
    coefficient_constraints: int
    cost_class: str
    reason: str


@dataclass(frozen=True)
class SOSSearchPlan:
    """Planner decision intended for callers such as semialg dispatchers."""

    launch: bool
    estimate: SOSCostEstimate
    reason: str


@dataclass
class SOSCertificateSearchResult:
    """Semialg-facing SOS search result with an explicitly verified certificate.

    ``certified`` is true only when the exact polynomial identity is zero and
    every Gram block has been proved positive semidefinite exactly. Numerical
    solver feasibility is retained only as diagnostic metadata in ``raw_result``.
    """

    attempted: bool
    certified: bool
    status: str
    polynomial: sp.Expr
    variables: list[sp.Symbol]
    plan: SOSSearchPlan
    certificate: SOSCertificate | None
    raw_result: SOSResult | None
    message: str

    @property
    def gram_matrices(self) -> list[sp.Matrix]:
        """Recovered Gram matrices, or an empty list when no certificate is available."""
        return [] if self.certificate is None else list(self.certificate.gram_matrices)

    @property
    def exact_residual_zero(self) -> bool:
        """Stored exact-identity summary; full certificate replay remains separate."""
        return bool(
            self.certificate is not None and self.certificate.exact_residual_zero
        )

    @property
    def exact_psd_checks(self) -> list[bool | None]:
        """Stored Gram-positivity summaries from the recovered certificate."""
        return (
            [] if self.certificate is None else list(self.certificate.exact_psd_checks)
        )


def _poly_degree(expr: sp.Expr, variables: Sequence[sp.Symbol]) -> int | None:
    try:
        degree = sp.Poly(sp.expand(expr), *variables).total_degree()
    except (PolynomialError, TypeError, ValueError, ArithmeticError):
        return None
    if degree is sp.S.NegativeInfinity:
        return 0
    return int(degree)


def estimate_sos_cost(
    polynomial: sp.Expr,
    variables: sp.Symbol | Sequence[sp.Symbol] | None = None,
    *,
    inequalities: Sequence[sp.Expr] = (),
    equalities: Sequence[sp.Expr] = (),
    constraints: object = None,
    ineq_mult_degree: int = 1,
    eq_mult_degree: int = 1,
) -> SOSCostEstimate:
    """Estimate dense SOS size without compiling or solving an SDP.

    The estimate is conservative. Sparse/chordal compilation can
    make the realized problem much smaller, so callers may force a search after
    inspecting the plan when they have domain-specific reasons to do so.
    """
    polynomial, variables, inequalities, equalities = _normalize_sos_domain(
        polynomial, variables, inequalities, equalities, constraints
    )
    expressions = [polynomial, *inequalities, *equalities]
    if not all(isinstance(v, sp.Symbol) for v in variables):
        return SOSCostEstimate(
            False,
            len(variables),
            None,
            None,
            0,
            0,
            0,
            0,
            "unsupported",
            "variables must be SymPy symbols",
        )
    if ineq_mult_degree < 0 or eq_mult_degree < 0:
        return SOSCostEstimate(
            False,
            len(variables),
            None,
            None,
            0,
            0,
            0,
            0,
            "unsupported",
            "multiplier degrees must be nonnegative",
        )
    degrees = [_poly_degree(expr, variables) for expr in expressions]
    if any(degree is None for degree in degrees):
        return SOSCostEstimate(
            False,
            len(variables),
            None,
            None,
            0,
            0,
            0,
            0,
            "unsupported",
            "SOS search requires polynomial expressions",
        )

    target_degree = degrees[0] or 0
    order = (target_degree + 1) // 2
    nvars = len(variables)
    main_dim = comb(nvars + order, order) if nvars else 1
    ineq_dim = comb(nvars + ineq_mult_degree, ineq_mult_degree) if nvars else 1
    eq_dim = comb(nvars + eq_mult_degree, eq_mult_degree) if nvars else 1
    psd_blocks = 1 + len(inequalities)
    scalar_decisions = main_dim * (main_dim + 1) // 2
    scalar_decisions += len(inequalities) * ineq_dim * (ineq_dim + 1) // 2
    scalar_decisions += 2 * len(equalities) * eq_dim

    represented_degree = max(
        [2 * order]
        + [
            2 * ineq_mult_degree + (degree or 0)
            for degree in degrees[1 : 1 + len(inequalities)]
        ]
        + [
            eq_mult_degree + (degree or 0)
            for degree in degrees[1 + len(inequalities) :]
        ]
    )
    coeff_constraints = (
        comb(nvars + represented_degree, represented_degree) if nvars else 1
    )

    if main_dim <= 10 and scalar_decisions <= 100:
        cost_class = "tiny"
    elif main_dim <= 25 and scalar_decisions <= 500:
        cost_class = "small"
    elif main_dim <= 45 and scalar_decisions <= 1500 and coeff_constraints <= 1000:
        cost_class = "moderate"
    else:
        cost_class = "large"
    return SOSCostEstimate(
        True,
        nvars,
        target_degree,
        order,
        main_dim,
        psd_blocks,
        scalar_decisions,
        coeff_constraints,
        cost_class,
        "dense combinatorial estimate; sparse/chordal compilation may reduce the realized SDP",
    )


def plan_sos_search(
    polynomial: sp.Expr,
    variables: sp.Symbol | Sequence[sp.Symbol] | None = None,
    *,
    inequalities: Sequence[sp.Expr] = (),
    equalities: Sequence[sp.Expr] = (),
    constraints: object = None,
    ineq_mult_degree: int = 1,
    eq_mult_degree: int = 1,
    max_gram_dimension: int = 45,
    max_scalar_variables: int = 1500,
    max_constraints: int = 1000,
) -> SOSSearchPlan:
    """Decide whether SOS is cheap enough to launch automatically.

    A ``False`` decision means "defer to cheaper exact methods first", not that
    SOS is mathematically inapplicable. This is the hook semialg can use before
    its Zeng/ARS/CAD/SOS dispatcher commits to numerical SDP work.
    """
    estimate = estimate_sos_cost(
        polynomial,
        variables,
        inequalities=inequalities,
        equalities=equalities,
        constraints=constraints,
        ineq_mult_degree=ineq_mult_degree,
        eq_mult_degree=eq_mult_degree,
    )
    if not estimate.applicable:
        return SOSSearchPlan(False, estimate, estimate.reason)
    if estimate.main_gram_dimension > max_gram_dimension:
        return SOSSearchPlan(
            False,
            estimate,
            "estimated Gram dimension exceeds the automatic SOS budget; try cheaper exact routes first",
        )
    if estimate.scalar_decision_vars > max_scalar_variables:
        return SOSSearchPlan(
            False,
            estimate,
            "estimated SDP decision-variable count exceeds the automatic SOS budget; try cheaper exact routes first",
        )
    if estimate.coefficient_constraints > max_constraints:
        return SOSSearchPlan(
            False,
            estimate,
            "estimated coefficient-matching system exceeds the automatic SOS budget; try cheaper exact routes first",
        )
    return SOSSearchPlan(
        True,
        estimate,
        f"SOS search is within the automatic {estimate.cost_class} cost budget",
    )


def search_sos_certificate(
    polynomial: sp.Expr,
    variables: sp.Symbol | Sequence[sp.Symbol] | None = None,
    *,
    inequalities: Sequence[sp.Expr] = (),
    equalities: Sequence[sp.Expr] = (),
    constraints: object = None,
    ineq_mult_degree: int = 1,
    eq_mult_degree: int = 1,
    planner: bool = True,
    force: bool = False,
    solver_config: SDPSolverConfig | None = None,
) -> SOSCertificateSearchResult:
    """Search for an exact SOS certificate for ``polynomial >= 0``.

    The returned certificate represents
    ``polynomial = sigma0 + sum(sigma_i*g_i) + sum(tau_j*h_j)``.
    No numerical SDP status, approximate Gram matrix, or residual tolerance is
    ever promoted to ``certified=True``.
    """
    polynomial, variables, inequalities, equalities = _normalize_sos_domain(
        polynomial, variables, inequalities, equalities, constraints
    )
    plan = plan_sos_search(
        polynomial,
        variables,
        inequalities=inequalities,
        equalities=equalities,
        ineq_mult_degree=ineq_mult_degree,
        eq_mult_degree=eq_mult_degree,
    )
    if planner and not force and not plan.launch:
        return SOSCertificateSearchResult(
            False,
            False,
            "deferred",
            polynomial,
            variables,
            plan,
            None,
            None,
            plan.reason,
        )
    if not plan.estimate.applicable:
        return SOSCertificateSearchResult(
            False,
            False,
            "unsupported",
            polynomial,
            variables,
            plan,
            None,
            None,
            plan.reason,
        )

    degree = plan.estimate.relaxation_order or 0
    try:
        compilation: SOSCompilation = compile_sos_relaxation(
            polynomial,
            variables,
            degree_main=degree,
            inequalities=inequalities,
            equalities=equalities,
            fixed_gamma=sp.Integer(0),
            ineq_mult_degree=ineq_mult_degree,
            eq_mult_degree=eq_mult_degree,
        )
        sdp_result = solve_sdp(
            compilation.sdp_problem,
            solver_config or SDPSolverConfig(max_iter=120, tol=1e-8),
        )
        certificate = _decode_certificate(compilation, sdp_result)
    except (
        ValueError,
        TypeError,
        ArithmeticError,
        RuntimeError,
        FloatingPointError,
    ) as exc:
        return SOSCertificateSearchResult(
            True,
            False,
            "failed",
            polynomial,
            variables,
            plan,
            None,
            None,
            f"SOS/SDP search failed: {exc}",
        )

    certified = certificate.verified
    basis, exact_gram, numeric_gram, terms = _main_block_data(
        compilation, sdp_result, certificate
    )
    raw = SOSResult(
        certified,
        polynomial,
        variables,
        degree,
        basis,
        exact_gram,
        certificate.residual,
        numeric_gram,
        terms,
        "Recovered exact constrained SOS certificate."
        if certified
        else "Numerical SDP search did not yield an exact SOS certificate.",
        sdp_result.status,
        compilation,
        certificate,
    )
    return SOSCertificateSearchResult(
        True,
        certified,
        "certified" if certified else "uncertified",
        polynomial,
        variables,
        plan,
        certificate if certified else None,
        raw,
        raw.message,
    )
