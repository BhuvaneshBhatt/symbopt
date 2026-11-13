from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import mpmath as mp
import numpy as np
import sympy as sp
from sympy.polys.polyerrors import PolynomialError

from .exact_refinement import reconstruct_exact_vector
from .semidefinite import SDPSolverConfig, solve_sdp
from .sos_program import SOSCompilation, compile_sos_relaxation, monomial_basis


@dataclass
class SOSCertificate:
    """An exact polynomial identity and its nonnegative Gram-block premises.

    The statement is ``polynomial - lower_bound = normalization_scale * rhs``.
    The right side is the sum of Gram forms times ``gram_multipliers`` and
    equality multiples. It is a lower-bound proof where every Gram multiplier
    is nonnegative and every equality vanishes. Stored residual and PSD fields
    are summaries; :meth:`verify` reconstructs their mathematical obligations.
    """

    gram_matrices: list[sp.Matrix]
    eq_multiplier_polys: list[sp.Expr]
    residual: sp.Expr
    exact_psd_checks: list[bool | None]
    exact_residual_zero: bool
    polynomial: sp.Expr
    variables: tuple[sp.Symbol, ...]
    gram_bases: tuple[tuple[sp.Expr, ...], ...]
    gram_multipliers: tuple[sp.Expr, ...]
    equalities: tuple[sp.Expr, ...]
    lower_bound: sp.Expr = sp.S.Zero
    normalization_scale: sp.Expr = sp.S.One

    def pretty(self) -> str:
        """Describe the bound, exact residual and Gram positivity summaries."""
        return "\n".join(
            [
                f"lower_bound: {self.lower_bound}",
                f"exact_residual_zero: {self.exact_residual_zero}",
                f"exact_psd_checks: {self.exact_psd_checks}",
                f"residual: {sp.sstr(self.residual)}",
            ]
        )

    def verify(
        self, *, polynomial=None, lower_bound=None, inequalities=None, equalities=None
    ) -> bool:
        """Rebuild the identity and PSD checks, optionally binding source premises.

        Extra caller constraints may restrict the certified domain. Every
        multiplieir and equality used by this proof must occur among the supplied
        premises. No numerical solver or recovery routine participates in replay.
        """
        if type(self) is not SOSCertificate:
            return False
        try:
            variables = tuple(self.variables)
            if (
                not all(isinstance(v, sp.Symbol) for v in variables)
                or any(v.is_real is False for v in variables)
                or len(set(variables)) != len(variables)
                or type(self.exact_residual_zero) is not bool
            ):
                return False
            count = len(self.gram_matrices)
            if not (
                count
                == len(self.gram_bases)
                == len(self.gram_multipliers)
                == len(self.exact_psd_checks)
            ):
                return False
            if len(self.eq_multiplier_polys) != len(self.equalities):
                return False
            scale = _exact_expression(self.normalization_scale)
            if scale.is_positive is not True or scale.is_finite is not True:
                return False
            target = _exact_expression(self.polynomial)
            bound = _exact_expression(self.lower_bound)
            if bound.free_symbols.intersection(variables):
                return False
            if (
                polynomial is not None
                and sp.expand(target - _exact_expression(polynomial)) != 0
            ):
                return False
            if lower_bound is not None and bound != _exact_expression(lower_bound):
                return False
            multipliers = tuple(_exact_expression(g) for g in self.gram_multipliers)
            equations = tuple(_exact_expression(h) for h in self.equalities)
            if inequalities is not None:
                allowed = {_exact_expression(g) for g in inequalities} | {sp.S.One}
                if any(g not in allowed for g in multipliers):
                    return False
            if equalities is not None and not set(equations).issubset(
                {_exact_expression(h) for h in equalities}
            ):
                return False
            rhs = sp.S.Zero
            actual_checks = []
            expressions = [target, bound, *multipliers, *equations]
            for gram, basis, multiplier in zip(
                self.gram_matrices, self.gram_bases, multipliers
            ):
                matrix = sp.ImmutableMatrix(gram).applyfunc(_exact_expression)
                basis = sp.ImmutableMatrix([_exact_expression(v) for v in basis])
                if not basis.rows or matrix.shape != (basis.rows, basis.rows):
                    return False
                if any(value.free_symbols.intersection(variables) for value in matrix):
                    return False
                if matrix != matrix.T:
                    return False
                check = _checked_gram(matrix)
                actual_checks.append(check)
                expressions.extend((*matrix, *basis))
                rhs += multiplier * (basis.T * matrix * basis)[0]
            for tau, equation in zip(self.eq_multiplier_polys, equations):
                tau = _exact_expression(tau)
                expressions.append(tau)
                rhs += tau * equation
            for expression in expressions:
                coefficients = (
                    sp.Poly(expression, *variables).coeffs()
                    if variables
                    else [expression]
                )
                if any(
                    c.is_real is not True or c.is_finite is not True
                    for c in coefficients
                ):
                    return False
            residual = sp.expand(target - bound - scale * rhs)
            return (
                self.exact_residual_zero is True
                and residual == 0
                and _exact_expression(self.residual) == residual
                and all(
                    actual is True and stated is True
                    for actual, stated in zip(actual_checks, self.exact_psd_checks)
                )
            )
        except (
            AttributeError,
            TypeError,
            ValueError,
            ArithmeticError,
            NotImplementedError,
            PolynomialError,
        ):
            return False

    @property
    def verified(self) -> bool:
        """Whether replay proves the stored polynomial lower bound on its domain."""
        return self.verify()


def _exact_expression(expression):
    expression = sp.sympify(expression)
    return expression.xreplace({v: sp.Rational(v) for v in expression.atoms(sp.Float)})


@lru_cache(maxsize=256)
def _checked_gram(matrix):
    return _exact_psd_check(matrix)


def _bound_certificate(compilation, grams, taus, residual, checks, zero):
    """Bind recovered Gram data to the polynomial and domain used in compilation."""
    return SOSCertificate(
        grams,
        taus,
        residual,
        checks,
        zero,
        polynomial=compilation.polynomial,
        variables=tuple(compilation.variables),
        gram_bases=tuple(tuple(block.basis) for block in compilation.gram_blocks),
        gram_multipliers=tuple(
            sp.S.One if block.multiplier is None else block.multiplier
            for block in compilation.gram_blocks
        ),
        equalities=tuple(block.polynomial for block in compilation.equality_blocks),
        lower_bound=compilation.fixed_gamma,
        normalization_scale=compilation.normalization_scale,
    )


@dataclass
class SOSResult:
    """Result of an SOS decomposition attempt, including exact/numeric Gram data and certificate metadata."""

    success: bool
    polynomial: sp.Expr
    variables: list[sp.Symbol]
    degree: int
    monomial_basis: list[sp.Expr]
    gram_matrix: sp.Matrix
    exact_residual: sp.Expr
    numeric_gram: np.ndarray | None
    certificate_terms: list[sp.Expr]
    message: str
    sdp_status: str | None = None
    compilation: SOSCompilation | None = None
    certificate: SOSCertificate | None = None


@dataclass
class SOSRelaxationResult:
    """Result of a constrained SOS relaxation.

    ``success`` means the numerical relaxation produced a usable iterate.
    ``certified`` is stronger: it requires an exact polynomial identity and
    exact PSD verification. ``lower_bound`` is populated only for certified
    bounds; approximate solver values are reported through
    ``lower_bound_numeric`` and ``status``.
    """

    success: bool
    lower_bound: sp.Expr | None
    lower_bound_numeric: float | None
    sos_result: SOSResult | None
    multipliers: dict[sp.Symbol, sp.Expr]
    message: str
    compilation: SOSCompilation | None = None
    certified: bool = False
    status: str = "unknown"


def _project_psd(M: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, eps)
    P = (V * w) @ V.T
    return 0.5 * (P + P.T)


def _nsimplify_matrix(M: np.ndarray, tol: float = 1e-10) -> sp.Matrix:
    return sp.Matrix(
        [
            [
                sp.nsimplify(
                    0.0 if abs(float(M[i, j])) < tol else float(M[i, j]), rational=True
                )
                for j in range(M.shape[1])
            ]
            for i in range(M.shape[0])
        ]
    )


def _symbolic_zero_check(
    expr: sp.Expr, vars: Sequence[sp.Symbol], tol: float = 1e-10
) -> bool:
    expr = sp.expand(expr)
    if expr == 0:
        return True
    try:
        poly = sp.Poly(expr, *vars)
        return all(abs(complex(sp.N(c, 50))) <= tol for c in poly.coeffs())
    except (PolynomialError, TypeError, ValueError, ArithmeticError):
        return False


def _exact_psd_check(M: sp.Matrix) -> bool | None:
    try:
        v = M.is_positive_semidefinite
        if v is not None:
            return bool(v)
    except (TypeError, ValueError, ArithmeticError, NotImplementedError):
        pass
    try:
        eigenvalues = M.eigenvals()
        if eigenvalues:
            signs = [sp.simplify(value).is_nonnegative for value in eigenvalues]
            if all(sign is True for sign in signs):
                return True
            if any(sp.simplify(value).is_negative is True for value in eigenvalues):
                return False
    except (TypeError, ValueError, ArithmeticError, NotImplementedError):
        pass
    try:
        n = M.rows
        if n <= 6:
            import itertools

            for k in range(1, n + 1):
                for idx in itertools.combinations(range(n), k):
                    minor = sp.simplify(M.extract(idx, idx).det())
                    if minor.is_negative is True:
                        return False
                    if minor.is_nonnegative is not True:
                        return None
            return True
    except (TypeError, ValueError, ArithmeticError, NotImplementedError):
        pass
    return None


def _certificate_terms(Q_num: np.ndarray, basis: Sequence[sp.Expr]) -> list[sp.Expr]:
    w, v = np.linalg.eigh(0.5 * (Q_num + Q_num.T))
    terms = []
    for lam, vec in zip(w, v.T):
        if lam <= 1e-12:
            continue
        lin = sp.Integer(0)
        for coeff, mon in zip(np.sqrt(lam) * vec, basis):
            if abs(float(coeff)) > 1e-12:
                lin += sp.Float(float(coeff)) * mon
        terms.append(sp.expand(lin))
    return terms


def _rational_approx(value: float, denominator: int) -> sp.Rational:
    from fractions import Fraction

    fraction = Fraction(str(float(value))).limit_denominator(denominator)
    return sp.Rational(fraction.numerator, fraction.denominator)


def _certificate_expression(
    compilation: SOSCompilation,
    gram_matrices: Sequence[sp.Matrix],
    tau_polys: Sequence[sp.Expr],
) -> sp.Expr:
    expression = sp.Integer(0)
    for block, matrix in zip(compilation.gram_blocks, gram_matrices):
        basis = sp.Matrix(block.basis)
        term = (basis.T * matrix * basis)[0]
        if block.multiplier is not None:
            term = sp.expand(block.multiplier * term)
        expression += term
    for tau, block in zip(tau_polys, compilation.equality_blocks):
        expression += sp.expand(tau * block.polynomial)
    return sp.expand(expression)


def _numeric_tau_coeffs(compilation: SOSCompilation, sdp_result) -> list[float]:
    if not compilation.equality_blocks:
        return []
    values = (
        np.zeros(len(compilation.scalar_names), dtype=float)
        if sdp_result.z is None
        else np.asarray(sdp_result.z, dtype=float)
    )
    coeffs = []
    index = 0
    for block in compilation.equality_blocks:
        for _ in block.basis:
            pos = values[index] if index < len(values) else 0.0
            index += 1
            neg = values[index] if index < len(values) else 0.0
            index += 1
            coeffs.append(float(pos - neg))
    return coeffs


def _fit_free_params(
    solution: Sequence[sp.Expr],
    free: Sequence[sp.Symbol],
    hint: Sequence[float],
    digits: int,
) -> dict[sp.Symbol, float]:
    """Fit affine free parameters to a numerical certificate in high precision."""
    if not free:
        return {}
    zero_subs = {symbol: sp.Integer(0) for symbol in free}
    base_expr = [sp.N(value.subs(zero_subs), digits) for value in solution]
    col_exprs = []
    for free_symbol in free:
        unit = dict(zero_subs)
        unit[free_symbol] = sp.Integer(1)
        col_exprs.append(
            [
                sp.N(value.subs(unit) - value.subs(zero_subs), digits)
                for value in solution
            ]
        )

    with mp.workdps(digits):
        base = mp.matrix([mp.mpf(str(value)) for value in base_expr])
        matrix = mp.matrix(
            [
                [mp.mpf(str(col_exprs[col][row])) for col in range(len(free))]
                for row in range(len(solution))
            ]
        )
        target = mp.matrix([mp.mpf(repr(float(value))) for value in hint])
        rhs = target - base
        try:
            fitted, _ = mp.qr_solve(matrix, rhs)
        except (ValueError, ZeroDivisionError, ArithmeticError):
            normal = matrix.T * matrix
            scale = mp.power(10, -(digits // 2))
            normal += scale * mp.eye(len(free))
            fitted = mp.lu_solve(normal, matrix.T * rhs)
        return {symbol: float(fitted[index]) for index, symbol in enumerate(free)}


def _recover_certificate(
    compilation: SOSCompilation,
    sdp_result,
    denominator_schedule=(100, 1000, 10000, 100000, 1000000),
    precision_schedule=(50, 80, 120),
) -> SOSCertificate | None:
    """Recover an exact affine SOS certificate near a numerical SDP point.

    The coefficient identity is solved exactly. Underdetermined affine
    parameters are fitted to the numerical Gram/multiplier point and then
    rationalized with increasingly large denominator budgets. Every candidate
    is accepted only after exact residual and exact PSD verification.
    """
    unknowns = []
    gram_templates = []
    hint = []

    for block_index, block in enumerate(compilation.gram_blocks):
        size = len(block.basis)
        matrix = sp.MutableDenseMatrix.zeros(size, size)
        numeric = _project_psd(sdp_result.X_blocks[block_index])
        for row in range(size):
            for col in range(row, size):
                symbol = sp.Dummy(f"q_{block_index}_{row}_{col}", real=True)
                unknowns.append(symbol)
                hint.append(float(numeric[row, col]))
                matrix[row, col] = symbol
                matrix[col, row] = symbol
        gram_templates.append(sp.Matrix(matrix))

    tau_templates = []
    tau_hint = _numeric_tau_coeffs(compilation, sdp_result)
    tau_hint_index = 0
    for block_index, block in enumerate(compilation.equality_blocks):
        expression = sp.Integer(0)
        for term_index, basis_term in enumerate(block.basis):
            symbol = sp.Dummy(f"tau_{block_index}_{term_index}", real=True)
            unknowns.append(symbol)
            hint.append(tau_hint[tau_hint_index])
            tau_hint_index += 1
            expression += symbol * basis_term
        tau_templates.append(sp.expand(expression))

    # Solve the coefficient identity exactly before using the numerical SDP
    # point. This keeps rational recovery constrained to the true affine
    # certificate space instead of rationalizing Gram entries independently.
    target = sp.expand(
        (compilation.polynomial - _exact_expression(compilation.fixed_gamma))
        / compilation.normalization_scale
    )
    expression = _certificate_expression(compilation, gram_templates, tau_templates)
    residual = sp.expand(target - expression)
    try:
        poly = sp.Poly(residual, *compilation.variables)
        equations = [sp.expand(coeff) for coeff in poly.coeffs()]
        matrix, rhs = sp.linear_eq_to_matrix(equations, unknowns)
        solution_set = sp.linsolve((matrix, rhs), unknowns)
        if solution_set is sp.EmptySet:
            return None
        solution = list(next(iter(solution_set)))
    except (
        PolynomialError,
        TypeError,
        ValueError,
        ArithmeticError,
        NotImplementedError,
    ):
        return None

    unknown_set = set(unknowns)
    free = sorted(
        set()
        .union(*(value.free_symbols for value in solution))
        .intersection(unknown_set),
        key=lambda symbol: symbol.name,
    )

    # Precision controls the projection onto the affine solution space;
    # denominator growth controls only the subsequent rational reconstruction.
    # Exact identity and PSD checks below remain the final acceptance criteria.
    for digits in precision_schedule:
        try:
            free_values = _fit_free_params(solution, free, hint, digits)
        except (TypeError, ValueError, ArithmeticError, ZeroDivisionError):
            free_values = {symbol: 0.0 for symbol in free}

        proposals = []
        free_order = tuple(free)
        if free_order:
            values = [free_values[symbol] for symbol in free_order]
            proposals.extend(
                dict(zip(free_order, exact_values))
                for exact_values, _ in reconstruct_exact_vector(
                    values, common_number_field=False
                )
            )
        else:
            proposals.append({})
        for denominator in denominator_schedule:
            proposals.append(
                {
                    symbol: _rational_approx(value, denominator)
                    for symbol, value in free_values.items()
                }
            )

        seen = set()
        for substitutions in proposals:
            key = tuple(
                (str(symbol), sp.srepr(value))
                for symbol, value in sorted(
                    substitutions.items(), key=lambda item: str(item[0])
                )
            )
            if key in seen:
                continue
            seen.add(key)
            exact_values = [
                sp.simplify(value.subs(substitutions)) for value in solution
            ]
            value_map = dict(zip(unknowns, exact_values))
            grams = [
                sp.Matrix(template.subs(value_map)).applyfunc(sp.simplify)
                for template in gram_templates
            ]
            taus = [sp.expand(template.subs(value_map)) for template in tau_templates]
            exact_expr = _certificate_expression(compilation, grams, taus)
            exact_residual = sp.expand(
                compilation.polynomial
                - _exact_expression(compilation.fixed_gamma)
                - compilation.normalization_scale * exact_expr
            )
            if exact_residual != 0:
                continue
            checks = [_checked_gram(sp.ImmutableMatrix(matrix)) for matrix in grams]
            if all(check is True for check in checks):
                certificate = _bound_certificate(
                    compilation, grams, taus, exact_residual, checks, True
                )
                if certificate.verified:
                    return certificate

    return None


def _decode_certificate(compilation: SOSCompilation, sdp_result) -> SOSCertificate:
    recovered = _recover_certificate(compilation, sdp_result)
    if recovered is not None:
        return recovered

    gram_num = [_project_psd(matrix) for matrix in sdp_result.X_blocks]
    gram_sym = [_nsimplify_matrix(matrix) for matrix in gram_num]
    tau_polys = []
    tau_coeffs = _numeric_tau_coeffs(compilation, sdp_result)
    index = 0
    for block in compilation.equality_blocks:
        expression = sp.Integer(0)
        for basis_term in block.basis:
            expression += sp.nsimplify(tau_coeffs[index]) * basis_term
            index += 1
        tau_polys.append(sp.expand(expression))

    expression = _certificate_expression(compilation, gram_sym, tau_polys)
    residual = sp.expand(
        compilation.polynomial
        - compilation.fixed_gamma
        - compilation.normalization_scale * expression
    )
    checks = [_checked_gram(sp.ImmutableMatrix(matrix)) for matrix in gram_sym]
    return _bound_certificate(
        compilation,
        gram_sym,
        tau_polys,
        residual,
        checks,
        residual == 0,
    )


def _main_block_data(compilation: SOSCompilation, sdp_result, certificate):
    count = compilation.main_gram_count
    bases = [block.basis for block in compilation.gram_blocks[:count]]
    exact_blocks = certificate.gram_matrices[:count]
    numeric_blocks = [_project_psd(matrix) for matrix in sdp_result.X_blocks[:count]]
    basis = [term for block_basis in bases for term in block_basis]
    exact = sp.diag(*exact_blocks) if exact_blocks else sp.Matrix()
    numeric = None
    if numeric_blocks:
        sizes = [matrix.shape[0] for matrix in numeric_blocks]
        total = sum(sizes)
        numeric = np.zeros((total, total), dtype=float)
        offset = 0
        for matrix in numeric_blocks:
            size = matrix.shape[0]
            numeric[offset : offset + size, offset : offset + size] = matrix
            offset += size
    terms = []
    for numeric_matrix, block_basis in zip(numeric_blocks, bases):
        terms.extend(_certificate_terms(numeric_matrix, block_basis))
    return basis, exact, numeric, terms


def _exact_square_result(
    poly: sp.Expr,
    vars: Sequence[sp.Symbol],
    degree: int,
) -> SOSResult | None:
    """Return an exact Gram certificate for an exact polynomial square.

    Factorization is used rather than syntactic ``Pow`` matching so that
    constant square factors such as ``36*(x - 1)**2`` are recognised as well.
    """
    if poly == 0:
        base = sp.Integer(0)
    else:
        try:
            coeff, factors = sp.factor_list(poly, *vars)
        except (PolynomialError, TypeError, ValueError):
            return None
        if coeff.is_nonnegative is not True:
            return None
        if any(exponent % 2 for _, exponent in factors):
            return None
        base = sp.sqrt(coeff)
        for factor, exponent in factors:
            base *= factor ** (exponent // 2)
        base = sp.expand(base)
        if sp.expand(base**2 - poly) != 0:
            return None

    if not base.is_polynomial(*vars):
        return None
    try:
        base_degree = sp.Poly(base, *vars).total_degree()
        if base_degree is not sp.S.NegativeInfinity and base_degree > degree:
            return None
    except (PolynomialError, TypeError, ValueError):
        return None

    basis = monomial_basis(vars, degree)
    base_poly = sp.Poly(base, *vars)
    coeffs = sp.Matrix([base_poly.coeff_monomial(term) for term in basis])
    gram = sp.simplify(coeffs * coeffs.T)
    basis_vec = sp.Matrix(basis)
    residual = sp.expand(poly - (basis_vec.T * gram * basis_vec)[0])
    cert = SOSCertificate(
        gram_matrices=[gram],
        eq_multiplier_polys=[],
        residual=residual,
        exact_psd_checks=[True],
        exact_residual_zero=(residual == 0),
        polynomial=poly,
        variables=tuple(vars),
        gram_bases=(tuple(basis),),
        gram_multipliers=(sp.S.One,),
        equalities=(),
    )
    numeric = np.asarray(gram.evalf(), dtype=float)
    return SOSResult(
        success=cert.verified,
        polynomial=poly,
        variables=list(vars),
        degree=degree,
        monomial_basis=basis,
        gram_matrix=gram,
        exact_residual=residual,
        numeric_gram=numeric,
        certificate_terms=[base],
        message="Recovered exact polynomial-square certificate.",
        sdp_status="exact_square",
        compilation=None,
        certificate=cert,
    )


def sos_decompose(
    poly: sp.Expr,
    vars: Sequence[sp.Symbol],
    degree: int | None = None,
    solver_config: SDPSolverConfig | None = None,
) -> SOSResult:
    """Attempt an exact sum-of-squares decomposition of a polynomial.

    A numerical SDP supplies a Gram hint. The recovery layer then solves the
    coefficient identity exactly and searches increasingly fine rational
    reconstructions. ``success`` is true only when the residual vanishes exactly
    and every recovered Gram block is exactly positive semidefinite."""
    poly = sp.expand(poly)
    vars = list(vars)

    # The zero polynomial has a canonical trivial SOS certificate.  Handle it
    # before degree inference, factorization, cache lookup, or numerical SDP so
    # the result is independent of backend and test-suite state.
    if poly == 0:
        basis = [sp.Integer(1)]
        gram = sp.zeros(1)
        cert = SOSCertificate(
            gram_matrices=[gram],
            eq_multiplier_polys=[],
            residual=sp.Integer(0),
            exact_psd_checks=[True],
            exact_residual_zero=True,
            polynomial=sp.S.Zero,
            variables=tuple(vars),
            gram_bases=((sp.S.One,),),
            gram_multipliers=(sp.S.One,),
            equalities=(),
        )
        return SOSResult(
            success=True,
            polynomial=sp.Integer(0),
            variables=vars,
            degree=0,
            monomial_basis=basis,
            gram_matrix=gram,
            exact_residual=sp.Integer(0),
            numeric_gram=np.zeros((1, 1), dtype=float),
            certificate_terms=[sp.Integer(0)],
            message="Zero polynomial is a trivial exact SOS.",
            sdp_status="exact_zero",
            compilation=None,
            certificate=cert,
        )

    total_degree = sp.Poly(poly, *vars).total_degree()
    if degree is None:
        if total_degree % 2 != 0:
            return SOSResult(
                False,
                poly,
                vars,
                total_degree,
                [],
                sp.Matrix(),
                poly,
                None,
                [],
                "Polynomial has odd total degree => it cannot be SOS.",
            )
        degree = total_degree // 2

    exact_square = _exact_square_result(poly, vars, degree)
    if exact_square is not None:
        return exact_square

    compilation = compile_sos_relaxation(
        poly,
        vars,
        degree_main=degree,
        inequalities=(),
        equalities=(),
        fixed_gamma=0,
    )
    sdp_res = solve_sdp(
        compilation.sdp_problem,
        solver_config or SDPSolverConfig(max_iter=120, tol=1e-8),
    )
    cert = _decode_certificate(compilation, sdp_res)
    basis, exact_gram, numeric_gram, terms = _main_block_data(
        compilation, sdp_res, cert
    )
    success = cert.verified
    return SOSResult(
        success,
        poly,
        vars,
        degree,
        basis,
        exact_gram,
        cert.residual,
        numeric_gram,
        terms,
        (
            "Recovered exact SOS certificate."
            if success
            else "SOS feasibility SDP did not yield an exact certificate."
        ),
        sdp_res.status,
        compilation,
        cert,
    )


def sos_lower_bound(
    poly: sp.Expr,
    vars: Sequence[sp.Symbol],
    search_lo: float = -100.0,
    search_hi: float = 100.0,
    max_iter: int = 12,
    solver_config: SDPSolverConfig | None = None,
):
    """Search for a global SOS lower bound of an unconstrained even-degree polynomial."""
    vars = list(vars)
    total_degree = sp.Poly(poly, *vars).total_degree()
    if total_degree % 2 != 0:
        raise ValueError("Only even-degree polynomials are supported.")
    if total_degree == 2:
        try:
            grad = [sp.diff(poly, v) for v in vars]
            sol = sp.solve([sp.Eq(g, 0) for g in grad], vars, dict=True)
            if sol:
                val = sp.simplify(poly.subs(sol[0]))
                dec = sos_decompose(
                    sp.expand(poly - val), vars, degree=1, solver_config=solver_config
                )
                return sp.simplify(val), dec
        except (TypeError, ValueError, ArithmeticError, NotImplementedError):
            pass
    degree = total_degree // 2
    lo, hi = search_lo, search_hi
    best_gamma = lo
    warm = None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        compilation = compile_sos_relaxation(
            poly,
            vars,
            degree_main=degree,
            inequalities=(),
            equalities=(),
            fixed_gamma=sp.Rational(mid),
        )
        sdp_res = solve_sdp(
            compilation.sdp_problem,
            solver_config or SDPSolverConfig(max_iter=100, tol=1e-8),
            warm_start=warm,
        )
        cert = _decode_certificate(compilation, sdp_res)
        feasible = (
            sdp_res.status in {"optimal", "approximate", "max_iter"} and cert.verified
        )
        if feasible:
            best_gamma = mid
            lo = mid
            warm = {"X_blocks": sdp_res.X_blocks, "z": sdp_res.z}
        else:
            hi = mid
    return sp.Rational(best_gamma), sos_decompose(
        sp.expand(poly - sp.Rational(best_gamma)),
        vars,
        degree=degree,
        solver_config=solver_config,
    )


def sos_poly_relaxation(
    poly: sp.Expr,
    vars: Sequence[sp.Symbol],
    inequalities: Sequence[sp.Expr] = (),
    equalities: Sequence[sp.Expr] = (),
    search_lo: float = -100.0,
    search_hi: float = 100.0,
    max_iter: int = 12,
    ineq_mult_degree: int = 1,
    eq_mult_degree: int = 1,
    solver_config: SDPSolverConfig | None = None,
) -> SOSRelaxationResult:
    """Compute a constrained polynomial-multiplier SOS lower-bound relaxation.

    Numerical feasibility can guide bisection and warm starts. The result is marked
    certified only after exact certificate recovery succeeds."""
    vars = list(vars)
    poly = sp.expand(poly)
    total_degree = sp.Poly(poly, *vars).total_degree()
    if total_degree % 2 != 0:
        return SOSRelaxationResult(
            False,
            None,
            None,
            None,
            {},
            "Only even-degree polynomials are supported.",
            status="unsupported",
        )
    degree = total_degree // 2
    lo, hi = search_lo, search_hi
    best_gamma = lo
    best_pack = None
    warm = None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        compilation = compile_sos_relaxation(
            poly,
            vars,
            degree_main=degree,
            inequalities=inequalities,
            equalities=equalities,
            fixed_gamma=sp.Rational(mid),
            ineq_mult_degree=ineq_mult_degree,
            eq_mult_degree=eq_mult_degree,
        )
        sdp_res = solve_sdp(
            compilation.sdp_problem,
            solver_config or SDPSolverConfig(max_iter=120, tol=1e-8),
            warm_start=warm,
        )
        cert = _decode_certificate(compilation, sdp_res)
        exact_feasible = cert.verified
        numerical_feasible = (
            sdp_res.status in {"optimal", "approximate", "max_iter"}
            and all(check is not False for check in cert.exact_psd_checks)
            and sdp_res.primal_residual_norm <= 1e-5
        )
        feasible = exact_feasible or numerical_feasible
        if feasible:
            best_gamma = mid
            lo = mid
            best_pack = (compilation, sdp_res, cert)
            warm = {"X_blocks": sdp_res.X_blocks, "z": sdp_res.z}
        else:
            hi = mid
    if best_pack is None:
        return SOSRelaxationResult(
            False,
            None,
            None,
            None,
            {},
            "No feasible coupled SOS relaxation found.",
            status="infeasible_or_failed",
        )
    compilation, sdp_res, cert = best_pack
    main_basis, main_exact, main_numeric, main_terms = _main_block_data(
        compilation, sdp_res, cert
    )
    exact_feasible = cert.verified
    sos_result = SOSResult(
        exact_feasible,
        poly,
        vars,
        degree,
        main_basis,
        main_exact,
        cert.residual,
        main_numeric,
        main_terms,
        (
            "Computed exact coupled SOS certificate."
            if exact_feasible
            else "Computed approximate coupled SOS relaxation."
        ),
        sdp_res.status,
        compilation,
        cert,
    )
    multipliers = {}
    for offset, gb in enumerate(
        compilation.gram_blocks[compilation.main_gram_count :], start=1
    ):
        gram_index = compilation.main_gram_count + offset - 1
        zmat = sp.Matrix(gb.basis)
        multipliers[sp.Symbol(f"sigma_{offset}", real=True)] = (
            zmat.T * cert.gram_matrices[gram_index] * zmat
        )[0]
    for i, tau in enumerate(cert.eq_multiplier_polys, start=1):
        multipliers[sp.Symbol(f"tau_{i}", real=True)] = tau
    return SOSRelaxationResult(
        True,
        sp.Rational(best_gamma) if exact_feasible else None,
        float(best_gamma),
        sos_result,
        multipliers,
        (
            "Computed exact polynomial-multiplier SOS lower bound."
            if exact_feasible
            else "Computed numerically feasible SOS relaxation; no exact lower-bound certificate was recovered."
        ),
        compilation,
        exact_feasible,
        "certified" if exact_feasible else "numerically_feasible",
    )


def sos_scalar_relaxation(
    poly: sp.Expr,
    vars: Sequence[sp.Symbol],
    inequalities: Sequence[sp.Expr],
    search_lo: float = -100.0,
    search_hi: float = 100.0,
    max_iter: int = 12,
    multiplier_degree: int = 1,
) -> SOSRelaxationResult:
    """Convenience wrapper for an SOS relaxation with scalar/low-degree inequality multipliers."""
    return sos_poly_relaxation(
        poly,
        vars,
        inequalities=inequalities,
        equalities=(),
        search_lo=search_lo,
        search_hi=search_hi,
        max_iter=max_iter,
        ineq_mult_degree=multiplier_degree,
        eq_mult_degree=0,
    )
