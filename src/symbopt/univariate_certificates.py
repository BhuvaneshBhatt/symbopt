"""Assembly of replayable univariate global-minimum certificates."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .univariate_optimization import ComponentMinimumResult, finite_component_minimum


def _differentiate(f, x, order=1):
    from funcprops import normalize_real_expression

    return sp.diff(normalize_real_expression(f), x, order)


def _identity(left, right):
    from funcprops import verify_real_identity

    return verify_real_identity(left, right)


def _zero(value):
    return _identity(value, 0)


@dataclass(frozen=True)
class GlobalMinimumCertificate:
    """Bind a minimum value, location and uniqueness to sufficient global premises."""

    expression: sp.Expr
    variable: sp.Symbol
    point: sp.Expr
    value: sp.Expr
    unique: bool
    source: str
    domain: object = sp.S.Reals
    coercive: bool = False
    domain_record: object = None
    component_result: object = None

    @property
    def requested_domain(self):
        """The original requested source set, before source-definedness exclusions."""
        return (
            self.domain_record.requested
            if self.domain_record is not None
            else self.domain
        )

    @property
    def complete(self):
        """Whether the global minimum and any uniqueness claim replay successfully."""
        return self.verify()

    @property
    def certified(self):
        """Whether the stored mathematical evidence passes certificate replay."""
        return self.verify()

    @property
    def certificate(self):
        """The source-bound mathematical evidence represented by this record."""
        return self

    @property
    def scope(self):
        """Scope identifier for a global minimum on the effective source domain."""
        return "global-minimum-on-effective-domain"

    def verify(self):
        """Return whether the record satisfies its declared replay obligations."""

        try:
            return self._verify()
        except (
            AttributeError,
            TypeError,
            ValueError,
            NotImplementedError,
            ArithmeticError,
        ):
            return False

    def _verify(self):
        from funcprops import DomainRecord

        if (
            type(self) is not GlobalMinimumCertificate
            or type(self.unique) is not bool
            or type(self.coercive) is not bool
            or (
                self.domain_record is not None
                and type(self.domain_record) is not DomainRecord
            )
        ):
            return False
        if self.source == "finite-components":
            result = self.component_result
            return (
                type(result) is ComponentMinimumResult
                and result.verify()
                and result.attained
                and result.expression == self.expression
                and result.variable == self.variable
                and result.domain_record == self.domain_record
                and result.domain_record.effective == self.domain
                and result.minimizers.contains(self.point) is sp.S.true
                and _identity(result.infimum, self.value) is True
                and self.unique is result.unique
                and self.coercive is False
            )
        if self.component_result is not None:
            return False
        if self.domain_record is not None and (
            not self.domain_record.verify()
            or self.domain_record.expression != self.expression
            or self.domain_record.variable != self.variable
            or self.domain_record.effective != self.domain
        ):
            return False
        x = self.variable
        f = self.expression
        p = self.point
        D = self.domain
        if not _curvature_theorem_domain(f, x, D):
            return False
        if D.contains(p) is not sp.S.true:
            return False
        if _zero(_differentiate(f, x).subs(x, p)) is not True:
            return False
        if _identity(f.subs(x, p), self.value) is not True:
            return False
        d2 = sp.simplify(_differentiate(f, x, 2))
        try:
            lo = sp.calculus.util.minimum(d2, x, D)
        except (ValueError, TypeError, NotImplementedError, sp.PolynomialError):
            return False
        convex = lo is not None and lo is not -sp.oo and lo.is_nonnegative is True
        strict = lo is not None and lo is not -sp.oo and lo.is_positive is True
        if not convex:
            return False
        if self.unique and not strict:
            return False
        if self.coercive and D is not sp.S.Reals:
            return False
        if self.coercive and D is sp.S.Reals:
            try:
                if (
                    sp.limit(f, x, sp.oo) is not sp.oo
                    or sp.limit(f, x, -sp.oo) is not sp.oo
                ):
                    return False
            except (ValueError, NotImplementedError):
                return False
        return self.source in {
            "convex-stationary",
            "strict-convex-stationary",
            "coercive-convex-stationary",
        }


def _curvature_minimum(expr, var, domain=sp.S.Reals):
    f = sp.sympify(expr)
    x = sp.sympify(var)
    from funcprops import condition_to_real_set as condition_to_set

    D = domain if isinstance(domain, sp.Set) else condition_to_set(domain, x)
    if not _curvature_theorem_domain(f, x, D):
        return None
    d1 = _differentiate(f, x)
    d2 = sp.simplify(_differentiate(f, x, 2))
    try:
        lo = sp.calculus.util.minimum(d2, x, D)
    except (ValueError, TypeError, NotImplementedError, sp.PolynomialError):
        return None
    if lo is None or lo is -sp.oo or lo.is_nonnegative is not True:
        return None
    roots = sp.solveset(sp.Eq(d1, 0), x, domain=D)
    if isinstance(roots, sp.FiniteSet) and len(roots) == 1:
        p = next(iter(roots))
    else:
        try:
            proposals = sp.solve(d1, x)
        except (ValueError, TypeError, NotImplementedError):
            return None
        proposals = [
            v for v in proposals if v.is_real is True and D.contains(v) is sp.true
        ]
        if not proposals:
            return None
        p = proposals[0]
    strict = lo.is_positive is True
    coercive = False
    if D is sp.S.Reals:
        try:
            coercive = (
                sp.limit(f, x, sp.oo) is sp.oo and sp.limit(f, x, -sp.oo) is sp.oo
            )
        except (ValueError, NotImplementedError):
            coercive = False
    src = (
        "coercive-convex-stationary"
        if coercive
        else "strict-convex-stationary"
        if strict
        else "convex-stationary"
    )
    c = GlobalMinimumCertificate(
        f, x, p, sp.simplify(f.subs(x, p)), strict, src, D, coercive
    )
    return c if c.verify() else None


def _curvature_theorem_domain(expr, var, domain):
    """Prove real C2 interval premises without losing original source holes."""
    from funcprops import derivative_on_domain, track_domain

    if not isinstance(domain, sp.Interval):
        return False
    if expr.has(sp.Float) or domain.has(sp.Float):
        return False
    record = track_domain(expr, var, domain)
    return (
        record.complete
        and record.effective == domain
        and derivative_on_domain(expr, var, domain, 2) is not None
    )


def certify_global_minimum(expr, var, domain=sp.S.Reals, *, work_budget=None):
    """Certify an attained minimum by interval curvature or exact components.

    Return None when the available sufficient premises cannot establish an
    attained minimum. A positive work_budget runs discovery in a process and
    also returns None if the deadline expires before a certificate is ready.
    """
    if work_budget is not None:
        from .optimization_budget import BudgetExhausted, bounded_optimization

        result = bounded_optimization(
            "certify_global_minimum", expr, (var, domain), {}, work_budget
        )
        return None if isinstance(result, BudgetExhausted) else result
    from funcprops import track_domain

    result = finite_component_minimum(expr, var, domain)
    if result.complete:
        if not result.attained:
            return None
        minima = result.minimizers
        if isinstance(minima, sp.Union):
            minima = minima.args[0]
        if isinstance(minima, sp.FiniteSet):
            point = next(iter(minima))
        elif isinstance(minima, sp.Interval):
            point = (
                (minima.start + minima.end) / 2
                if minima.start.is_finite and minima.end.is_finite
                else minima.start + 1
                if minima.start.is_finite
                else minima.end - 1
                if minima.end.is_finite
                else sp.S.Zero
            )
        else:
            return None
        cert = GlobalMinimumCertificate(
            sp.sympify(expr),
            sp.sympify(var),
            point,
            result.infimum,
            result.unique,
            "finite-components",
            result.domain_record.effective,
            False,
            result.domain_record,
            result,
        )
        return cert if cert.verify() else None
    record = track_domain(expr, var, domain)
    # Curvature may not discard source holes while differentiating.
    if not record.complete:
        return None
    cert = _curvature_minimum(expr, var, record.effective)
    if cert is None:
        return None
    from dataclasses import replace

    cert = replace(cert, domain_record=record)
    return cert if cert.verify() else None
