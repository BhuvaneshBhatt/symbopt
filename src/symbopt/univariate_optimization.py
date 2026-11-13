"""Exact finite-component minimization using the shared rational enumeration route."""

from dataclasses import dataclass, field, replace

import sympy as sp

from .input_normalization import normalize_constraint_branches
from .rational_optimization import _compare, try_univariate_rational


@dataclass(frozen=True)
class OptimizationCandidate:
    """A checked value candidate, its location or limit, and source component."""

    point: object
    value: sp.Expr
    attained: bool
    domain: sp.Set
    kind: str


@dataclass(frozen=True)
class ComponentMinimumResult:
    """An exact rational infimum and minimizing set on a checked source domain."""

    expression: sp.Expr
    variable: sp.Symbol
    domain_record: object
    infimum: sp.Expr | None = None
    minimizers: sp.Set = sp.S.EmptySet
    candidates: tuple = ()
    complete: bool = False
    reason: str | None = None
    certified: bool = field(default=False, compare=False)

    @property
    def domain(self):
        """Return the effective source domain, including exclusions."""
        return self.domain_record.effective

    @property
    def attained(self):
        """Whether a completed enumeration has an actual minimizing point."""
        return self.complete is True and self.minimizers is not sp.S.EmptySet

    @property
    def unique(self):
        """Whether the minimizing set contains exactly one point."""
        return isinstance(self.minimizers, sp.FiniteSet) and len(self.minimizers) == 1

    @property
    def certificate(self):
        """Return replayable completed evidence, or None for an unresolved result."""
        return self if self.complete else None

    @property
    def scope(self):
        """State the domain to which the exact infimum claim applies."""
        return "global-infimum-on-effective-domain"

    def verify(self):
        """Replay source definedness and complete rational value enumeration."""
        from funcprops import DomainRecord, analysis_data_equal

        if (
            type(self) is not ComponentMinimumResult
            or type(self.domain_record) is not DomainRecord
            or type(self.certified) is not bool
            or self.complete is not True
            or type(self.candidates) is not tuple
            or not self.domain_record.verify()
            or any(type(c) is not OptimizationCandidate for c in self.candidates)
        ):
            return False
        try:
            actual = finite_component_minimum(
                self.expression, self.variable, self.domain_record.requested
            )
            return (
                actual.complete
                and self.domain_record.matches(actual.domain_record)
                and analysis_data_equal(
                    (self.infimum, self.minimizers, self.candidates, self.reason),
                    (
                        actual.infimum,
                        actual.minimizers,
                        actual.candidates,
                        actual.reason,
                    ),
                )
            )
        except (
            TypeError,
            ValueError,
            ArithmeticError,
            NotImplementedError,
            AttributeError,
        ):
            return False


def finite_component_minimum(expr, var, domain=sp.S.Reals, *, certified=False):
    """Find a fixed real algebraic rational function's exact infimum and minimizers.

    Open endpoints and original source holes are retained. The same critical,
    boundary and limit enumeration serves symbopt's ordinary rational route.
    ``certified=True`` independently replays the completed result. Unsupported
    domains, coefficients or comparisons return an incomplete result.
    """
    from funcprops import normalize_real_expression, track_domain

    if not isinstance(certified, bool):
        raise TypeError("certified must be a boolean")
    f, x = sp.sympify(expr), sp.sympify(var)
    if not isinstance(x, sp.Symbol) or x.is_real is False:
        raise ValueError("var must be a real coordinate symbol")
    record = track_domain(f, x, domain)
    result = ComponentMinimumResult(f, x, record)
    if not record.complete:
        return replace(result, reason=record.reason)
    D = record.effective
    if (
        f.has(sp.Float)
        or D.has(sp.Float)
        or f.free_symbols - {x}
        or not f.is_rational_function(x)
    ):
        return replace(result, reason="unsupported-rational-coefficients")
    if D is sp.S.EmptySet:
        result = replace(result, infimum=sp.oo, complete=True)
    else:
        parts = D.args if isinstance(D, sp.Union) else (D,)
        if any(not isinstance(p, (sp.Interval, sp.FiniteSet)) for p in parts):
            return replace(result, reason="unsupported-component-geometry")
        formula = D.as_relational(x)
        branches = normalize_constraint_branches([] if formula is sp.true else formula)
        runs = [
            try_univariate_rational(normalize_real_expression(f), branch, (x,))
            for branch in branches
        ]
        if not runs or any(r is None for r in runs):
            return replace(result, reason="incomplete-rational-enumeration")
        best = runs[0].value
        for run in runs[1:]:
            if _compare(run.value, best) < 0:
                best = run.value
        candidates = tuple(
            OptimizationCandidate(
                point,
                value,
                point is not None,
                D,
                "limit" if point is None else "point",
            )
            for run in runs
            for value, point in run.metadata["enumerated_candidates"]
        )
        minimizers = sp.FiniteSet(
            *(
                c.point
                for c in candidates
                if c.attained and _compare(c.value, best) == 0
            )
        )
        if sp.diff(normalize_real_expression(f), x) == 0:
            minimizers = D
        result = replace(
            result,
            infimum=best,
            minimizers=minimizers,
            candidates=candidates,
            complete=True,
        )
    return replace(result, certified=certified and result.verify())
