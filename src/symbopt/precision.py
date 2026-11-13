from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import sympy as sp
from sympy.core.evalf import prec_to_dps


@dataclass(frozen=True)
class PrecisionProvenance:
    """Requested/input precision and the resulting numerical-backend routing policy."""

    requested_precision: int | None
    effective_precision: int
    min_input_precision: int | None
    max_input_precision: int | None
    contains_inexact_input: bool
    float64_backends_allowed: bool = True
    routing_policy: str = "standard"
    notes: tuple[str, ...] = ()

    def as_dict(self):
        """Return the stored capability or precision metadata as a dictionary."""
        return asdict(self)


def _float_precisions(expressions: Iterable[sp.Basic]):
    precisions = []
    for expression in expressions:
        for value in sp.preorder_traversal(sp.sympify(expression)):
            if isinstance(value, sp.Float):
                precisions.append(int(prec_to_dps(value._prec)))
    return precisions


def precision_provenance(objective, constraints, working_precision=None):
    """Infer the effective precision and whether float64-only routes are safe."""
    if working_precision is not None:
        if isinstance(working_precision, bool) or not isinstance(
            working_precision, int
        ):
            raise TypeError("working_precision must be a positive integer or None")
        if working_precision <= 0:
            raise ValueError("working_precision must be positive")
    precisions = _float_precisions([objective, *constraints])
    effective = int(working_precision or (max(precisions) if precisions else 50))
    high_precision_inexact = bool(precisions) and effective > 16
    float64_allowed = not high_precision_inexact
    routing_policy = (
        "preserve_high_precision_inexact" if high_precision_inexact else "standard"
    )
    notes = (
        "Exact symbolic routes preserve exact arithmetic independently of working_precision.",
        (
            "High-precision inexact input detected: float64-only LP/QP/SDP fast paths are bypassed so symbolic/nsolve or verified-interval routes can preserve the requested precision."
            if high_precision_inexact
            else "Float64 fast paths may be used; any exact certification still requires exact postprocessing/proof evidence."
        ),
    )
    return PrecisionProvenance(
        requested_precision=working_precision,
        effective_precision=effective,
        min_input_precision=min(precisions) if precisions else None,
        max_input_precision=max(precisions) if precisions else None,
        contains_inexact_input=bool(precisions),
        float64_backends_allowed=float64_allowed,
        routing_policy=routing_policy,
        notes=notes,
    )


def decimal_digits_to_bits(digits: int) -> int:
    """Return a conservative binary precision for a decimal-digit request."""
    return max(2, math.ceil(int(digits) * math.log2(10)) + 8)


def float64_backends_allowed(objective, constraints, working_precision=None) -> bool:
    """Return whether float64-only fast paths are appropriate for this input."""
    return precision_provenance(
        objective, constraints, working_precision
    ).float64_backends_allowed


def outward_float(value, direction):
    """Enclose an exact finite endpoint by a binary float on the requested side.

    Existing floats already define binary endpoints and stay unchanged. Exact
    endpoints are compared after conversion so an interval can never shrink
    because its rational or algebraic boundary rounds inward.
    """
    if isinstance(value, float):
        return value
    exact = sp.sympify(value)
    numeric = float(exact)
    if not math.isfinite(numeric):
        raise ValueError("interval endpoints must fit finite floating-point bounds")
    if isinstance(exact, sp.Rational):
        # Cross-multiplication compares the binary float with the exact rational
        # without constructing and simplifying another symbolic difference.
        numerator, denominator = numeric.as_integer_ratio()
        if direction * (numerator * exact.q - exact.p * denominator) < 0:
            return math.nextafter(numeric, -math.inf if direction < 0 else math.inf)
        return numeric
    for _ in range(3):
        delta = direction * (sp.Rational(numeric) - exact)
        if delta.is_nonnegative is True:
            return numeric
        numeric = math.nextafter(numeric, -math.inf if direction < 0 else math.inf)
    raise ValueError("could not enclose the exact interval endpoint")


def finite_midpoint(lower, upper):
    """Return a float midpoint of finite endpoints without overflow.

    Halving the endpoints separately is needed only when their sum overflows;
    using it for small endpoints could lose subnormal values.
    """
    total = lower + upper
    return 0.5 * total if math.isfinite(total) else 0.5 * lower + 0.5 * upper
