from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Mapping, Sequence, Tuple
import math

import sympy as sp
from sympy.polys.polyerrors import CoercionFailed, PolynomialError

from .numeric_utils import eval_float

try:
    from flint import arb, ctx
except ImportError:  # pragma: no cover - exercised when python-flint is absent
    arb = None
    ctx = None


Bounds = Tuple[Tuple[float, float], ...]


@dataclass(frozen=True)
class VerifiedBound:
    """A numerical enclosure together with rigor and provenance metadata.

    Only bounds marked rigorous are eligible to contribute to global
    certification."""

    lower: float
    upper: float
    rigorous: bool
    method: str
    backend: str
    domain: str = "valid"

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def finite(self) -> bool:
        return math.isfinite(self.lower) and math.isfinite(self.upper)

    def intersect(self, other: "VerifiedBound") -> "VerifiedBound":
        if not self.rigorous or not other.rigorous:
            return self if self.rigorous else other
        lower = max(self.lower, other.lower)
        upper = min(self.upper, other.upper)
        if lower > upper:
            lower = min(self.lower, other.lower)
            upper = max(self.upper, other.upper)
        return VerifiedBound(
            lower,
            upper,
            True,
            f"{self.method}+{other.method}",
            self.backend if self.backend == other.backend else "mixed",
            _merge_domain(self.domain, other.domain),
        )


@dataclass(frozen=True)
class EvalConfig:
    """Configuration for rigorous expression enclosure.

    Controls Arb precision and which rigorous tightening strategies are attempted."""

    precision: int = 128
    use_bernstein: bool = True
    use_monotonicity: bool = True
    use_taylor: bool = True


def arb_available() -> bool:
    """Return ``True`` when the python-flint Arb backend can be imported."""
    return arb is not None


def _merge_domain(first: str, second: str) -> str:
    order = {"valid": 0, "uncertain": 1, "invalid": 2}
    return first if order[first] >= order[second] else second


def _float_rational(value: float) -> sp.Rational:
    numerator, denominator = float(value).as_integer_ratio()
    return sp.Rational(numerator, denominator)


def _exact_number(expr: sp.Expr) -> sp.Rational | sp.Integer | None:
    if isinstance(expr, sp.Integer):
        return expr
    if isinstance(expr, sp.Rational):
        return expr
    if isinstance(expr, sp.Float):
        try:
            return sp.Rational(expr)
        except (TypeError, ValueError, ArithmeticError):
            return None
    return None


def _number_to_arb(value):
    if arb is None:
        raise RuntimeError("python-flint is required for Arb evaluation")
    if isinstance(value, float):
        exact = _float_rational(value)
        return arb(f"{exact.p}/{exact.q}")
    expr = sp.sympify(value)
    exact = _exact_number(expr)
    if exact is not None:
        if isinstance(exact, sp.Integer):
            return arb(int(exact))
        return arb(f"{exact.p}/{exact.q}")
    if expr == sp.pi:
        return arb.pi()
    if expr == sp.E:
        return arb.const_e()
    raise TypeError(f"Unsupported numeric constant for Arb: {expr}")


def _box_ball(lower: float, upper: float):
    low_ball = _number_to_arb(lower)
    high_ball = _number_to_arb(upper)
    return low_ball.union(high_ball)


def _exact_arb_float(value, direction: int) -> float:
    if not value.is_finite():
        text = str(value)
        if "-inf" in text:
            return float("-inf")
        return float("inf")
    mantissa, exponent = value.man_exp()
    try:
        result = math.ldexp(float(int(mantissa)), int(exponent))
    except OverflowError:
        return math.copysign(float("inf"), int(mantissa))
    target = float("-inf") if direction < 0 else float("inf")
    return math.nextafter(result, target)


def _ball_bound(value, method: str) -> VerifiedBound:
    if not value.is_finite():
        return VerifiedBound(
            float("-inf"),
            float("inf"),
            True,
            method,
            "arb",
            "uncertain",
        )
    lower = _exact_arb_float(value.lower(), -1)
    upper = _exact_arb_float(value.upper(), 1)
    return VerifiedBound(lower, upper, True, method, "arb")


def _contains_zero(value) -> bool:
    return value.contains(0)


@dataclass(frozen=True)
class ArbNode:
    op: str
    value: object = None
    children: Tuple["ArbNode", ...] = ()

    def evaluate(self, env: Mapping[sp.Symbol, object]):
        if self.op == "const":
            return _number_to_arb(self.value)
        if self.op == "pi":
            return arb.pi()
        if self.op == "e":
            return arb.const_e()
        if self.op == "symbol":
            return env[self.value]
        if self.op == "add":
            result = arb(0)
            for child in self.children:
                result += child.evaluate(env)
            return result
        if self.op == "mul":
            result = arb(1)
            for child in self.children:
                result *= child.evaluate(env)
            return result
        if self.op == "pow_int":
            base = self.children[0].evaluate(env)
            power = int(self.value)
            if power < 0 and _contains_zero(base):
                raise ZeroDivisionError("interval denominator contains zero")
            return base**power
        if self.op == "pow_rat":
            base = self.children[0].evaluate(env)
            numerator, denominator = self.value
            root = base.root(denominator)
            if numerator < 0 and _contains_zero(root):
                raise ZeroDivisionError("root denominator contains zero")
            return root**numerator
        if self.op == "pow":
            return self.children[0].evaluate(env) ** self.children[1].evaluate(env)
        if self.op == "abs":
            return abs(self.children[0].evaluate(env))
        if self.op == "func":
            argument = self.children[0].evaluate(env)
            return getattr(argument, self.value)()
        if self.op == "method":
            receiver = self.children[0].evaluate(env)
            args = [child.evaluate(env) for child in self.children[1:]]
            return getattr(receiver, self.value)(*args)
        if self.op == "method_int":
            receiver = self.children[0].evaluate(env)
            return getattr(receiver, self.value[0])(*self.value[1:])
        raise TypeError(f"Unknown Arb node operation: {self.op}")


@lru_cache(maxsize=8192)
def compile_arb(expr: sp.Expr) -> ArbNode:
    if expr.is_Number:
        return ArbNode("const", expr)
    if expr == sp.pi:
        return ArbNode("pi")
    if expr == sp.E:
        return ArbNode("e")
    if isinstance(expr, sp.Symbol):
        return ArbNode("symbol", expr)
    if expr.is_Add:
        return ArbNode("add", children=tuple(compile_arb(arg) for arg in expr.args))
    if expr.is_Mul:
        return ArbNode("mul", children=tuple(compile_arb(arg) for arg in expr.args))
    if expr.is_Pow:
        base, exponent = expr.args
        child = compile_arb(base)
        if exponent.is_Integer:
            return ArbNode("pow_int", int(exponent), (child,))
        if exponent.is_Rational:
            return ArbNode(
                "pow_rat",
                (int(exponent.p), int(exponent.q)),
                (child,),
            )
        return ArbNode("pow", children=(child, compile_arb(exponent)))
    if expr.func is sp.Abs:
        return ArbNode("abs", children=(compile_arb(expr.args[0]),))
    unary = {
        sp.sin: "sin",
        sp.cos: "cos",
        sp.tan: "tan",
        sp.cot: "cot",
        sp.sec: "sec",
        sp.csc: "csc",
        sp.exp: "exp",
        sp.log: "log",
        sp.sqrt: "sqrt",
        sp.sinh: "sinh",
        sp.cosh: "cosh",
        sp.tanh: "tanh",
        sp.coth: "coth",
        sp.sech: "sech",
        sp.csch: "csch",
        sp.asin: "asin",
        sp.acos: "acos",
        sp.atan: "atan",
        sp.asinh: "asinh",
        sp.acosh: "acosh",
        sp.atanh: "atanh",
        sp.erf: "erf",
        sp.erfc: "erfc",
        sp.erfi: "erfi",
        sp.gamma: "gamma",
        sp.loggamma: "lgamma",
        sp.Ei: "ei",
        sp.Si: "si",
        sp.Ci: "ci",
        sp.Shi: "shi",
        sp.Chi: "chi",
        sp.fresnelc: "fresnel_c",
        sp.fresnels: "fresnel_s",
        sp.sinc: "sinc",
    }
    method = unary.get(expr.func)
    if method is not None and len(expr.args) == 1:
        return ArbNode("func", method, (compile_arb(expr.args[0]),))

    if expr.func is sp.LambertW and len(expr.args) in (1, 2):
        branch = int(expr.args[1]) if len(expr.args) == 2 else 0
        if branch not in (0, -1):
            raise TypeError("Arb real LambertW supports branches 0 and -1")
        return ArbNode("method_int", ("lambertw", branch), (compile_arb(expr.args[0]),))

    if expr.func is sp.zeta and len(expr.args) in (1, 2):
        receiver = compile_arb(expr.args[0])
        if len(expr.args) == 1:
            return ArbNode("func", "zeta", (receiver,))
        return ArbNode("method", "zeta", (receiver, compile_arb(expr.args[1])))

    if expr.func is sp.polylog and len(expr.args) == 2:
        order, argument = expr.args
        return ArbNode("method", "polylog", (compile_arb(argument), compile_arb(order)))

    bessel_methods = {
        sp.besselj: "bessel_j",
        sp.bessely: "bessel_y",
        sp.besseli: "bessel_i",
        sp.besselk: "bessel_k",
    }
    bessel_method = bessel_methods.get(expr.func)
    if bessel_method is not None and len(expr.args) == 2:
        order, argument = expr.args
        return ArbNode(
            "method", bessel_method, (compile_arb(argument), compile_arb(order))
        )

    if expr.func is sp.expint and len(expr.args) == 2:
        order, argument = expr.args
        return ArbNode("method", "expint", (compile_arb(argument), compile_arb(order)))

    if expr.func is sp.polygamma and len(expr.args) == 2 and expr.args[0] == 0:
        return ArbNode("func", "digamma", (compile_arb(expr.args[1]),))

    if expr.func is sp.RisingFactorial and len(expr.args) == 2:
        base, order = expr.args
        return ArbNode("method", "rising", (compile_arb(base), compile_arb(order)))

    if getattr(sp, "erfinv", None) is not None and expr.func is sp.erfinv:
        return ArbNode("func", "erfinv", (compile_arb(expr.args[0]),))
    if getattr(sp, "erfcinv", None) is not None and expr.func is sp.erfcinv:
        return ArbNode("func", "erfcinv", (compile_arb(expr.args[0]),))

    airy_map = {
        getattr(sp, "airyai", None): ("airy_ai", 0),
        getattr(sp, "airyaiprime", None): ("airy_ai", 1),
        getattr(sp, "airybi", None): ("airy_bi", 0),
        getattr(sp, "airybiprime", None): ("airy_bi", 1),
    }
    airy = airy_map.get(expr.func)
    if airy is not None:
        return ArbNode("method_int", airy, (compile_arb(expr.args[0]),))

    raise TypeError(f"Unsupported Arb expression: {expr.func}")


def _eval_arb(expr: sp.Expr, env: Mapping[sp.Symbol, object]):
    return compile_arb(expr).evaluate(env)


def _domain_arg_bound(
    expr: sp.Expr,
    variables: Tuple[sp.Symbol, ...],
    bounds: Bounds,
    precision: int,
) -> VerifiedBound:
    try:
        polynomial = _bernstein_bound(expr, variables, bounds)
        if polynomial is not None:
            return polynomial
    except (
        PolynomialError,
        CoercionFailed,
        TypeError,
        ValueError,
        ArithmeticError,
        NotImplementedError,
    ):
        pass
    if arb_available():
        return _arb_natural_raw(expr, variables, bounds, precision)
    return _heuristic_bound(expr, variables, bounds)


def _domain_status(
    expr: sp.Expr,
    variables: Tuple[sp.Symbol, ...],
    bounds: Bounds,
    precision: int,
) -> str:
    checks = []
    if expr.is_Pow:
        base, exponent = expr.args
        if exponent.is_Integer and int(exponent) < 0:
            checks.append(("nonzero", base))
        elif exponent.is_Rational:
            if int(exponent.q) % 2 == 0:
                checks.append(("nonnegative", base))
            if int(exponent.p) < 0:
                checks.append(("nonzero", base))
    elif expr.func in (sp.log, sp.loggamma):
        checks.append(("positive", expr.args[0]))
    elif expr.func is sp.sqrt:
        checks.append(("nonnegative", expr.args[0]))
    elif expr.func in (sp.asin, sp.acos):
        checks.append(("unit", expr.args[0]))
    elif expr.func is sp.acosh:
        checks.append(("ge_one", expr.args[0]))
    elif expr.func is sp.atanh:
        checks.append(("open_unit", expr.args[0]))
    elif expr.func is sp.tan:
        cosine = sp.cos(expr.args[0])
        checks.append(("nonzero", cosine))
    elif expr.func is sp.LambertW:
        branch = int(expr.args[1]) if len(expr.args) == 2 else 0
        if branch == 0:
            checks.append(("lambert_principal", expr.args[0]))
        elif branch == -1:
            checks.append(("lambert_minus_one", expr.args[0]))

    status = "valid"
    for kind, argument in checks:
        bound = _domain_arg_bound(argument, variables, bounds, precision)
        if bound.domain == "invalid":
            return "invalid"
        if not bound.finite:
            status = "uncertain"
            continue
        low, high = bound.lower, bound.upper
        if kind == "positive":
            if high <= 0:
                return "invalid"
            if low <= 0:
                status = "uncertain"
        elif kind == "nonnegative":
            if high < 0:
                return "invalid"
            if low < 0:
                status = "uncertain"
        elif kind == "nonzero":
            if low == 0 and high == 0:
                return "invalid"
            if low <= 0 <= high:
                status = "uncertain"
        elif kind == "unit":
            if high < -1 or low > 1:
                return "invalid"
            if low < -1 or high > 1:
                status = "uncertain"
        elif kind == "ge_one":
            if high < 1:
                return "invalid"
            if low < 1:
                status = "uncertain"
        elif kind == "open_unit":
            if high <= -1 or low >= 1:
                return "invalid"
            if low <= -1 or high >= 1:
                status = "uncertain"
        elif kind == "lambert_principal":
            threshold = -1.0 / math.e
            if high < threshold:
                return "invalid"
            if low < threshold:
                status = "uncertain"
        elif kind == "lambert_minus_one":
            threshold = -1.0 / math.e
            if high >= 0 or low < threshold:
                if high < threshold or low >= 0:
                    return "invalid"
                status = "uncertain"

    for argument in expr.args:
        if isinstance(argument, sp.Expr) and argument.free_symbols:
            child = _domain_status(argument, variables, bounds, precision)
            status = _merge_domain(status, child)
            if status == "invalid":
                return status
    return status


def _arb_natural_raw(
    expr: sp.Expr,
    variables: Tuple[sp.Symbol, ...],
    bounds: Bounds,
    precision: int,
) -> VerifiedBound:
    if arb is None:
        raise RuntimeError("python-flint is not installed")
    old_precision = ctx.prec
    ctx.prec = precision
    try:
        env = {
            variable: _box_ball(lower, upper)
            for variable, (lower, upper) in zip(variables, bounds)
        }
        value = _eval_arb(expr, env)
        return _ball_bound(value, "natural")
    except (AttributeError, NotImplementedError, TypeError):
        # The real domain was already validated symbolically.  Older python-flint
        # builds do not expose every special-function method; the whole real line
        # remains a rigorous, if uninformative, enclosure in that case.
        return VerifiedBound(
            float("-inf"),
            float("inf"),
            True,
            "natural",
            "arb",
            "valid",
        )
    except (ValueError, ZeroDivisionError, OverflowError):
        return VerifiedBound(
            float("-inf"),
            float("inf"),
            True,
            "natural",
            "arb",
            "uncertain",
        )
    finally:
        ctx.prec = old_precision


def _arb_natural(
    expr: sp.Expr,
    variables: Tuple[sp.Symbol, ...],
    bounds: Bounds,
    precision: int,
) -> VerifiedBound:
    domain = _domain_status(expr, variables, bounds, precision)
    if domain != "valid":
        return VerifiedBound(
            float("-inf"),
            float("inf"),
            True,
            "natural",
            "arb",
            domain,
        )
    return _arb_natural_raw(expr, variables, bounds, precision)


def _bounds_as_rationals(bounds: Bounds):
    return tuple(
        (_float_rational(lower), _float_rational(upper)) for lower, upper in bounds
    )


def _rational_outward(value: sp.Rational, direction: int) -> float:
    numeric = float(value)
    try:
        if sp.Rational(*numeric.as_integer_ratio()) == value:
            return numeric
    except (TypeError, ValueError, ArithmeticError):
        pass
    target = float("-inf") if direction < 0 else float("inf")
    return math.nextafter(numeric, target)


def _bernstein_bound(
    expr: sp.Expr,
    variables: Tuple[sp.Symbol, ...],
    bounds: Bounds,
) -> VerifiedBound | None:
    if expr.is_Rational:
        value = sp.Rational(expr)
        return VerifiedBound(
            _rational_outward(value, -1),
            _rational_outward(value, 1),
            True,
            "exact-constant",
            "exact-polynomial",
        )
    if not expr.is_polynomial(*variables):
        return None
    rational_bounds = _bounds_as_rationals(bounds)
    tvars = sp.symbols(f"_b0:{len(variables)}", real=True)
    substitutions = {
        variable: lower + (upper - lower) * tvar
        for variable, tvar, (lower, upper) in zip(variables, tvars, rational_bounds)
    }
    transformed = sp.Poly(
        sp.expand(expr.subs(substitutions)),
        *tvars,
        domain="QQ",
    )
    degrees = tuple(transformed.degree(tvar) for tvar in tvars)
    monomial = {powers: sp.Rational(coeff) for powers, coeff in transformed.terms()}
    coefficients = []
    ranges = [range(degree + 1) for degree in degrees]
    for index in product(*ranges):
        value = sp.Rational(0)
        k_ranges = [range(i + 1) for i in index]
        for powers in product(*k_ranges):
            coeff = monomial.get(powers, sp.Rational(0))
            if coeff == 0:
                continue
            factor = sp.Rational(1)
            for i, k, degree in zip(index, powers, degrees):
                if degree == 0:
                    continue
                factor *= sp.Rational(
                    math.comb(i, k),
                    math.comb(degree, k),
                )
            value += coeff * factor
        coefficients.append(value)

    if not coefficients:
        coefficients = [sp.Rational(0)]
    low = min(coefficients)
    high = max(coefficients)
    return VerifiedBound(
        _rational_outward(low, -1),
        _rational_outward(high, 1),
        True,
        "bernstein",
        "exact-polynomial",
    )


def _arb_point_value(expr: sp.Expr, point: Mapping[sp.Symbol, float], precision: int):
    old_precision = ctx.prec
    ctx.prec = precision
    try:
        env = {symbol: _number_to_arb(value) for symbol, value in point.items()}
        return _eval_arb(expr, env)
    finally:
        ctx.prec = old_precision


def _taylor_bound(
    expr: sp.Expr,
    variables: Tuple[sp.Symbol, ...],
    bounds: Bounds,
    precision: int,
) -> VerifiedBound | None:
    if arb is None or not variables:
        return None
    exact_bounds = _bounds_as_rationals(bounds)
    midpoint = {
        variable: (lower + upper) / 2
        for variable, (lower, upper) in zip(variables, exact_bounds)
    }
    old_precision = ctx.prec
    ctx.prec = precision
    try:
        center = _arb_point_value(expr, midpoint, precision)
        remainder = arb(0)
        deltas = {
            variable: _number_to_arb(lower - midpoint[variable]).union(
                _number_to_arb(upper - midpoint[variable])
            )
            for variable, (lower, upper) in zip(variables, exact_bounds)
        }
        for variable in variables:
            derivative = sp.diff(expr, variable)
            slope = _arb_point_value(derivative, midpoint, precision)
            center += slope * deltas[variable]

        for i, first in enumerate(variables):
            for j, second in enumerate(variables):
                derivative = sp.diff(expr, first, second)
                derivative_bound = _arb_natural(
                    derivative,
                    variables,
                    bounds,
                    precision,
                )
                if not derivative_bound.rigorous or derivative_bound.domain != "valid":
                    return None
                hessian_ball = _box_ball(
                    derivative_bound.lower,
                    derivative_bound.upper,
                )
                remainder += arb("1/2") * hessian_ball * deltas[first] * deltas[second]
        return _ball_bound(center + remainder, "taylor2")
    except Exception:
        return None
    finally:
        ctx.prec = old_precision


def _monotone_bound(
    expr: sp.Expr,
    variables: Tuple[sp.Symbol, ...],
    bounds: Bounds,
    precision: int,
) -> VerifiedBound | None:
    if arb is None or not variables:
        return None
    current = VerifiedBound(float("-inf"), float("inf"), True, "monotone", "arb")
    tightened = False
    for index, variable in enumerate(variables):
        derivative = sp.diff(expr, variable)
        derivative_bound = _arb_natural(derivative, variables, bounds, precision)
        if derivative_bound.domain != "valid":
            continue
        increasing = derivative_bound.lower >= 0
        decreasing = derivative_bound.upper <= 0
        if not increasing and not decreasing:
            continue

        low_face = list(bounds)
        high_face = list(bounds)
        lower, upper = bounds[index]
        if increasing:
            low_face[index] = (lower, lower)
            high_face[index] = (upper, upper)
        else:
            low_face[index] = (upper, upper)
            high_face[index] = (lower, lower)
        low_bound = _arb_natural(expr, variables, tuple(low_face), precision)
        high_bound = _arb_natural(expr, variables, tuple(high_face), precision)
        if low_bound.domain != "valid" or high_bound.domain != "valid":
            continue
        candidate = VerifiedBound(
            low_bound.lower,
            high_bound.upper,
            True,
            "monotone",
            "arb",
        )
        current = candidate if not tightened else current.intersect(candidate)
        tightened = True
    return current if tightened else None


def _heuristic_bound(
    expr: sp.Expr,
    variables: Tuple[sp.Symbol, ...],
    bounds: Bounds,
) -> VerifiedBound:
    values = []
    choices = []
    for lower, upper in bounds:
        choices.append((lower, 0.5 * (lower + upper), upper))
    for point in product(*choices):
        subs = dict(zip(variables, point))
        try:
            values.append(eval_float(expr.subs(subs)))
        except (
            PolynomialError,
            CoercionFailed,
            TypeError,
            ValueError,
            ArithmeticError,
            NotImplementedError,
        ):
            continue
    if not values:
        return VerifiedBound(
            float("-inf"),
            float("inf"),
            False,
            "unavailable",
            "heuristic",
            "uncertain",
        )
    return VerifiedBound(
        min(values),
        max(values),
        False,
        "sample",
        "heuristic",
    )


@lru_cache(maxsize=32768)
def enclose_expression(
    expr: sp.Expr,
    variables: Tuple[sp.Symbol, ...],
    bounds: Bounds,
    config: EvalConfig = EvalConfig(),
) -> VerifiedBound:
    """Compute a rigorous or explicitly heuristic enclosure of a SymPy expression on a box.

    The result records the method, backend, real-domain status, and whether the
    enclosure is mathematically rigorous."""
    domain = _domain_status(expr, variables, bounds, config.precision)
    if domain == "invalid":
        return VerifiedBound(
            float("inf"),
            float("inf"),
            True,
            "domain",
            "symbolic+arb" if arb_available() else "symbolic",
            "invalid",
        )
    if domain == "uncertain" and not expr.is_polynomial(*variables):
        return VerifiedBound(
            float("-inf"),
            float("inf"),
            True,
            "domain",
            "symbolic+arb" if arb_available() else "symbolic",
            "uncertain",
        )
    candidates = []
    if config.use_bernstein:
        try:
            bernstein = _bernstein_bound(expr, variables, bounds)
            if bernstein is not None:
                candidates.append(bernstein)
        except (
            PolynomialError,
            CoercionFailed,
            TypeError,
            ValueError,
            ArithmeticError,
            NotImplementedError,
        ):
            pass

    if arb_available():
        natural = _arb_natural(expr, variables, bounds, config.precision)
        candidates.append(natural)
        if natural.domain == "valid" and config.use_monotonicity:
            monotone = _monotone_bound(expr, variables, bounds, config.precision)
            if monotone is not None:
                candidates.append(monotone)
        if natural.domain == "valid" and config.use_taylor:
            taylor = _taylor_bound(expr, variables, bounds, config.precision)
            if taylor is not None:
                candidates.append(taylor)

    rigorous = [
        candidate
        for candidate in candidates
        if candidate.rigorous and candidate.domain == "valid"
    ]
    if rigorous:
        best = rigorous[0]
        for candidate in rigorous[1:]:
            best = best.intersect(candidate)
        return best

    uncertain = [candidate for candidate in candidates if candidate.rigorous]
    if uncertain:
        best = min(uncertain, key=lambda bound: bound.width)
        return best

    return _heuristic_bound(expr, variables, bounds)


def verify_point(
    constraints: Sequence[sp.Rel],
    point: Mapping[sp.Symbol, float],
    precision: int = 128,
) -> bool:
    exact_point = {symbol: _float_rational(value) for symbol, value in point.items()}
    exact_verified = True
    for relation in constraints:
        try:
            result = sp.simplify(relation.subs(exact_point))
        except (
            PolynomialError,
            CoercionFailed,
            TypeError,
            ValueError,
            ArithmeticError,
            NotImplementedError,
        ):
            exact_verified = False
            break
        if result is sp.S.true or result is True:
            continue
        exact_verified = False
        break
    if exact_verified:
        return True

    if not arb_available():
        return False
    for relation in constraints:
        residual = sp.expand(relation.lhs - relation.rhs)
        try:
            value = _arb_point_value(residual, point, precision)
        except Exception:
            return False
        low = _exact_arb_float(value.lower(), -1)
        high = _exact_arb_float(value.upper(), 1)
        if isinstance(relation, sp.Equality):
            if low != 0 or high != 0:
                return False
        elif relation.func is sp.Ge:
            if low < 0:
                return False
        elif relation.func is sp.Gt:
            if low <= 0:
                return False
        elif relation.func is sp.Le:
            if high > 0:
                return False
        elif relation.func is sp.Lt:
            if high >= 0:
                return False
        elif isinstance(relation, sp.Unequality):
            if low <= 0 <= high:
                return False
    return True
