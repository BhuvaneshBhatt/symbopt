from __future__ import annotations

import importlib
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import metadata as importlib_metadata

import sympy as sp
from sympy.core.relational import Relational
from sympy.polys.polyerrors import PolynomialError

from .diagnostics import DiagnosticKind, SolverDiagnostic


@dataclass(frozen=True)
class SemialgCapabilities:
    """Features advertised by the installed optional :mod:`semialg` backend."""

    version: str | None
    has_optimizer: bool
    supports_cert_policy: bool
    supports_range_limit: bool
    supports_range_result: bool
    result_has_certified: bool
    result_has_attained: bool
    result_has_diagnostics: bool

    def as_dict(self) -> Mapping[str, object]:
        """Return the stored capability or precision metadata as a dictionary."""
        return {
            "version": self.version,
            "has_optimizer": self.has_optimizer,
            "supports_cert_policy": self.supports_cert_policy,
            "supports_range_limit": self.supports_range_limit,
            "supports_range_result": self.supports_range_result,
            "result_has_certified": self.result_has_certified,
            "result_has_attained": self.result_has_attained,
            "result_has_diagnostics": self.result_has_diagnostics,
        }


@dataclass(frozen=True)
class SemialgOptResult:
    """Exact optimization result converted from the optional semialg backend."""

    value: sp.Expr
    points: tuple[Mapping[sp.Symbol, sp.Expr], ...]
    attained: bool
    method: str
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    certified: bool | None = None
    backend_version: str | None = None
    backend_capabilities: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SemialgRangeResult:
    """Exact range formula and endpoint information from ``semialg``.

    The range formula is a semialgebraic condition on ``value_symbol``.  Endpoint
    values are obtained by optimizing that one-dimensional image, which avoids
    assuming that a QE result is a single interval.
    """

    formula: sp.Expr
    value_symbol: sp.Symbol
    infimum: sp.Expr | None
    supremum: sp.Expr | None
    infimum_attained: bool | None
    supremum_attained: bool | None
    method: str = "semialg_function_range"


@dataclass(frozen=True)
class SemialgProof:
    """Exact CAD/QE proof result for a candidate global lower bound."""

    proven: bool | None
    method: str = "unknown"
    query: sp.Expr | None = None
    witness: Mapping[sp.Symbol, sp.Expr] | None = None


def _add_diag(
    diagnostics: list[SolverDiagnostic] | None,
    stage: str,
    kind: DiagnosticKind,
    exc: BaseException,
) -> None:
    if diagnostics is None:
        return
    item = SolverDiagnostic.from_exception("semialg", stage, kind, exc)
    if any(
        old.solver == item.solver
        and old.stage == item.stage
        and old.kind == item.kind
        and old.message == item.message
        for old in diagnostics
    ):
        return
    diagnostics.append(item)


def _load_module(backend=None):
    return backend if backend is not None else importlib.import_module("semialg")


def _backend_version(module, *, installed: bool) -> str | None:
    value = getattr(module, "__version__", None)
    if value is not None:
        return str(value)
    if installed:
        try:
            return importlib_metadata.version("semialg")
        except importlib_metadata.PackageNotFoundError:
            pass
    return None


def _optimizer_parameters(module) -> set[str]:
    optimizer = getattr(module, "semialgebraic_minimize", None)
    if optimizer is None:
        return set()
    try:
        return set(inspect.signature(optimizer).parameters)
    except (TypeError, ValueError):
        return set()


def _detect_capabilities(module, *, installed: bool) -> SemialgCapabilities:
    params = _optimizer_parameters(module)
    range_function = getattr(module, "function_range", None)
    try:
        range_params = (
            set(inspect.signature(range_function).parameters)
            if range_function
            else set()
        )
    except (TypeError, ValueError):
        range_params = set()
    result_type = getattr(module, "OptimizationResult", None)
    if result_type is None and installed:
        # semialg >= 1.3 keeps result records in their owning public specialist
        # namespace rather than duplicating them at the package root.
        try:
            result_module = importlib.import_module("semialg.optimization_results")
        except ImportError:
            result_module = None
        if result_module is not None:
            result_type = getattr(result_module, "OptimizationResult", None)
    fields = set(getattr(result_type, "__dataclass_fields__", {}))
    return SemialgCapabilities(
        version=_backend_version(module, installed=installed),
        has_optimizer=hasattr(module, "semialgebraic_minimize"),
        supports_cert_policy="certification" in params,
        supports_range_limit="range_cost_limit" in params,
        supports_range_result="return_result" in range_params,
        result_has_certified="certified" in fields,
        result_has_attained=("attained" in fields) or not fields,
        result_has_diagnostics=("diagnostics" in fields) or not fields,
    )


@lru_cache(maxsize=1)
def _installed_capabilities() -> SemialgCapabilities:
    return _detect_capabilities(_load_module(), installed=True)


def get_semialg_capabilities(backend=None) -> SemialgCapabilities:
    """Inspect the optional semialg backend and report supported optimization features."""
    if backend is None:
        return _installed_capabilities()
    return _detect_capabilities(_load_module(backend), installed=False)


def semialg_available() -> bool:
    """Return whether the installed backend exposes the APIs symbopt uses.

    ``semialg`` lazily imports some submodules, so importing its root package is
    not sufficient to establish useability. Probe the decision and optimization
    entry points that symbopt relies on and treat missing transitive dependencies
    as an unavailable optional backend.
    """
    try:
        module = _load_module()
        required = (
            "is_satisfiable",
            "implies",
            "semialgebraic_minimize",
        )
        return all(callable(getattr(module, name, None)) for name in required)
    except (ImportError, AttributeError):
        return False


def make_formula(constraints) -> sp.Expr:
    """Convert a constraint sequence into one SymPy Boolean formula."""
    if constraints is None:
        return sp.true
    if isinstance(constraints, (list, tuple, set, frozenset)):
        items = [sp.sympify(item) for item in constraints]
        return sp.And(*items) if items else sp.true
    if constraints is True:
        return sp.true
    if constraints is False:
        return sp.false
    return sp.sympify(constraints)


def _rational_expr(expr: sp.Expr, variables) -> bool:
    try:
        num, den = sp.fraction(sp.together(expr))
        return bool(
            sp.expand(num).is_polynomial(*variables)
            and sp.expand(den).is_polynomial(*variables)
        )
    except (TypeError, ValueError, PolynomialError):
        return False


def _formula_ok(formula: sp.Expr, variables) -> bool:
    if formula in (sp.true, sp.false):
        return True
    if isinstance(formula, Relational):
        return _rational_expr(formula.lhs - formula.rhs, variables)
    if isinstance(formula, (sp.And, sp.Or)):
        return all(_formula_ok(arg, variables) for arg in formula.args)
    if isinstance(formula, sp.Not):
        return _formula_ok(formula.args[0], variables)
    return False


def can_reason(constraints, variables) -> bool:
    """Return whether constraints are in the conservative backend subset."""
    variables = tuple(variables)
    return _formula_ok(make_formula(constraints), variables)


@lru_cache(maxsize=2048)
def _sat_cached(formula: sp.Expr, variables: tuple):
    module = _load_module()
    return module.is_satisfiable(formula, variables, return_result=True)


def _sat_call(formula, variables, backend=None):
    if backend is None:
        return _sat_cached(formula, tuple(variables))
    module = _load_module(backend)
    return module.is_satisfiable(formula, tuple(variables), return_result=True)


def _optional_bool(value) -> bool | None:
    """Convert backend proof states."""
    if value is True or value is sp.true:
        return True
    if value is False or value is sp.false:
        return False
    if isinstance(value, bool):
        return value
    return None


def exact_feasible(
    constraints,
    variables,
    diagnostics: list[SolverDiagnostic] | None = None,
    backend=None,
) -> bool | None:
    """Decide exact semialgebraic feasibility when the backend is applicable."""
    formula = make_formula(constraints)
    variables = tuple(variables)
    if formula is sp.false or formula == sp.false:
        return False
    if formula is sp.true or formula == sp.true:
        return True
    if not can_reason(formula, variables):
        return None
    try:
        result = _sat_call(formula, variables, backend)
    except ImportError as exc:
        _add_diag(diagnostics, "feasibility", DiagnosticKind.UNSUPPORTED, exc)
        return None
    except NotImplementedError as exc:
        _add_diag(diagnostics, "feasibility", DiagnosticKind.UNSUPPORTED, exc)
        return None
    except (TypeError, ValueError, ArithmeticError) as exc:
        _add_diag(
            diagnostics,
            "feasibility",
            DiagnosticKind.SYMBOLIC_FAILURE,
            exc,
        )
        return None
    return _optional_bool(getattr(result, "satisfiable", result))


def prune_redundant(
    constraints: Sequence[Relational],
    variables,
    diagnostics: list[SolverDiagnostic] | None = None,
    max_checks: int = 12,
    backend=None,
):
    """Remove constraints exactly implied by the remaining constraints."""
    items = list(constraints)
    variables = tuple(variables)
    if len(items) < 2 or len(items) > max_checks:
        return items
    if not can_reason(items, variables):
        return items
    try:
        module = _load_module(backend)
    except ImportError as exc:
        _add_diag(diagnostics, "redundancy", DiagnosticKind.UNSUPPORTED, exc)
        return items

    kept = list(items)
    index = 0
    checks = 0
    while index < len(kept) and checks < max_checks:
        current = kept[index]
        others = kept[:index] + kept[index + 1 :]
        checks += 1
        try:
            result = module.implies(
                make_formula(others),
                current,
                variables,
                return_result=True,
            )
            valid = _optional_bool(getattr(result, "valid", result))
        except (ImportError, NotImplementedError) as exc:
            _add_diag(diagnostics, "redundancy", DiagnosticKind.UNSUPPORTED, exc)
            index += 1
            continue
        except (TypeError, ValueError, ArithmeticError) as exc:
            _add_diag(
                diagnostics,
                "redundancy",
                DiagnosticKind.SYMBOLIC_FAILURE,
                exc,
            )
            index += 1
            continue
        if valid is True:
            kept.pop(index)
        else:
            index += 1
    return kept


def pattern_feasible(
    inequalities,
    equalities,
    variables,
    pattern,
    diagnostics: list[SolverDiagnostic] | None = None,
    backend=None,
) -> bool | None:
    """Check whether a requested KKT active subset is exactly feasible."""
    constraints = [sp.Ge(expr, 0) for expr in inequalities]
    constraints.extend(sp.Eq(expr, 0) for expr in equalities)
    constraints.extend(
        sp.Eq(expr, 0) for expr, active in zip(inequalities, pattern) if active
    )
    if len(variables) > 4 or len(constraints) > 14:
        return None
    return exact_feasible(constraints, variables, diagnostics, backend)


def _sympy_exact_points(equations, variables):
    """Return finite exact polynomial solutions when SymPy can solve them."""
    try:
        raw = sp.solve_poly_system([sp.expand(eq) for eq in equations], *variables)
    except (NotImplementedError, ValueError, TypeError, PolynomialError):
        return []
    points = []
    for values in raw or ():
        if len(values) != len(variables):
            continue
        if any(sp.sympify(value).free_symbols for value in values):
            continue
        points.append(
            {var: sp.simplify(value) for var, value in zip(variables, values)}
        )
    return points


def solve_exact_points(
    equations,
    variables,
    diagnostics: list[SolverDiagnostic] | None = None,
    count: int = 64,
    backend=None,
):
    """Solve a finite polynomial system through semialg's exact RUR path."""
    variables = tuple(variables)
    constraints = [sp.Eq(sp.expand(eq), 0) for eq in equations]
    if not constraints or not can_reason(constraints, variables):
        return []
    try:
        module = _load_module(backend)
        solver = getattr(module, "solve_semialgebraic", None)
        if not callable(solver):
            return _sympy_exact_points(equations, variables)
        result = solver(
            make_formula(constraints),
            variables,
            method="rur",
            count=count,
        )
    except (ImportError, NotImplementedError) as exc:
        _add_diag(diagnostics, "rur_solve", DiagnosticKind.UNSUPPORTED, exc)
        return _sympy_exact_points(equations, variables)
    except (
        TypeError,
        ValueError,
        RuntimeError,
        ArithmeticError,
        PolynomialError,
    ) as exc:
        _add_diag(
            diagnostics,
            "rur_solve",
            DiagnosticKind.SYMBOLIC_FAILURE,
            exc,
        )
        return _sympy_exact_points(equations, variables)

    samples = getattr(result, "samples", ())
    points = []
    for sample in samples or ():
        if isinstance(sample, Mapping) and all(var in sample for var in variables):
            points.append({var: sp.simplify(sample[var]) for var in variables})
    return points or _sympy_exact_points(equations, variables)


def _exact_point(point) -> bool:
    return bool(point) and all(
        not sp.sympify(value).has(sp.Float) and not sp.sympify(value).free_symbols
        for value in point.values()
    )


def certify_global_min(
    objective,
    constraints,
    variables,
    point,
    value,
    diagnostics: list[SolverDiagnostic] | None = None,
    backend=None,
) -> SemialgProof:
    """Prove global minimality by excluding every feasible lower value."""
    variables = tuple(variables)
    value = sp.simplify(value)
    if value.has(sp.Float) or not _exact_point(point):
        return SemialgProof(None, method="inexact_candidate")
    formula = make_formula(constraints)
    if not can_reason(formula, variables):
        return SemialgProof(None, method="unsupported_formula")

    fixed = [sp.Eq(var, point[var]) for var in variables]
    feasible = exact_feasible([formula, *fixed], variables, diagnostics, backend)
    if feasible is not True:
        return SemialgProof(False, method="candidate_feasibility")

    query = sp.And(formula, sp.Lt(objective, value))
    try:
        result = _sat_call(query, variables, backend)
    except ImportError as exc:
        _add_diag(diagnostics, "global_proof", DiagnosticKind.UNSUPPORTED, exc)
        return SemialgProof(None, query=query)
    except NotImplementedError as exc:
        _add_diag(diagnostics, "global_proof", DiagnosticKind.UNSUPPORTED, exc)
        return SemialgProof(None, query=query)
    except (TypeError, ValueError, ArithmeticError) as exc:
        _add_diag(
            diagnostics,
            "global_proof",
            DiagnosticKind.CERTIFICATE_FAILURE,
            exc,
        )
        return SemialgProof(None, query=query)

    sat = _optional_bool(getattr(result, "satisfiable", result))
    method = str(getattr(result, "method", "semialg"))
    witness = getattr(result, "witness", None)
    proven = True if sat is False else False if sat is True else None
    return SemialgProof(proven, method, query, witness)


def minimize_exact(
    objective,
    constraints,
    variables,
    diagnostics: list[SolverDiagnostic] | None = None,
    backend=None,
    *,
    certification: str = "auto",
    range_cost_limit: int | None = None,
) -> SemialgOptResult | None:
    """Run semialg's exact semialgebraic minimizer for suitable problems.

    The adapter discovers optimizer keyword support at runtime. In particular,
    no dimension ceiling is imposed by symbopt; dimensional support belongs to
    the installed semialg backend.
    """
    if certification not in {"auto", "candidate", "complete"}:
        raise ValueError(
            "semialg certification must be 'auto', 'candidate', or 'complete'"
        )
    if range_cost_limit is not None and range_cost_limit < 0:
        raise ValueError("semialg range_cost_limit must be nonnegative")

    variables = tuple(variables)
    formula = make_formula(constraints)
    if not variables:
        return None
    if not _rational_expr(sp.sympify(objective), variables):
        return None
    if not can_reason(formula, variables):
        return None
    try:
        module = _load_module(backend)
        capabilities = get_semialg_capabilities(backend)
        if not capabilities.has_optimizer:
            return None
        if certification != "auto" and not capabilities.supports_cert_policy:
            raise NotImplementedError(
                "installed semialg does not expose optimisation certification policies"
            )
        if range_cost_limit is not None and not capabilities.supports_range_limit:
            raise NotImplementedError(
                "installed semialg does not expose range_cost_limit"
            )

        kwargs = {"return_result": True}
        if capabilities.supports_cert_policy:
            kwargs["certification"] = certification
        if range_cost_limit is not None and capabilities.supports_range_limit:
            kwargs["range_cost_limit"] = range_cost_limit
        result = module.semialgebraic_minimize(objective, formula, variables, **kwargs)
    except ImportError as exc:
        _add_diag(diagnostics, "minimize", DiagnosticKind.UNSUPPORTED, exc)
        return None
    except NotImplementedError as exc:
        _add_diag(diagnostics, "minimize", DiagnosticKind.UNSUPPORTED, exc)
        return None
    except (TypeError, ValueError, ArithmeticError) as exc:
        _add_diag(
            diagnostics,
            "minimize",
            DiagnosticKind.SYMBOLIC_FAILURE,
            exc,
        )
        return None

    points = tuple(
        {var: sp.simplify(point[var]) for var in variables}
        for point in getattr(result, "points", ())
        if all(var in point for var in variables)
    )
    certified = _optional_bool(getattr(result, "certified", None))
    backend_diagnostics = dict(getattr(result, "diagnostics", {}) or {})
    backend_diagnostics.setdefault("backend_version", capabilities.version)
    backend_diagnostics.setdefault("certification_policy", certification)
    if range_cost_limit is not None:
        backend_diagnostics.setdefault("range_cost_limit", range_cost_limit)
    return SemialgOptResult(
        value=sp.simplify(result.value),
        points=points,
        attained=bool(getattr(result, "attained", bool(points))),
        method=str(getattr(result, "method", "semialg")),
        diagnostics=backend_diagnostics,
        certified=certified,
        backend_version=capabilities.version,
        backend_capabilities=capabilities.as_dict(),
    )


def _truth_value(result) -> bool | None:
    direct = _optional_bool(result)
    if direct is not None:
        return direct
    for name in ("proven", "valid", "holds", "result"):
        value = _optional_bool(getattr(result, name, None))
        if value is not None:
            return value
    return None


def prove_nonnegative_exact(
    expr,
    constraints,
    variables,
    diagnostics: list[SolverDiagnostic] | None = None,
    backend=None,
) -> SemialgProof:
    """Prove ``expr >= 0`` exactly with semialg sign/CAD reasoning.

    For an unconstrained polynomial this uses semialg's dedicated
    ``prove_nonnegative`` sign prover. With semialgebraic constraints, the same
    proposition is expressed as an exact implication because the dedicated sign
    API is global rather than assumption-parametrized.
    """
    variables = tuple(variables)
    expr = sp.expand(sp.sympify(expr))
    if not expr.is_polynomial(*variables):
        return SemialgProof(None, method="nonpolynomial_sign_query")
    formula = make_formula(constraints)
    if not can_reason(formula, variables):
        return SemialgProof(None, method="unsupported_formula")
    query = sp.Ge(expr, 0)
    try:
        module = _load_module(backend)
        if formula is sp.true or formula == sp.true:
            result = module.prove_nonnegative(expr, variables, return_result=True)
            proven = _truth_value(result)
            method = str(
                getattr(
                    result,
                    "method",
                    getattr(result, "proof_method", "semialg_sign"),
                )
            )
            witness = getattr(
                result,
                "counterexample",
                getattr(result, "witness", None),
            )
            return SemialgProof(proven, method, query, witness)

        result = module.implies(
            formula,
            query,
            variables,
            return_result=True,
        )
        proven = _truth_value(result)
        method = str(getattr(result, "method", "semialg_implication"))
        witness = getattr(result, "counterexample", None)
        return SemialgProof(proven, method, sp.Implies(formula, query), witness)
    except ImportError as exc:
        _add_diag(diagnostics, "sign_proof", DiagnosticKind.UNSUPPORTED, exc)
    except NotImplementedError as exc:
        _add_diag(diagnostics, "sign_proof", DiagnosticKind.UNSUPPORTED, exc)
    except (TypeError, ValueError, ArithmeticError) as exc:
        _add_diag(
            diagnostics,
            "sign_proof",
            DiagnosticKind.CERTIFICATE_FAILURE,
            exc,
        )
    return SemialgProof(None, query=query)


def function_range_exact(
    objective,
    constraints,
    variables,
    diagnostics: list[SolverDiagnostic] | None = None,
    backend=None,
) -> SemialgRangeResult | None:
    """Compute an exact semialgebraic objective range and its endpoints.

    ``semialg.function_range`` performs the image/QE step. The resulting formula
    may be disconnected or otherwise non-interval, so endpoint extraction is
    delegated back to semialg's exact one-variable minimizer/maximizer instead of
    trying to parse the Boolean formula structurally in ``symbopt``.
    """
    variables = tuple(variables)
    if not variables:
        return None
    formula = make_formula(constraints)
    if not can_reason(formula, variables):
        return None
    value_symbol = sp.Dummy("range_value", real=True)
    try:
        module = _load_module(backend)
        capabilities = get_semialg_capabilities(backend)
        if capabilities.supports_range_result:
            range_result = module.function_range(
                objective,
                formula,
                variables,
                value_symbol=value_symbol,
                return_result=True,
            )
            image = sp.sympify(getattr(range_result, "formula", sp.false))
            if image is sp.false or image == sp.false:
                return None
            low = getattr(range_result, "infimum", None)
            high = getattr(range_result, "supremum", None)
            if low is not None or high is not None:
                low_attained = getattr(
                    range_result,
                    "minimum_attained",
                    getattr(range_result, "lower_bound_attained", None),
                )
                high_attained = getattr(
                    range_result,
                    "maximum_attained",
                    getattr(range_result, "upper_bound_attained", None),
                )
                return SemialgRangeResult(
                    formula=image,
                    value_symbol=getattr(range_result, "value_symbol", value_symbol),
                    infimum=None if low is None else sp.simplify(low),
                    supremum=None if high is None else sp.simplify(high),
                    infimum_attained=(
                        None if low_attained is None else bool(low_attained)
                    ),
                    supremum_attained=(
                        None if high_attained is None else bool(high_attained)
                    ),
                    method=str(
                        getattr(range_result, "method", "semialg_function_range")
                    ),
                )
        else:
            image = module.function_range(
                objective,
                formula,
                variables,
                value_symbol=value_symbol,
            )
            image = sp.sympify(image)
            if image is sp.false or image == sp.false:
                return None

        low_result = module.semialgebraic_minimize(
            value_symbol,
            image,
            (value_symbol,),
            return_result=True,
        )
        high_result = module.semialgebraic_maximize(
            value_symbol,
            image,
            (value_symbol,),
            return_result=True,
        )
        low = sp.simplify(getattr(low_result, "value", sp.oo))
        high = sp.simplify(getattr(high_result, "value", -sp.oo))
        low_attained = getattr(low_result, "attained", None)
        high_attained = getattr(high_result, "attained", None)
        method = str(
            getattr(
                low_result,
                "method",
                getattr(high_result, "method", "semialg_function_range"),
            )
        )
        return SemialgRangeResult(
            formula=image,
            value_symbol=value_symbol,
            infimum=low,
            supremum=high,
            infimum_attained=(None if low_attained is None else bool(low_attained)),
            supremum_attained=(None if high_attained is None else bool(high_attained)),
            method=method,
        )
    except ImportError as exc:
        _add_diag(diagnostics, "function_range", DiagnosticKind.UNSUPPORTED, exc)
    except NotImplementedError as exc:
        _add_diag(diagnostics, "function_range", DiagnosticKind.UNSUPPORTED, exc)
    except (TypeError, ValueError, ArithmeticError) as exc:
        _add_diag(
            diagnostics,
            "function_range",
            DiagnosticKind.SYMBOLIC_FAILURE,
            exc,
        )
    return None
