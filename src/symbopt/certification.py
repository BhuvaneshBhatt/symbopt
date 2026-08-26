from __future__ import annotations
from typing import Dict, Optional
import math
import sympy as sp
from .algebraic_numbers import exactify, exactify_dict
from .problem_models import OptimalityCertificate, ProofArtifact


def exact_float_scalar(value):
    """Represent a numeric enclosure endpoint without changing its value."""
    if isinstance(value, (float,)):
        if math.isnan(value):
            return sp.nan
        if math.isinf(value):
            return sp.oo if value > 0 else -sp.oo
        numerator, denominator = value.as_integer_ratio()
        return sp.Rational(numerator, denominator)
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return sp.sympify(value)
    if math.isnan(numeric):
        return sp.nan
    if math.isinf(numeric):
        return sp.oo if numeric > 0 else -sp.oo
    numerator, denominator = numeric.as_integer_ratio()
    return sp.Rational(numerator, denominator)


def rationalize_scalar(x, maxsteps: int = 6):
    try:
        return sp.nsimplify(x, rational=True)
    except (TypeError, ValueError, ArithmeticError, sp.SympifyError):
        return sp.sympify(x)


def rationalize_dict(d: Dict[sp.Symbol, object]) -> Dict[sp.Symbol, sp.Expr]:
    out = {}
    for k, v in d.items():
        try:
            out[k] = rationalize_scalar(v)
        except (TypeError, ValueError, ArithmeticError, sp.SympifyError):
            out[k] = sp.sympify(v)
    return out


def build_certificate(
    best_candidate=None,
    interval_result=None,
    sos_result=None,
    sos_lower_bound=None,
    sos_bound_numeric=None,
    lp_qp_kind: Optional[str] = None,
    constraints=(),
    sos_rigorous: bool = False,
    direct_metadata=None,
    semialg_result=None,
    semialg_proof=None,
    semialg_sign=None,
    semialg_range=None,
):
    cert = OptimalityCertificate()
    if best_candidate is not None:
        cert.primal_solution = exactify_dict(best_candidate.primal_values)
        cert.primal_value = exactify(best_candidate.objective_value)
        cert.upper_bound = cert.primal_value
        cert.proof_objects.append(
            ProofArtifact(
                "candidate",
                f"Candidate from {best_candidate.source}",
                {
                    "classification": best_candidate.local_classification,
                    "primal_solution_exact": cert.primal_solution,
                    "primal_value_exact": cert.primal_value,
                },
            )
        )
    if interval_result is not None:
        if interval_result.lower_bound_rigorous:
            cert.lower_bound = exact_float_scalar(interval_result.best_lower_bound)
        if cert.upper_bound is None and interval_result.upper_bound_rigorous:
            cert.upper_bound = exact_float_scalar(interval_result.best_upper_bound)
        cert.proof_objects.append(
            ProofArtifact(
                "interval",
                "Rigorous interval bound"
                if interval_result.lower_bound_rigorous
                else "Heuristic interval search",
                {
                    "lower": interval_result.best_lower_bound,
                    "upper": interval_result.best_upper_bound,
                    "lower_rigorous": interval_result.lower_bound_rigorous,
                    "upper_rigorous": interval_result.upper_bound_rigorous,
                    "method": interval_result.lower_bound_method,
                    "backend": interval_result.interval_backend,
                },
            )
        )
    if sos_lower_bound is not None or sos_bound_numeric is not None:
        lb = sos_lower_bound if sos_lower_bound is not None else sos_bound_numeric
        if sos_rigorous:
            cert.lower_bound = rationalize_scalar(lb)
        cert.proof_objects.append(
            ProofArtifact(
                "sos",
                "Exact SOS lower bound" if sos_rigorous else "Approximate SOS bound",
                {
                    "lower_bound": str(lb),
                    "rigorous": sos_rigorous,
                    "success": getattr(sos_result, "success", None),
                    "exact_residual_zero": getattr(
                        getattr(sos_result, "certificate", None),
                        "exact_residual_zero",
                        None,
                    ),
                },
            )
        )
    if lp_qp_kind is not None:
        feasibility = (
            verify_lp_qp_certificate(
                cert.primal_solution,
                cert.primal_value,
                constraints,
            )
            if cert.primal_solution
            else []
        )
        metadata = direct_metadata or {}
        cert.proof_objects.append(
            ProofArtifact(
                lp_qp_kind,
                f"Direct {lp_qp_kind.upper()} solve",
                {
                    "feasibility_check": feasibility,
                    "exact_postprocessing": metadata,
                },
            )
        )
        if metadata.get("rigorous_global") and cert.primal_value is not None:
            cert.lower_bound = cert.primal_value
            cert.upper_bound = cert.primal_value

    if semialg_sign is not None:
        proven = getattr(semialg_sign, "proven", None)
        cert.proof_objects.append(
            ProofArtifact(
                "semialg_sign",
                "Exact semialgebraic nonnegativity proof"
                if proven is True
                else "Semialgebraic nonnegativity query",
                {
                    "proven": proven,
                    "method": getattr(semialg_sign, "method", None),
                    "query": sp.sstr(getattr(semialg_sign, "query", None)),
                    "counterexample": getattr(semialg_sign, "witness", None),
                },
            )
        )
        if proven is True and cert.primal_value is not None:
            cert.lower_bound = cert.primal_value
            cert.upper_bound = cert.primal_value

    if semialg_range is not None:
        infimum = getattr(semialg_range, "infimum", None)
        if infimum is not None and infimum not in (sp.oo, -sp.oo):
            cert.lower_bound = sp.simplify(infimum)
        cert.proof_objects.append(
            ProofArtifact(
                "semialg_range",
                "Exact semialgebraic objective range",
                {
                    "formula": sp.sstr(semialg_range.formula),
                    "infimum": infimum,
                    "supremum": getattr(semialg_range, "supremum", None),
                    "infimum_attained": getattr(
                        semialg_range, "infimum_attained", None
                    ),
                    "supremum_attained": getattr(
                        semialg_range, "supremum_attained", None
                    ),
                    "method": getattr(semialg_range, "method", None),
                },
            )
        )

    if semialg_proof is not None:
        proven = getattr(semialg_proof, "proven", None)
        cert.proof_objects.append(
            ProofArtifact(
                "semialg_qe",
                "Exact semialgebraic global-optimality proof"
                if proven is True
                else "Semialgebraic global-optimality query",
                {
                    "proven": proven,
                    "method": getattr(semialg_proof, "method", None),
                    "query": sp.sstr(getattr(semialg_proof, "query", None)),
                    "witness": getattr(semialg_proof, "witness", None),
                },
            )
        )
        if proven is True and cert.primal_value is not None:
            cert.lower_bound = cert.primal_value
            cert.upper_bound = cert.primal_value

    if semialg_result is not None:
        value = sp.simplify(semialg_result.value)
        semialg_certified = getattr(semialg_result, "certified", None)
        if semialg_certified is True:
            cert.lower_bound = value
        if getattr(semialg_result, "attained", False):
            cert.upper_bound = value
        cert.proof_objects.append(
            ProofArtifact(
                "semialg_opt",
                "Exact semialgebraic optimization result",
                {
                    "value": value,
                    "attained": semialg_result.attained,
                    "certified": semialg_certified,
                    "method": semialg_result.method,
                    "points": semialg_result.points,
                    "diagnostics": dict(semialg_result.diagnostics),
                    "backend_version": getattr(semialg_result, "backend_version", None),
                    "backend_capabilities": dict(
                        getattr(semialg_result, "backend_capabilities", {})
                    ),
                },
            )
        )

    if cert.lower_bound is not None and cert.upper_bound is not None:
        try:
            cert.gap = sp.nsimplify(sp.simplify(cert.upper_bound - cert.lower_bound))
            if cert.gap == 0:
                cert.status = "proven_optimal"
            else:
                cert.status = "bounded"
        except (TypeError, ValueError, ArithmeticError):
            cert.status = "bounded"
    elif cert.primal_value is not None:
        cert.status = "candidate_only"
    if (
        semialg_result is not None
        and getattr(semialg_result, "certified", None) is True
        and not getattr(semialg_result, "attained", False)
    ):
        cert.status = "proven_infimum"
    if (
        semialg_range is not None
        and cert.primal_value is None
        and getattr(semialg_range, "infimum", None) is not None
    ):
        cert.status = "proven_infimum"
    return cert


def verify_lp_qp_certificate(
    primal_solution: Dict[sp.Symbol, sp.Expr], objective_value, constraints
):
    feas = []
    for rel in constraints:
        try:
            resid = sp.simplify((rel.lhs - rel.rhs).subs(primal_solution))
            ok = None
            if isinstance(rel, sp.Equality):
                ok = sp.simplify(resid) == 0
            elif rel.func in (sp.Ge, sp.Gt):
                ok = resid.is_nonnegative if rel.func is sp.Ge else resid.is_positive
            elif rel.func in (sp.Le, sp.Lt):
                rr = -resid
                ok = rr.is_nonnegative if rel.func is sp.Le else rr.is_positive
            else:
                ok = True
            feas.append((sp.sstr(rel), bool(ok) if ok is not None else None))
        except (TypeError, ValueError, ArithmeticError):
            feas.append((sp.sstr(rel), None))
    return feas
