"""Typed replay for univariate optimization evidence."""

from .extrema_bounds import ExtremaBoundsProof
from .univariate_certificates import GlobalMinimumCertificate
from .univariate_optimization import ComponentMinimumResult


def verify_univariate_certificate(certificate, *, expr=None, var=None):
    """Replay a supported optimization record, optionally binding its source.

    Subclasses and unrelated analysis records are rejected. An extrema enclosure
    may verify with incomplete root coverage or unmet requested accuracy; inspect
    its completion properties separately. Component minima replay exact rational
    enumeration, and curvature certificates check their sufficient premises.
    """
    if type(certificate) not in {
        ExtremaBoundsProof,
        GlobalMinimumCertificate,
        ComponentMinimumResult,
    }:
        return False
    import sympy as sp
    from funcprops import analysis_data_equal

    try:
        source = sp.sympify(expr) if expr is not None else None
        variable = sp.sympify(var) if var is not None else None
    except (TypeError, ValueError, sp.SympifyError):
        return False
    if source is not None and not analysis_data_equal(certificate.expression, source):
        return False
    if variable is not None and not analysis_data_equal(certificate.variable, variable):
        return False
    return certificate.verify() is True
