import symbopt
from symbopt.api import __all__ as stable_api


def test_stable_public_api_has_substantive_docstrings():
    missing = []
    for name in stable_api:
        obj = getattr(symbopt, name)
        doc = getattr(obj, "__doc__", None)
        if doc is None or len(doc.strip()) < 40:
            missing.append(name)
    assert not missing


def test_core_result_types_have_docstrings():
    names = [
        "PipelineResult",
        "CandidateSolution",
        "OptimalityCertificate",
        "IntervalResult",
        "SDPProblem",
        "SDPResult",
        "SOSResult",
        "SOSCertificate",
    ]
    missing = [
        name
        for name in names
        if len((getattr(symbopt, name).__doc__ or "").strip()) < 40
    ]
    assert not missing
