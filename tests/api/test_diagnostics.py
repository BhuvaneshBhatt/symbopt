from symbopt import DiagnosticKind, SolverDiagnostic


def test_diagnostic_types_are_public():
    diagnostic = SolverDiagnostic(
        "interval",
        "branch_and_bound",
        DiagnosticKind.NUMERICAL_FAILURE,
        "test",
    )
    assert diagnostic.kind is DiagnosticKind.NUMERICAL_FAILURE


def test_diagnostic_from_exception_preserves_context():
    error = ValueError("interval evaluation failed")
    diagnostic = SolverDiagnostic.from_exception(
        "interval",
        "branch_and_bound",
        DiagnosticKind.NUMERICAL_FAILURE,
        error,
    )
    assert diagnostic.solver == "interval"
    assert diagnostic.stage == "branch_and_bound"
    assert diagnostic.kind is DiagnosticKind.NUMERICAL_FAILURE
    assert diagnostic.exception_type == "ValueError"
