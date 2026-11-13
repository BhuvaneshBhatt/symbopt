"""One executable regression or capability probe per mathematical problem."""

import pytest

from .checks import ReferenceFailure, execute_case, load_cases

CASES = load_cases()


def parameters():
    for case in CASES:
        marks = [pytest.mark.reference, pytest.mark.reference_corpus, pytest.mark.slow]
        if case["budget_limited"]:
            marks.append(pytest.mark.unsupported)
        if case["expected_failure"]:
            marks += [
                pytest.mark.unsupported,
                pytest.mark.xfail(
                    strict=not case.get("allow_success", False),
                    raises=ReferenceFailure,
                    reason=case["expected_failure"],
                ),
            ]
        yield pytest.param(case, id=case["case_id"], marks=marks)


@pytest.mark.parametrize("case", list(parameters()))
def test_reference_case(case):
    outcome = execute_case(case)
    if outcome["kind"] == "pass":
        return
    common = {
        "AssertionError",
        "NotImplementedError",
        "TypeError",
        "ValueError",
        "CoercionFailed",
    }
    if outcome["kind"] == "Timeout" and (
        case["expected_failure"] or case["budget_limited"]
    ):
        pytest.xfail(outcome["detail"])
    if outcome["kind"] not in common and outcome["kind"] != case["failure_kind"]:
        raise RuntimeError(f"Unexpected reference exception: {outcome}")
    if case["budget_limited"]:
        pytest.xfail(outcome["detail"])
    raise ReferenceFailure(outcome["detail"])
