"""Future reference cases grouped by mathematical input type."""

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.reference, pytest.mark.future]

_DATA = json.loads(
    (Path(__file__).with_name("reference_case_inventory.json")).read_text()
)
_CASES = (
    3,
    5,
    12,
    13,
    17,
    19,
    21,
    22,
    23,
    31,
    34,
    37,
    40,
    45,
    49,
    51,
    54,
    68,
    176,
    212,
    215,
    216,
    260,
)


@pytest.mark.parametrize(
    "case_no,description,family",
    [
        pytest.param(
            item["case"],
            item["description"],
            item["family"],
            id=f"reference-case-{item['case']:03d}",
            marks=pytest.mark.xfail(
                run=False,
                strict=True,
                reason=f"{item['family']}: {item['description']}",
            ),
        )
        for item in _DATA
        if item["case"] in _CASES
    ],
)
def test_future_reference_case(case_no, description, family):
    raise NotImplementedError(f"reference case {case_no} ({family}): {description}")
