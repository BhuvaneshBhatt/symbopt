"""Future reference cases grouped by mathematical input type."""

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.reference, pytest.mark.future]

_DATA = json.loads(
    (Path(__file__).with_name("reference_case_inventory.json")).read_text()
)
_CASES = (
    15,
    16,
    20,
    32,
    35,
    39,
    43,
    44,
    47,
    48,
    50,
    53,
    56,
    59,
    70,
    95,
    96,
    100,
    108,
    111,
    114,
    118,
    122,
    123,
    126,
    127,
    129,
    132,
    135,
    137,
    161,
    173,
    250,
    251,
    252,
    253,
    254,
    255,
    256,
    257,
    258,
    259,
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
