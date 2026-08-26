"""Future reference cases grouped by mathematical input type."""

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.reference, pytest.mark.future]

_DATA = json.loads(
    (Path(__file__).with_name("reference_case_inventory.json")).read_text()
)
_CASES = (
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    97,
    98,
    99,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    109,
    110,
    112,
    113,
    115,
    116,
    117,
    119,
    120,
    121,
    124,
    125,
    128,
    130,
    131,
    133,
    134,
    136,
    138,
    139,
    140,
    159,
    160,
    164,
    166,
    167,
    168,
    170,
    171,
    172,
    174,
    175,
    242,
    243,
    244,
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
