"""Future reference cases grouped by mathematical input type."""

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.reference, pytest.mark.future]

_DATA = json.loads(
    (Path(__file__).with_name("reference_case_inventory.json")).read_text()
)
_CASES = (
    10,
    11,
    14,
    18,
    26,
    30,
    33,
    36,
    38,
    41,
    42,
    46,
    52,
    55,
    62,
    65,
    66,
    67,
    69,
    71,
    73,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    148,
    149,
    150,
    151,
    152,
    153,
    154,
    155,
    156,
    157,
    158,
    213,
    214,
    217,
    218,
    219,
    220,
    221,
    222,
    226,
    227,
    234,
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
