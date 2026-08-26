import pytest

from symbopt.validation_suite import run_validation_suite

pytestmark = [pytest.mark.validation, pytest.mark.slow]


def test_full_numerical_validation_battery_green():
    r = run_validation_suite()
    failed = {k: v for k, v in r.items() if not v.get("passed", False)}
    assert not failed, failed
