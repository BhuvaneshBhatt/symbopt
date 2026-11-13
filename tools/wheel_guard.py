"""Pytest import-provenance check for the isolated installed-wheel harness."""

import os
from pathlib import Path

import pytest


def pytest_sessionstart(session):
    """Reject source imports in the test process before collection begins."""
    import symbopt

    origin = Path(symbopt.__file__).resolve()
    expected = Path(os.environ["SYMBOPT_TEST_PACKAGE_ROOT"]).resolve()
    if not origin.is_relative_to(expected):
        raise pytest.UsageError(f"wheel tests imported {origin} outside {expected}")
    session.config.pluginmanager.get_plugin("terminalreporter").write_line(
        f"Installed package confirmed: {origin}"
    )
