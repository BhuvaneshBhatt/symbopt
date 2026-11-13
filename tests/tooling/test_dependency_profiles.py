"""Declared runtime floors and installed-wheel test selection stay synchronized."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

from tools.test_wheel import CORE_TESTS

ROOT = Path(__file__).resolve().parents[2]


def test_minimum_dependency_floors():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    pins = [
        requirement
        for line in (ROOT / "tests/constraints-minimum.txt").read_text().splitlines()
        if line and not line.startswith("#")
        for requirement in [Requirement(line)]
    ]
    constraints = {}
    for requirement in pins:
        specifiers = list(requirement.specifier)
        assert len(specifiers) == 1 and specifiers[0].operator == "=="
        constraints[requirement.name] = specifiers[0].version
    dependencies = [Requirement(value) for value in metadata["project"]["dependencies"]]
    assert set(constraints) == {requirement.name for requirement in dependencies}
    for requirement in dependencies:
        floor = next(
            spec.version for spec in requirement.specifier if spec.operator == ">="
        )
        assert Version(constraints[requirement.name]) == Version(floor), (
            requirement.name
        )


def test_core_wheel_selection():
    for path in CORE_TESTS:
        assert (ROOT / path).exists(), path
    assert "tests/methods/test_polyhedral_oracles.py" in CORE_TESTS
    assert "tests/methods/certification" in CORE_TESTS
