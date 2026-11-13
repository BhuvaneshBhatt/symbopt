"""Run maintained tests in a harness containing no source package."""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_TESTS = [
    "tests/api/test_compact_extremum_api.py",
    "tests/api/test_regions.py",
    "tests/api/test_region_analysis.py",
    "tests/methods/test_polyhedral_oracles.py",
    "tests/methods/test_audit_regressions.py",
    "tests/infrastructure/test_finite_candidates.py",
    "tests/infrastructure/test_reduction_semantics.py",
    "tests/methods/certification",
    "tests/methods/linear_programming",
    "tests/methods/quadratic_programming",
    "tests/methods/positivstellensatz",
    "tests/methods/intervals",
    "tests/methods/sos",
    "tests/methods/test_certificate_adversarial.py",
    "tests/methods/test_global_quadratic.py",
    "tests/tooling/test_mathematical_examples.py",
    "tests/tooling/test_capability_examples.py",
    "tests/tooling/test_dependency_profiles.py",
]


def main():
    """Check installed import provenance and run core or supported wheel tests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["core", "supported"], default="core")
    parser.add_argument("--package-root", type=Path, default=Path(sys.prefix))
    parser.add_argument("--junitxml", type=Path)
    args = parser.parse_args()
    spec = importlib.util.find_spec("symbopt")
    if spec is None or not Path(spec.origin).resolve().is_relative_to(
        args.package_root.resolve()
    ):
        raise RuntimeError("symbopt must be installed in the selected environment")
    print(
        json.dumps({"installed_package": spec.origin, "suite": args.suite}), flush=True
    )
    with tempfile.TemporaryDirectory(prefix="symbopt-wheel-tests-") as directory:
        harness = Path(directory)
        for name in [
            "tests",
            "tools",
            "docs",
            "examples",
            "notebooks",
            "benchmarks",
            ".github",
            "README.md",
            "LICENSE",
            "pyproject.toml",
            ".readthedocs.yaml",
        ]:
            source, target = ROOT / name, harness / name
            if source.is_dir():
                shutil.copytree(
                    source,
                    target,
                    ignore=shutil.ignore_patterns(
                        "__pycache__", ".pytest_cache", ".ruff_cache", "_build"
                    ),
                )
            else:
                shutil.copyfile(source, target)
        # pytest's source-tree pythonpath setting is overridden. Removing the
        # inherited path also prevents an editable checkout from shadowing the
        # installed wheel in the child process.
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["SYMBOPT_TEST_PACKAGE_ROOT"] = str(args.package_root.resolve())
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "tools.wheel_guard",
            "-o",
            "pythonpath=",
            "-m",
            "not unsupported and not reference_corpus",
        ]
        if args.junitxml:
            command.append("--junitxml=" + str(args.junitxml.resolve()))
        if args.suite == "core":
            command.extend(CORE_TESTS)
        else:
            command.extend(
                [
                    "--ignore=tests/tooling/test_source_hygiene.py",
                    "-k",
                    "not test_public_members_have_docstrings",
                ]
            )
        subprocess.run(command, cwd=harness, env=environment, check=True)


if __name__ == "__main__":
    main()
