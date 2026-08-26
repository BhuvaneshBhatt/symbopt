import pytest

from pathlib import Path
import sys

pytestmark = pytest.mark.slow


def test_benchmark_suite_is_persistent_and_loadable():
    root = Path(__file__).resolve().parents[2]
    runner = root / "benchmarks" / "run_benchmarks.py"
    cases_dir = root / "benchmarks"
    cases = cases_dir / "cases.py"
    assert runner.exists()
    assert cases.exists()

    sys.path.insert(0, str(cases_dir))
    try:
        import cases as benchmark_cases_module

        names = {case.name for case in benchmark_cases_module.benchmark_cases()}
    finally:
        sys.path.pop(0)
    assert {
        "qp_dispatch",
        "sos_compile_quartic",
        "moment_matrix_order3",
        "interval_polynomial",
        "sdp_small_dense",
    }.issubset(names)
