import sys
from pathlib import Path

import pytest
import sympy as sp

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


@pytest.mark.parametrize(
    "name, expected",
    [
        ("power_substitution", 2),
        ("weighted_median", 4),
        ("generalized_rayleigh", 2),
        ("unmatched_convex", 2 - 2 * sp.log(2)),
    ],
)
def test_optimization_benchmark_values(name, expected):
    cases_dir = Path(__file__).resolve().parents[2] / "benchmarks"
    sys.path.insert(0, str(cases_dir))
    try:
        from cases import benchmark_cases

        result = next(case for case in benchmark_cases() if case.name == name).run()
    finally:
        sys.path.pop(0)
    assert result.certified and result.attained is True
    assert sp.simplify(result.optimum_value - expected) == 0
