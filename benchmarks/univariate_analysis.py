"""Run optional function-analysis benchmarks against installed packages.

Execute from any directory with: python benchmarks/univariate_analysis.py
"""

import json
import statistics
from time import perf_counter

import funcprops as fp
import sympy as sp

from symbopt import certify_extrema_bounds, finite_component_minimum, minimize_result


def main():
    """Measure representative exact, open-domain and implicit-root work."""
    x = sp.Symbol("x", real=True)
    jobs = {
        "rational_components": lambda: finite_component_minimum(x * x + 1 / x**2, x),
        "rational_replay": lambda: finite_component_minimum(
            x * x + 1 / x**2, x, certified=True
        ),
        "asinh_endpoints": lambda: minimize_result(
            sp.asinh(x), [x >= -1, x <= 1], variables=x
        ),
        "implicit_extrema": lambda: certify_extrema_bounds(
            sp.exp(x) + x * x,
            x,
            sp.Interval(-1, 1),
            tolerance=sp.Rational(1, 10000),
            policy=fp.EnclosurePolicy(max_cells=160, max_depth=7, max_refinements=1),
        ),
    }
    results = {}
    for name, run in jobs.items():
        samples = []
        for _ in range(3):
            start = perf_counter()
            result = run()
            samples.append(perf_counter() - start)
        results[name] = {
            "median_seconds": statistics.median(samples),
            "samples_seconds": samples,
            "result_type": type(result).__name__,
        }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
