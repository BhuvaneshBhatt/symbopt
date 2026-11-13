from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

from cases import benchmark_cases

from symbopt import clear_symbolic_caches


def _time_case(case, repeats: int, cold_cache: bool):
    samples = []
    for _ in range(repeats):
        if cold_cache:
            clear_symbolic_caches()
        start = time.perf_counter()
        case.run()
        samples.append(time.perf_counter() - start)
    return {
        "minimum_seconds": min(samples),
        "median_seconds": statistics.median(samples),
        "mean_seconds": statistics.fmean(samples),
        "samples_seconds": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run persistent symbopt benchmarks.")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--cold-cache", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    selected = [
        case
        for case in benchmark_cases()
        if not args.group or case.group in set(args.group)
    ]
    results = {
        "schema_version": 1,
        "python": sys.version,
        "platform": platform.platform(),
        "cold_cache": args.cold_cache,
        "repeats": args.repeats,
        "benchmarks": {},
    }
    for case in selected:
        entry = _time_case(case, args.repeats, args.cold_cache)
        entry["group"] = case.group
        entry["description"] = case.description
        results["benchmarks"][case.name] = entry
        print(
            f"{case.name:24s} "
            f"median={entry['median_seconds']:.6f}s "
            f"min={entry['minimum_seconds']:.6f}s"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
