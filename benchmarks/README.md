# Performance benchmarks

This directory contains a persistent, version-controlled benchmark corpus for
performance-sensitive parts of `symbopt`. The suite records timings rather than
asserting machine-specific thresholds.

Run the full warm-cache suite from the repository root:

```bash
PYTHONPATH=src python benchmarks/run_benchmarks.py --repeats 5 \
  --output benchmark-results.json
```

For cold-cache symbolic measurements:

```bash
PYTHONPATH=src python benchmarks/run_benchmarks.py --cold-cache --repeats 3
```

Filter by subsystem with repeated `--group` arguments, for example
`--group sos --group moments`.

The benchmark corpus currently covers high-level QP dispatch, SOS compilation,
moment-matrix construction, verified interval branch-and-bound, and the internal
SDP solver. New performance-oriented changes should add or extend a benchmark
case when they affect a distinct hot path.
