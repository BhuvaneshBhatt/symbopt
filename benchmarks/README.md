# Performance benchmarks

This directory contains a persistent, version-controlled benchmark corpus for performance-sensitive parts of `symbopt`. The suite records timings rather than asserting machine-specific thresholds.

Run the full warm-cache suite from the repository root:

```bash
PYTHONPATH=src python benchmarks/run_benchmarks.py --repeats 5 \
  --output benchmark-results.json
```

For cold-cache symbolic measurements:

```bash
PYTHONPATH=src python benchmarks/run_benchmarks.py --cold-cache --repeats 3
```

Filter by subsystem with repeated `--group` arguments, for example `--group sos --group moments`.

The benchmark corpus currently covers high-level QP dispatch, SOS compilation, moment-matrix construction, verified interval branch-and-bound, and the internal SDP solver. New performance-oriented changes should add or extend a benchmark case when they affect a distinct hot path.

Structural optimization workloads cover power substitution, weighted medians and generalised Rayleigh quotients. The `fallback` group measures an unmatched expression, so recognition overhead remains visible. Run both warm and cold cache profiles with `--group structural --group fallback`; timings depend on machine and dependency configuration.

`python benchmarks/univariate_analysis.py` measures optional `funcprops` integration, exact rational components, certificate replay and implicit extrema bounds. It runs against the installed packages and reports timings without a machine-dependent pass/fail threshold.

The `preprocessing` group measures exact rational endpoint conversion, repeated constraint reduction, quadratic equality slices, uncached interval preconditioning and exact KKT postprocessing. Pair this group with `--group interval` when comparing a change that affects both setup and branch-and-bound. Save the dependency versions and compare repeated samples on the same machine; wall-clock thresholds do not belong in the correctness suite.
