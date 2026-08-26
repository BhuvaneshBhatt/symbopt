# Utility API

## Bound inference

```{autofunction} symbopt.infer_box_bounds
```

## Symbolic caches

```{autofunction} symbopt.cache_info
```

```{autofunction} symbopt.clear_symbolic_caches
```

The cache utilities are primarily useful for benchmarking, long-lived
interactive sessions, and reproducible performance measurements.

## Polynomial structure

```{autofunction} symbopt.build_sparsity_graph
```

```{autofunction} symbopt.build_symmetry_report
```

These helpers expose the variable-interaction and lightweight symmetry analyses
used by sparse polynomial/SOS routing.
