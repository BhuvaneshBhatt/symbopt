# Semidefinite optimization API

```{autoclass} symbopt.SDPSolverConfig
:members:
```

```{autofunction} symbopt.solve_sdp
```

Advanced matrix/result types:

```{autoclass} symbopt.SDPProblem
:members:
```

```{autoclass} symbopt.SDPResult
:members:
```

```{autofunction} symbopt.solve_sdp_dense
```

```{autofunction} symbopt.solve_sdp_sparse_chordal
```

The sparse route supports disjoint blocks and overlapping chordal-clique PSD-completion formulations with explicit separator consistency equalities. See {doc}`../methods/semidefinite-programming`.
