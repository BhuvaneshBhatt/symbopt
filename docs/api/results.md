# Result and certificate types

## Integrated optimization

```{autoclass} symbopt.PipelineResult
:members:
```

`PipelineResult` is the canonical high-level result. The most important fields are `optimum_value`, `attained`, `certified`, `status`, `best_candidate`, and `certificate`. `optimization_sense` records whether the original request was a minimization or maximization.

KKT data is lazy: `kkt_system` can be `None` if an earlier route solved the problem. When present for a maximization result, `kkt_objective` records the normalized minimization objective represented by the stored equations.

Precision information is available through `working_precision` and `precision_provenance`. Solver-specific provenance can be inspected through `structural_result`, `semialg_result`, `sos_result`, `interval_result`, and `diagnostics`.

```{autoclass} symbopt.CandidateSolution
:members:
```

```{autoclass} symbopt.OptimalityCertificate
:members:
```

```{autoclass} symbopt.ProofArtifact
:members:
```

## Symbolic analysis

```{autoclass} symbopt.CriticalPointResult
:members:
```

```{autoclass} symbopt.KKTSystem
:members:
```

## Lifted problems

```{autoclass} symbopt.LiftedProblem
:members:
```

```{autoclass} symbopt.LiftRecord
:members:
```

```{autoclass} symbopt.BranchCase
:members:
```

## Solver diagnostics

```{autoclass} symbopt.SolverDiagnostic
:members:
```

```{autoclass} symbopt.DiagnosticKind
:members:
```

`PipelineResult.diagnostics` records typed failures from solver routes that were attempted but did not produce a usable result. Unsupported operations, numerical failures, certificate-recovery failures, and internal errors are distinguished.

## Exact SOS proof status

For SOS-backed results, exact proof status means the polynomial identity has been verified symbolically and every Gram block has passed exact PSD verification. Numerical SDP residuals alone do not imply exact certification.

## Integrated status semantics

Rigorous interval or SOS bounds do not automatically imply `certified=True`. `global_piecewise_infimum` can be exact even when no minimizer is attained. At certificate level, `proven_optimal` means an exact attained optimum, whereas `proven_infimum` means an exact infimum has been established without requiring attainment.

See {doc}`../getting-started/understanding-results` and {doc}`../guide/certification`.
