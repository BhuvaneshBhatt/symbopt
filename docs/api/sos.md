# Sum-of-squares API

```{autofunction} symbopt.sos_decompose
```

```{autofunction} symbopt.sos_lower_bound
```

```{autofunction} symbopt.sos_scalar_relaxation
```

```{autofunction} symbopt.sos_poly_relaxation
```

```{autofunction} symbopt.search_sos_hierarchy
```

Important result types:

```{autoclass} symbopt.SOSResult
:members:
```

```{autoclass} symbopt.SOSRelaxationResult
:members:
```

```{autoclass} symbopt.SOSCertificate
:members:
```

See {doc}`../methods/sum-of-squares` for coupled high-precision exact recovery, multi-block certificate semantics, and exact PSD requirements.

## `SOSRelaxationResult` status semantics

For constrained relaxations, the result separates solver progress from proof status:

- `status="certified"`: an exact polynomial identity was recovered and every Gram block was verified exactly positive semidefinite.
- `status="numerically_feasible"`: the SDP relaxation produced a usable numerical iterate, but no exact certificate was recovered.
- `status="infeasible_or_failed"`: no usable relaxation bound was obtained.
- `status="unsupported"`: the requested relaxation is outside the implemented class.

`success` means the numerical relaxation produced a usable result. `certified` is stronger. The symbolic `lower_bound` field is populated only for a certified bound; approximate numerical values are exposed through `lower_bound_numeric`.
