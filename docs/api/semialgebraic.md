# Semialgebraic backend API

The optional `semialg` dependency provides exact CAD/QE/RUR and polynomial optimization capabilities. symbopt discovers the installed backend's API and result capabilities at runtime rather than assuming a fixed top-level module layout.

Install it with:

```bash
python -m pip install "symbopt[semialg]"
```

## High-level policy

`minimize()` accepts:

```python
semialg_certification = "auto"  # default
semialg_certification = "candidate"
semialg_certification = "complete"
```

and optionally `semialg_range_cost_limit=...`.

`"auto"` is cost-aware. The backend is not necessarily invoked before symbolic KKT/elimination: small polynomial systems can be cheaper to solve symbolically, whereas suitable semialgebraic problems can route directly to semialg. Explicit `"candidate"` or `"complete"` requests pass that policy through when supported by the installed backend.

A certified direct optimizer result preserves:

- exact value,
- attainment,
- backend method,
- backend diagnostics,
- backend version and discovered capabilities.

An uncertified backend candidate may contribute a feasible candidate but is not used as a rigorous lower bound.

## Capability discovery

```{autofunction} symbopt.get_semialg_capabilities
```

```{autoclass} symbopt.SemialgCapabilities
:members:
```

The adapter probes the actual decision/optimization APIs used by symbopt. This allows backend functions to live in semialg submodules without requiring top-level re-exports.

```{autoclass} symbopt.SemialgRangeResult
:members:
```

```{autoclass} symbopt.SemialgProof
:members:
```

## Direct exact optimization

```{autofunction} symbopt.semialgebraic.minimize_exact
```

```{autoclass} symbopt.SemialgOptResult
:members:
```

There is no hard-coded one/two-variable restriction in the adapter; dimensional support is determined by the installed backend and its certification policy.

## Exact feasibility, implication, sign, and range

```{autofunction} symbopt.semialg_available
```

```{autofunction} symbopt.semialgebraic.certify_global_min
```

```{autofunction} symbopt.function_range_exact
```

```{autofunction} symbopt.prove_nonnegative_exact
```

These helpers are useful independently and as fallbacks for backend APIs that do not provide a direct optimizer result. The high-level pipeline avoids redundant repetition once a direct optimizer has authoritatively answered the same question.
