# Performance reference

The main performance costs differ by solver family.

## Symbolic solving

Stationary/KKT equation solving and Gröbner elimination can grow rapidly with
degree and variable count. The high-level pipeline therefore routes recognized
LP/QP problems before KKT work, shares one prioritized active-set plan between
exact and numerical candidate search, and skips the numerical KKT fallback only
when exact enumeration is known to be complete. Local critical-point
classification is performed only for feasible candidates rather than by solving
a separate global critical-point system first.

## Shared symbolic caches

The package caches reusable problem structure as well as symbolic primitives.
`LiftedProblem` retains original-variable, normalized-residual, polynomiality,
and constraint metadata so downstream stages do not repeatedly reconstruct it.
The shared symbolic caches include:

- monomial exponent tuples,
- monomial bases,
- polynomial term maps,
- selected canonical expansions,
- gradients and Hessians.

Use:

```python
from symbopt import symbolic_cache_info, clear_symbolic_caches
```

to inspect or reset caches in benchmarks and long-lived sessions.

## SOS compilation

The SOS compiler assembles SDP coefficient rows directly from exponent sums.
Exponent tuples are the canonical internal monomial representation; symbolic
monomial expressions are retained mainly at API/certificate boundaries. It does
not build and re-expand a symbolic \(z^TQz\) expression for every relaxation.

Support-aware and correlative-sparsity pruning reduce Gram basis size where the
implemented structural rules permit it.


## Early SOS block propagation

Disconnected correlative components are turned into separate Gram/PSD blocks
before the SDP is constructed. This reduces matrix dimensions earlier than a
post-hoc sparse decomposition and avoids creating known-zero cross-component
Gram entries.

\[
\text{support}\to\text{components}\to\text{Gram blocks}\to\text{PSD blocks}.
\]

High-precision arithmetic is concentrated in exact certificate recovery after a
numerical SDP candidate has been found. The entire SDP is therefore not rerun at
arbitrary precision merely to obtain an exact rational certificate.

## Moment methods

Moment and localizing matrices operate on exponent tuples internally, avoiding
repeated conversion of monomials back through symbolic polynomial parsing.

## Verified intervals

The interval engine compiles expression dependency information, equality
corrections, and trigonometric split structure once and reuses parent bounds
when a child split does not affect an expression.

Polynomial boxes can use exact Bernstein bounds. Supported transcendental boxes
use Arb evaluation plus derivative/Taylor tightening where applicable.

## SDP numerics

A reusable SDP workspace stores scaled linear operators, block tensors, and a
normal-system factorization across penalty iterations and restarts. The
factorized penalty solver supplies analytic gradients to L-BFGS-B and evaluates
primal/dual linear operators with vectorized NumPy operations.

## User-facing performance advice

- Provide finite, reasonably tight bounds for interval problems.
- Avoid unnecessary symbolic expansion before calling `minimize()`.
- Use lower SOS hierarchy orders first.
- Exploit natural separability in polynomial formulations.
- Expect dense SOS/SDP methods to become expensive as monomial-basis size grows.
- Clear caches only when reproducibility/memory pressure requires it; repeated
  calls often benefit from retained symbolic structure.

## Persistent benchmark corpus

The repository tracks representative benchmark cases for high-level QP
dispatch, SOS compilation, moment-matrix construction, verified interval
branch-and-bound, and the internal SDP solver. Result JSON files use a stable
schema but timing thresholds are not universal because hardware and BLAS
implementations differ.

