# Numerical tolerances and rigor

`symbopt` uses numerical tolerances for search and numerical acceptance, but a
floating-point tolerance is **not** allowed to silently turn heuristic evidence
into an exact proof.

## Categories of tolerances

### Candidate feasibility

Candidate-generation routines use small numerical tolerances when deciding
whether a floating-point candidate approximately satisfies equalities and weak
inequalities.

These tolerances affect **candidate acceptance**, not exact global
certification.

### Numerical-realness

Complex numerical roots whose imaginary part is below a small threshold can be
treated as numerically real candidates. Exact symbolic candidates are not
subject to that floating-point test.

### Hessian/eigenvalue tests

Convex-QP routing and local classification use eigenvalue tolerances to protect
against roundoff near zero. A matrix with a clearly negative eigenvalue is not
treated as positive semidefinite.

### SDP tolerances

SDP solver controls include feasibility and numerical regularization parameters.
They affect numerical convergence and approximate relaxation quality.

A small SDP residual does **not** by itself establish an exact SOS certificate.

### SOS certificate recovery

Exact SOS certification requires exact polynomial identity recovery and exact
positive-semidefinite checks on recovered Gram matrices. Numerical thresholds
may decide whether exact-recovery is attempted, but the final proof conditions
are exact.

### Verified intervals

The interval branch width tolerance controls subdivision termination/search
resolution. A rigorous bound can contribute to proof evidence, but numerical closeness to a candidate does not by itself set `certified=True`. Exact global certification requires exact equality/proof semantics; otherwise the rigorous gap is reported.

Arb and exact Bernstein enclosures are rigorous independently of ordinary
floating-point candidate tolerances.

## Guidance

If you change a tolerance, first ask which category it belongs to:

| Tolerance role | Can change search? | Can change candidate acceptance? | Can create proof by itself? |
|---|---:|---:|---:|
| root/feasibility tolerance | Yes | Yes | No |
| eigenvalue routing tolerance | Yes | Yes | No |
| SDP residual tolerance | Yes | Yes | No |
| exact reconstruction threshold | Yes | Indirectly | No |
| interval subdivision tolerance | Yes | Yes | No; rigor comes from the enclosure |
| exact symbolic verification | N/A | N/A | Yes |

The package should prefer an inconclusive result over a falsely certified result.
