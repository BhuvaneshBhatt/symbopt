# Semidefinite programming

The internal SDP representation supports block PSD variables \(X_k\succeq0\) and optional nonnegative scalar variables.

A typical model is

\[
\min
\sum_k\langle C_k,X_k\rangle+c^Tz
\]

subject to affine equality constraints.

## Dense solver

The dense internal solver uses a factorized penalty parameterization for PSD blocks, problem scaling, warm starts, and a reusable workspace for the linear operators and dual normal system.

`SDPSolverConfig(backend="auto")` can use an installed CVXPY conic solver for larger blocks. `backend="internal"` forces the package solver and `backend="cvxpy"` explicitly requests the external adapter. `external_solver` may name a particular installed CVXPY solver. External solves supply numerical candidates only; SOS exact recovery and certificate verification are unchanged.

## Sparse structure

PSD block structure can originate upstream in the SOS compiler. Disconnected correlative components and overlapping chordal cliques are converted into Gram blocks before numerical solution.

For a generic single sparse PSD block, the sparse route forms a chordal extension, allocates each objective/constraint coefficient to one containing maximal clique, and adds affine equality rows for entries duplicated across clique separators. This is a PSD-completion formulation rather than a heuristic block split. Returned numerical matrices are reconstructed in the original coordinates; as elsewhere, numerical SDP status is not an exact symbolic certificate.

## Scaling and returned certificates

Primal and dual quantities are transformed back to the original problem scaling before they are returned, so reported residuals refer to the original SDP coordinates.

## Numerical status

An SDP numerical result is not automatically an exact symbolic certificate. SOS certification applies an additional exact-recovery/verification layer.

## Further reading

See Vandenberghe–Boyd and Alizadeh in {doc}`../reference/references` for the mathematical background.

## External numerical SDP backends and exact post-verification

This page is the canonical description of the external SDP boundary. The internal SDP implementation is useful for small research problems. For larger proposal solves, `SDPSolverConfig(backend="cvxpy")` can use an installed conic solver such as Clarabel, SCS, CVXOPT, or MOSEK.

```python
from symbopt import SDPSolverConfig

config = SDPSolverConfig(
    backend="cvxpy",
    external_solver="CLARABEL",
    tol=1e-8,
)
```

A successful numerical solve is **candidate evidence only**. For SOS and related polynomial certificates the path is:

```text
external conic solver
→ numerical Gram/multiplier candidate
→ exact affine reconstruction
→ exact polynomial identity check
→ exact PSD check for every Gram block
→ verified certificate
```

Consequently, changing Clarabel to SCS can change floating-point iterates without changing the final exact certificate. A numerical `optimal` status never bypasses exact post-verification.
