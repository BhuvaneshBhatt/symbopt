# Semidefinite programming

The internal SDP representation supports block PSD variables \(X_k\succeq0\)
and optional nonnegative scalar variables.

A typical model is

\[
\min
\sum_k\langle C_k,X_k\rangle+c^Tz
\]

subject to affine equality constraints.

## Dense solver

The dense solver uses a factorized penalty parameterization for PSD blocks,
problem scaling, warm starts, and a reusable workspace for the linear operators
and dual normal system.

## Sparse structure

PSD block structure can originate upstream in the SOS compiler. Disconnected
correlative components and overlapping chordal cliques are converted into Gram
blocks before numerical solution.

For a generic single sparse PSD block, the sparse route forms a chordal extension,
allocates each objective/constraint coefficient to one containing maximal clique,
and adds affine equality rows for entries duplicated across clique separators.
This is a PSD-completion formulation rather than a heuristic block split. Returned
numerical matrices are reconstructed in the original coordinates; as elsewhere,
numerical SDP status is not an exact symbolic certificate.

## Scaling and returned certificates

Primal and dual quantities are transformed back to the original problem scaling
before they are returned, so reported residuals refer to the original SDP
coordinates.

## Numerical status

An SDP numerical result is not automatically an exact symbolic certificate.
SOS certification applies an additional exact-recovery/verification layer.

## Further reading

See Vandenberghe--Boyd and Alizadeh in {doc}`../reference/references` for the mathematical background.

