# Exact versus numerical computation

`symbopt` deliberately mixes symbolic and numerical computation, but the two
serve different roles.

## Exact computation

Exact algebra is preferred for:

- stationary equations when solvable symbolically,
- rational/algebraic candidate coordinates,
- polynomial identities,
- exact Gram recovery,
- exact moment identities,
- algebraization constraints.

## Numerical computation

Numerical methods are useful for:

- finding candidate roots,
- solving LP/QP/SDP subproblems,
- guiding SOS hierarchy search,
- estimating promising branches,
- ordering candidates when exact comparison is unavailable.

A numerical success status does not automatically become a proof.

## Rigorous numerics

Arb real-ball arithmetic occupies a third category: it is numerical but
mathematically rigorous because the returned ball is guaranteed to contain the
true value of the represented expression under the evaluator's supported
semantics.

This distinction is reflected in interval-bound provenance.

## High-precision exact recovery

SOS certificate recovery uses numerical solutions as targets for an exact affine
reconstruction problem. The package may reevaluate the affine fit at increasing
precision and try larger rational-denominator budgets. These operations improve
the chance of identifying an exact nearby certificate; they do not weaken the
final proof condition.

The recovered result is accepted only after exact symbolic residual and PSD
checks succeed.



## Exact semialgebraic reasoning

When installed, `semialg` adds exact RUR and CAD/QE decisions to the symbolic
side of the pipeline. These are exact proof mechanisms rather than floating
numerical fallbacks.
