# Precision and backend routing

`working_precision=N` is a package-level precision request and provenance
contract, not a promise that every third-party numerical routine supports
arbitrary precision.

Exact symbolic routes remain exact. For inexact problems, symbopt records input
precision and chooses routes that do not silently reduce a high-precision input
to float64. In particular, high-precision inexact problems bypass float64-only
LP/QP/SOS proposal paths when that conversion would violate the requested
precision semantics.

`PipelineResult.precision_provenance` records information such as the requested
precision, detected input precision, whether float64-only backends were allowed,
and the routing policy used.

Verified interval evaluation uses Arb precision where applicable. Symbolic and
`nsolve` fallbacks receive the requested working precision when supported.

Exact inputs are different: a float64 proposal stage may be used for speed when
the final candidate/value is independently reconstructed and exact-certified.
This does not turn a floating-point residual into a proof.
