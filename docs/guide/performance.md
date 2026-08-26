# Performance guidance

Performance depends strongly on problem structure. The most effective user-level
choices are usually structural rather than numerical micro-tuning:

- provide finite, reasonably tight boxes for verified interval problems,
- preserve polynomial structure where possible,
- start SOS hierarchy search at low order,
- formulate separable/sparse polynomial problems transparently,
- avoid unnecessary symbolic expansion before calling `minimize()`.

For implementation details, scaling behavior, caches, SOS basis growth, interval
reuse, and SDP workspace reuse, see {doc}`../reference/performance`.

For the role of solver tolerances and which quantities can affect rigorous
certification, see {doc}`../reference/numerical-tolerances`.
