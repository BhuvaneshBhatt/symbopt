# Known issues

These are interface inconsistencies, incomplete outputs and diagnostic limitations. The {doc}`capability-matrix` describes mathematical scope; {doc}`limitations` explains unsupported families.

| Area | Observed behavior | Useful improvement |
| --- | --- | --- |
| Boolean constraints | `constraints=False` is rejected by normalization; `region=False` accepts the empty domain. Explicit contradictory relations can represent infeasible branches. | Accept Boolean false consistently across branch-aware entry points. |
| Elimination helper | `solve_with_elimination` can return only the retained coordinate when direct solving fails. | Recover and validate the remaining coordinates, or expose partial solutions explicitly. |
| Degenerate local diagnostics | Directional Taylor signs are candidate evidence and can omit opposite-direction or curved-path behavior. | Classify odd leading orders and test independent curved paths before neighborhood claims. |
| Extreme box priorities | Floating-point width can overflow on a very large finite box and make an influence score uninformative. Bounds retain their enclosure meaning. | Scale width scores before ranking split coordinates. |
| Ill-conditioned moments | Numerical rank or separation can remain unresolved for an exact atomic measure. | Improve conditioning and exact reconstruction while retaining an unverified status on failure. |

Expected mathematical failures, oracle questions and resource-limited cases have separate meanings; see {doc}`expected-failures`. Unsupported or unfinished work must not be read as a proof of infeasibility.
