# Univariate optimization and reusable analysis

See {doc}`ranges-and-extrema` to choose between an image, bounds and an optimization result. Use this interface to distinguish an infimum from an attained minimum, retain excluded source points, or obtain rigorous value brackets when stationary coordinates have no closed form. symbopt owns optimization strategy and proof assembly. funcprops supplies domain records, real derivatives, root covers and outward enclosures through its public analysis interface. Install `symbopt[funcprops]` for these APIs.

| Task | Interface | Supported cases and limits |
| --- | --- | --- |
| Ordinary exact extrema | `minimize`, `maximize`, `minimize_result`, `maximize_result` | Existing symbolic portfolio; function analysis can precede algebraization for fixed exact univariate functions |
| Infimum and all minimizing components | `finite_component_minimum` | Parameter-free real algebraic rational functions on finite unions of intervals and finite sets; unresolved comparisons decline |
| Attained global-minimum proof | `certify_global_minimum` | Complete rational enumeration, or a real C2 interval with proved convexity and a verified stationary point |
| Rational infimum/supremum brackets | `certify_extrema_bounds` | Finite supported real components with checked C1 closure extensions; retained unresolved critical sites can widen bounds |
| Optimization certificate replay | `verify_univariate_certificate` | Exact registered record types, with optional source/coordinate binding |

```python
import sympy as sp
from symbopt import finite_component_minimum, certify_global_minimum

x = sp.Symbol("x", real=True)
r = finite_component_minimum(x * x, x, sp.Interval.open(1, 2))
assert r.infimum == 1 and r.minimizers is sp.S.EmptySet
assert r.complete and not r.attained and not r.certified
assert r.verify()
assert certify_global_minimum(x * x, x, sp.Interval.open(1, 2)) is None

raw = sp.Mul(x, sp.Pow(x, -1, evaluate=False), evaluate=False)
r = finite_component_minimum(raw, x, certified=True)
assert r.infimum == 1 and r.minimizers == sp.S.Reals - sp.FiniteSet(0)
assert r.certified
```

`certified=False` is the default for `finite_component_minimum`; pass `certified=True` to replay the completed enumeration before setting that flag. Calling a `certify_*` function explicitly requests proof work. A certificate object is evidence to check, not a substitute for successful replay. The rational certificate replays exact candidate enumeration; curvature certificates check sufficient global premises. These routes trust SymPy’s exact symbolic operations.

```python
import funcprops as fp
from symbopt import certify_extrema_bounds, verify_univariate_certificate

proof = certify_extrema_bounds(
    sp.exp(x) + x * x,
    x,
    sp.Interval(-1, 1),
    tolerance=sp.Rational(1, 10000),
    policy=fp.EnclosurePolicy(max_cells=160, max_depth=7, max_refinements=1),
)
assert verify_univariate_certificate(proof, expr=sp.exp(x) + x * x, var=x)
assert proof.complete
assert proof.minimum_bounds[1] - proof.minimum_bounds[0] <= proof.tolerance
```

The funcprops verifier raises `TypeError` for these optimization records. The extrema verifier replays stored domain, root-cover, site and outward-enclosure premises without calling the optimization or root proposers. `coverage_complete`, `critical_coverage_complete`, `accuracy_complete` and `exact_complete` answer different questions. Valid coarse bounds can verify while requested accuracy remains unmet. Bounds do not assert attainment or supply a minimizing coordinate. An empty effective domain has no finite extrema bounds; unbounded components, unsupported regularity and inexact inputs decline this enclosure route. A supplied `critical_covers` tuple can reuse independently checked funcprops covers.

An `EnclosurePolicy(seconds=...)` runs extrema-bound assembly in a separate process. A deadline returns the most recent checked checkpoint when one exists; otherwise the result is incomplete, has no value bounds, and records `work-budget-exhausted`. Worker transport preserves held source syntax. Replay does not need to rerun discovery. `certify_global_minimum(work_budget=...)` uses the same process boundary and returns `None` if no completed certificate is ready before the deadline.

Ordinary univariate analysis keeps unknown image membership and attainment as `None`. Endpoint substitution can recover an actual optimizer before a general equation solver is used. Parameter guards and semantic predicates that cannot be converted to real sets remain unresolved. The package does not infer a global proof from a partial root list or a local numerical candidate.

Run [examples/univariate_extrema.py](https://github.com/BhuvaneshBhatt/symbopt/blob/main/examples/univariate_extrema.py) against an installed wheel. See the [API reference](../api/univariate-analysis.md) for record fields and signatures and the [funcprops contract](https://github.com/BhuvaneshBhatt/funcprops/blob/main/docs/analysis-interface.md) for the shared primitives.

## Sharing analysis evidence

funcprops' public `critical_value_regions` owns the finite-cell endpoint, critical and unresolved region enumeration consumed by `certify_extrema_bounds`. The driver, its error tolerance, infimum/supremum assembly and optimization replay remain in symbopt. Supply `critical_covers` in source-cell order to reuse checked derivative-root coverage. Source expression, coordinate, requested interior and effective domain must match; a cover for a simplified source does not erase the original source's holes.

funcprops also exposes `analyze_range`, `refine_range` and exact `CriticalValue` expressions. These image-analysis results are not symbopt optimization certificates. Completion of an image does not establish automatic support for extracting every implicit minimizing coordinate. funcprops supports one-sided endpoint cells, exact image conditions and selected nonmonotone tail bounds. Its analytic parameter transitions beyond rational quadratics and more general tail facts remain incomplete. The optimizer-bound driver still accepts native closure-smooth root coverage, not the separate interior critical-cover type.
