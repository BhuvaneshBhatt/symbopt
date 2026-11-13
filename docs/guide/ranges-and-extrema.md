# Ranges, bounds and extrema

Use an image computation to ask which outputs are attainable. Use optimization to ask for an infimum, a minimizing point or a global proof. These questions share domain and critical-point analysis, but their return values have different meanings.

| Question | Package and entry point | What the answer establishes |
| --- | --- | --- |
| Which outputs occur for an algebraic graph? | semialg `function_range`, exact modes | An exact supported image formula, including holes and disconnected components |
| Which outputs occur for a supported transcendental source? | funcprops `range` or explicit `analyze_range` | An exact image, guarded condition or partial image information |
| What interval encloses the output values? | semialg `function_range(method="bounds")` | Extremal interval bounds; missing interior values may be included |
| What does the native exact semialgebraic optimizer establish? | semialg `semialgebraic_minimize` / `semialgebraic_maximize` | Supported exact real optimum/infimum values, attainment and witnesses |
| What is the infimum and where is it attained in the general portfolio? | symbopt `minimize_result`, supported `finite_component_minimum` | An optimization result with its own value, attainment and completion contract |
| What are rigorous extremal value brackets at implicit stationary points? | symbopt `certify_extrema_bounds` | Replayable infimum/supremum brackets, with independent coverage and accuracy flags |

## An infimum can be outside the image

```python
import sympy as sp
import funcprops as fp
from symbopt import finite_component_minimum

x, y = sp.symbols("x y", real=True)
D = sp.Interval.open(1, 2)
image = fp.condition_to_real_set(fp.range(x * x, x, y, domain=D), y)
assert image == sp.Interval.open(1, 4)
minimum = finite_component_minimum(x * x, x, D)
assert minimum.infimum == 1
assert minimum.minimizers is sp.S.EmptySet
assert minimum.complete and not minimum.attained and not minimum.certified
```

The value 1 is the infimum and is not an attainable output on this domain. A finite range endpoint must carry its ownership. A held removable hole can remove the only point attaining a value; a source-domain record must survive any normalization used in optimization.

`certified=False` is the default for `finite_component_minimum` and the new funcprops image-analysis interfaces. Requesting replay is separate from mathematical completion. Explicit `certify_*` functions request proof work. A certificate must be replayed with its owning package's verifier.

## Reusing critical coverage

```python
from symbopt import certify_extrema_bounds, verify_univariate_certificate

f = x * x + sp.exp(x)
D = sp.Interval(-1, 1)
policy = fp.EnclosurePolicy(max_cells=160, max_depth=7, max_refinements=1)
analysis = fp.analyze_range(f, x, D, policy=policy)
assert analysis.complete and analysis.verify()
assert isinstance(analysis.image.start, fp.CriticalValue)

proof = certify_extrema_bounds(
    f,
    x,
    D,
    critical_covers=tuple(cell.critical_cover for cell in analysis.cells),
    tolerance=sp.Rational(1, 10000),
    policy=policy,
)
assert verify_univariate_certificate(proof, expr=f, var=x)
assert proof.complete
assert proof.minimum_bounds[1] - proof.minimum_bounds[0] <= proof.tolerance
```

The image endpoint is an exact implicit number. symbopt reuses the derivative-root covers and builds outward extremal brackets under its own tolerance. The shared `critical_value_regions` interface binds endpoint, critical and unresolved regions to their source. Value-bound completion does not assert a minimizing coordinate or attainment. This general bound route requires finite components with checked C1 closure extensions.

`RangeAnalysis.verify()` or funcprops' verifier checks the image-analysis record. `verify_univariate_certificate` checks symbopt's optimization records. A partial image can verify while remaining incomplete; a coarse optimization proof can verify while its requested accuracy remains unmet.

## Choosing an analysis route

funcprops' automatic `range` keeps existing fast routes first and uses a bounded analytic fallback, including supported periodic and monotone/nonmonotone tail transfers. Its finite-domain quadratic parameter rules also participate in automatic dispatch for real coefficients. Explicit `analyze_range` adds configurable policy, partial information, exact condition output and supplied tail facts; `analyze_parameter_range` adds family transitions and replay metadata. An exact image in funcprops does not imply that symbopt automatically returns its implicit argmin.

semialg's exact `qe`, `cad` and `auto` range modes preserve disconnected outputs and fail explicitly when the projection is unsupported. The explicit `bounds` and `optimization` modes return extremal interval enclosures. Use `return_result=True` and inspect `diagnostics["range_kind"]` when consuming either kind.

A dense periodic image needs special care. An extremal bound on the closure of a trajectory does not decide whether simultaneous phases reach that bound. symbopt can retain unknown attainment; an exact image computation must decide endpoint membership or keep it unresolved.

## Package direction and remaining work

funcprops owns function properties, source domains, root coverage, value enclosures and images. semialg owns exact real algebraic projection, signs, inequalities, native semialgebraic optimization and parameter conditions. symbopt owns optimizer strategies and optimization certificate assembly. It uses funcprops through an optional extra; funcprops has no reverse dependency on symbopt.

The local asymptotic package already requires funcprops. Its future contribution to image analysis should pass source-bound limit and remainder facts through an adapter with explicit tail cutoffs. A leading term or a numerical trend is insufficient to exclude a tail from global optimization.

See the [funcprops range tutorial](https://github.com/BhuvaneshBhatt/funcprops/blob/main/docs/ranges.md), [real image methods and proof obligations](https://github.com/BhuvaneshBhatt/funcprops/blob/main/docs/design/real-images.md), [semialg exact-image guide](https://github.com/BhuvaneshBhatt/semialg/blob/main/docs/range-images.md) and {doc}`univariate-analysis` for the optimization API.

Native `CoverageProof` records from closure-smooth image cells can be supplied as `critical_covers`. The newer `InteriorCriticalCover` route uses one-sided endpoints and declared symbolic backend replay; it is not a `certify_extrema_bounds` input. That optimizer-bound driver retains its checked C1-closure contract. An exact `ConditionSet` image is a mathematical answer even when its implicit endpoints cannot be sorted into a canonical interval.
