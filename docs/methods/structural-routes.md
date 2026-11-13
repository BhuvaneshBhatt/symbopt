# Structural optimization routes

Before launching general KKT, SOS, semialgebraic, or interval machinery, `symbopt` looks for problem structures that admit a cheaper globally justified reduction. If a route precondition cannot be proved, the dispatcher falls through to the general solver pipeline.

## Linear-fractional programs

For an affine-over-affine objective

\[
\frac{a^T x+b}{c^T x+d}
\]

with affine constraints, `symbopt` uses a Charnes–Cooper transformation when the denominator sign is known. If the denominator can change sign, the feasible region is split into positive- and negative-denominator branches. Branch values are compared exactly; if exact ordering cannot be established, the route does not claim certification.

Strict constraints are handled through closure optimisation followed by an attainment check against the original problem. Recession solutions with transformed variable \(t=0\) can represent unattained finite limits or unbounded behavior.

## Monotone-composition reduction

If the objective is \(h(g(x))\) and the relevant real branch of \(h\) is known to be strictly monotone, optimization can be reduced to \(g\). Supported common cases include:

- `exp(g)`,
- `log(g)` on its positive domain,
- square roots and selected positive rational powers on their real domains,
- positive-base exponentials,
- odd positive integer powers.

Even powers are not treated as globally monotone.

## Exact commensurate trig optimization

One-variable rational expressions built from `sin`, `cos`, `tan`, `cot`, `sec`, and `csc` with modest rationally related affine frequencies are reduced to exact algebraic optimization. symbopt computes a common frequency, rewrites reciprocal trigonometric functions as sine/cosine quotients, and applies the tangent-half-angle substitution

\[
\sin u=\frac{2t}{1+t^2},\qquad
\cos u=\frac{1-t^2}{1+t^2}.
\]

Constant phases are accepted when their sine and cosine values simplify to exact algebraic numbers, including common rational multiples of `pi`. Rational-coefficient problems stay over `QQ`; algebraic coefficient fields are introduced only when a phase requires them.

After cancellation, real roots of the transformed denominator are isolated exactly. Each relevant pole is checked from both sides, and the omitted `t = +/- infinity` phase point is checked separately. This lets the route certify finite extrema, unattained `+/- infinity`, and bounded-domain endpoint limits without treating a pole as a stationary candidate. The derivative numerator supplies the remaining stationary algebraic candidates.

The route consumes the shared `UnivariateAnalysis`, so period/domain information comes from the same cached `funcprops` analysis used by the other univariate routes. The tangent-half-angle calculation and exact algebraic checks remain authoritative, so an inconclusive property query cannot create a certificate.

## Mixed periodic optimization

A different route handles genuinely incommensurate periodic components. If

\[
f(x)=g(f_1(x),\ldots,f_n(x)),
\]

all component functions are continuous and periodic, the feasible domain contains an unbounded interval, and the inverse periods are provably linearly independent over the rationals, the one-parameter orbit is dense in the Cartesian product of the component ranges. symbopt therefore optimizes `g` over those ranges and uses that reduced problem as a certified closure bound.

For example, `sin(x) + cos(sqrt(2)*x)` reduces to `u + v` on `-1 <= u,v <= 1`, giving the certified infimum `-2`. The route treats attainment as a separate theorem: a common preimage is reported when symbopt can prove one, and otherwise `PipelineResult.attained` remains `None`.

Component periods, real domains, discontinuities, and ranges are obtained through the same cached `UnivariateAnalysis` layer. `funcprops` is preferred and SymPy is a conservative fallback, so mixed-periodic decomposition no longer owns a parallel period/range recognition subsystem.

## Univariate domain, pole, and convexity analysis

All univariate structural routes share one lazy `UnivariateAnalysis` object. For common rational, logarithmic, exponential, radical, periodic, and composition problems it can reuse:

- the natural and user-restricted real domains,
- singularities and discontinuities,
- exact periodicity certificates,
- complete finite or indexed monotonic partitions,
- one-sided endpoint limits and attainment flags,
- stationary points inherited from complete derivative-root geometry,
- exact range conditions and extremum envelopes.

`funcprops` supplies these facts when it can certify them. Only the facts actually requested by a solver are computed, and result metadata never forces unrelated property queries. SymPy remains a conservative fallback. Specialized solvers still own the optimization proof itself: this is a shared analysis layer, not a monolithic univariate solver.

## Independent-variable decomposition

For additively separable objectives and compatible constraints, symbopt builds a decision-variable interaction graph. Disconnected components are solved independently and recombined only when every component result is certified. Coupling constraints suppress this decomposition.

For example,

\[
f(x,y)=f_1(x)+f_2(y)
\]

with independent constraints on \(x\) and \(y\) can be optimized componentwise. This route can combine different solver families, such as an exact polynomial component and a verified transcendental component.

## Relationship to the general pipeline

Structural routes are attempted early because they usually reduce symbolic expression growth and avoid unnecessary KKT, CAD, SDP, or branch-and-bound work. A structural result is preserved in `PipelineResult.structural_result`. If the route cannot prove its preconditions or cannot compare exact branch values, the problem continues through the normal pipeline.
