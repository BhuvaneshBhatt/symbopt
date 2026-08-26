# Structural optimization routes

Before launching general KKT, SOS, semialgebraic, or interval machinery,
`symbopt` looks for problem structures that admit a cheaper globally justified
reduction. These routes are conservative: if a precondition cannot be proved,
the dispatcher falls through to the general solver pipeline.

## Linear-fractional programs

For an affine-over-affine objective

\[
\frac{a^T x+b}{c^T x+d}
\]

with affine constraints, symbopt uses a Charnes--Cooper transformation when the
denominator sign is known. If the denominator can change sign, the feasible
region is split into positive- and negative-denominator branches. Branch values
are compared exactly; if exact ordering cannot be established, the route does
not claim certification.

Strict constraints are handled through closure optimization followed by an
attainment check against the original problem. Recession solutions with
transformed variable \(t=0\) can represent unattained finite limits or
unbounded behavior.

## Monotone-composition reduction

If the objective is \(h(g(x))\) and the relevant real branch of \(h\) is known
to be strictly monotone, optimization can be reduced to \(g\). Supported common
cases include:

- `exp(g)`,
- `log(g)` on its positive domain,
- square roots and selected positive rational powers on their real domains,
- positive-base exponentials,
- odd positive integer powers.

Even powers are not treated as globally monotone.

## Exact trigonometric polynomials

One-variable trigonometric polynomials with modest rationally related
frequencies are reduced to algebraic optimization. symbopt computes a common
frequency and applies the tangent-half-angle substitution

\[
\sin u=\frac{2t}{1+t^2},\qquad
\cos u=\frac{1-t^2}{1+t^2}.
\]

The stationary equation becomes a polynomial equation in \(t\). Real algebraic
roots are retained exactly, including `CRootOf` representations when radicals
are not appropriate. The route certifies a result only when candidate values
can be ordered exactly.

## Univariate domain, pole, and convexity analysis

For common univariate rational, logarithmic, exponential, and radical
expressions, symbopt analyzes the real continuous domain before generic search.
It can use:

- finite singularities and poles,
- one-sided limits,
- limits at infinity,
- exact first and second derivatives,
- convexity/concavity,
- exhaustive stationary points on domain components.

This can prove finite extrema, unattained infima/suprema, or \(\pm\infty\)
without interval subdivision.

## Independent-variable decomposition

For additively separable objectives and compatible constraints, symbopt builds a
decision-variable interaction graph. Disconnected components are solved
independently and recombined only when every component result is certified.
Coupling constraints suppress this decomposition.

For example,

\[
f(x,y)=f_1(x)+f_2(y)
\]

with independent constraints on \(x\) and \(y\) can be optimized componentwise.
This route can combine different solver families, such as an exact polynomial
component and a verified transcendental component.

## Relationship to the general pipeline

Structural routes are attempted early because they usually reduce symbolic
expression growth and avoid unnecessary KKT, CAD, SDP, or branch-and-bound work.
A structural result is preserved in `PipelineResult.structural_result`. If the
route cannot prove its preconditions or cannot compare exact branch values, the
problem continues through the normal pipeline rather than being approximated
silently.
