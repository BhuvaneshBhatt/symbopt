# Verified interval optimization

The interval optimizer is a package-owned symbolic branch-and-bound engine with
`python-flint`/Arb as the primary rigorous backend for supported
transcendentals.

## Bound provenance

Every enclosure records whether it is rigorous and which method produced it.
Only rigorous lower bounds are permitted to participate in global
certification.

## Polynomial bounds

Polynomial boxes can use exact Bernstein coefficient bounds. This avoids the
unsound practice of treating corner samples as interval enclosures.

For example, on \(x\in[-1,1]\),

\[
x^2\in[0,1],
\]

not `[1, 1]`.

## Arb real balls

For supported transcendental expressions, variable intervals are converted to
Arb balls and evaluated through a compiled expression graph. Outward lower and
upper ball endpoints give rigorous real enclosures.

## Tightening

The engine can intersect several rigorous bounds:

- natural Arb evaluation,
- derivative-based monotonicity,
- centered Taylor enclosures,
- exact polynomial Bernstein bounds.

Equality systems also receive verified contraction before subdivision. Scalar
equalities use interval Newton steps; square coupled systems use Krawczyk
operators. A contraction is accepted only from rigorous interval evaluations,
so feasible zeros cannot be discarded.

## Singularities

A box crossing a pole or domain boundary is marked uncertain and subdivided.
A box entirely outside the real domain is rejected.

Examples include \(1/x\) across zero, `log(x)` on negative boxes, and tangent
across a pole.

## Branch-and-bound

Parent enclosures are reused in child boxes when the split variable is absent
from the expression, reducing repeated evaluation.

## Further reading

See Moore--Kearfott--Cloud and Johansson in {doc}`../reference/references` for the mathematical background.


## Supported Arb function families

The compiled real-ball evaluator covers elementary trigonometric and hyperbolic
functions plus a broader set of special functions exposed by python-flint/Arb,
including error functions, exponential/sine/cosine/hyperbolic integrals,
log-gamma/digamma, zeta, the real Lambert W branches 0 and -1, polylogarithms,
Bessel functions, Airy functions, Fresnel integrals, sinc, and generalized
exponential integrals. Unsupported real branches remain unsupported rather than
being coerced through complex arithmetic. If a particular installed python-flint build does not expose a requested special-function method, the evaluator falls back conservatively rather than treating an unavailable implementation as a finite enclosure; unsupported boxes therefore remain rigorous even when they are less informative.

## Symbolic preconditioning and split scoring

Before interval evaluation, the engine constructs a small set of algebraically
equivalent forms using conservative factor/cancel/trigonometric simplification.
Transformed forms are intersected with the original enclosure only after the
original expression has been proved domain-valid on the box, so cancellation
cannot erase an original singularity.

Branch selection combines box width, objective derivative magnitude/uncertainty,
constraint sensitivity, unresolved feasibility, domain uncertainty, known
trigonometric breakpoints, and low-degree symbolic stationary points. This
focuses subdivision on variables that are most likely to tighten the objective
or resolve feasibility.
