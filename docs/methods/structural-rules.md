# Specialized optimization rules

These rules are used to obtain an exact answer from recognizable structure before constructing a large KKT system, SOS relaxation, or quantifier-elimination problem. The dispatcher checks each rule's mathematical conditions and preserves the original domain. An unrecognized sign, unsupported shape or failed witness check leaves the problem to the ordinary solver portfolio.

## Coverage and proof conditions

| Family | Recognized cases | Required proof |
| --- | --- | --- |
| Leading terms and rays | Rational asymptotic leading terms on coordinate and diagonal affine rays | Objective tends to minus infinity and every original constraint holds eventually |
| Visible nonnegative forms | Positive weighted even powers, real absolute values and structurally nonnegative terms | All nonnegative terms vanish at a checked feasible point |
| Affine equalities | Small affine systems with a fixed real algebraic coefficient matrix | A real parameterization covers the equality locus; transformed constraints are retained |
| Power substitutions | One-variable rational functions whose exponents share a divisor | Correct real image, transformed constraints and all applicable real inverse branches |
| Affine ridge functions | Dependence on one affine form, with constraints on that form or a finite product box | Exact image interval and a checked map back to the original variables |
| Radial functions | Dependence on the squared Euclidaen radius, with radial constraints | Verified functional identity and an explicit nonnegative image domain |
| Coordinate monotonicity | Proved derivative signs on finite product intervals | Correct endpoint selection, continuity/domain checks and separate attainment analysis |
| Multi-affine boxes | Numerator and denominator affine in every coordinate | A denominator with fixed strict sign throughout the closed box; all closure vertices evaluated |
| Weighted medians | Positive weighted absolute values of affine scalar forms | Exact ordering and cumulative weights; ties can yield an optimizer interval |
| Linear epigraphs | Positive affine Abs/Max terms with affine constraints | Equivalent epigraph inequalities and a certified reduced LP result |
| Exact quadratic algebra | Unconstrained small PSD quadratics | Exact stationary equations and feasibility; singular forms can return a representative optimizer |
| Spectral rules | Small symmetric Rayleigh quotients with positive-definite metrics, or homogenous quadratics on centered ellipsoid surfaces | Smallest generalized eigenvalue and a checked real eigenvector witness |
| Linear ellipsoids | Affine objective over a positive-definite ellipsoid | Exact support-function bound and a feasible point or an explicit interior approach path |
| Classical inequalities | Positive two-power AM-GM families; reciprocal allocation under a positive total-mass equality | Positivity assumptions and an attaining equality-case witness |
| Convexity certificates | A bounded grammar of affine, power, exponential, logarithmic, absolute-value and maximum compositions | Exactly feasible differentiable stationary/KKT point with nonnegative inequality multipliers |

These are sufficient recognizers. Failure to recognize a form is not evidence of nonconvexity, unboundedness or infeasibility. A visible lower bound without a witness is not reported as an attained optimum. General nonlinear optimizer loci are represented only by the points or locus metadata that the selected rule actually obtains.

## Polynomial structure

```python
import sympy as sp
from symbopt import minimize_result, argmin

x, y = sp.symbols("x y", real=True)
visible = minimize_result(5 + (x + y - 1) ** 2 + (x - y) ** 4, variables=(x, y))
assert visible.certified and visible.optimum_value == 5
assert visible.best_candidate.primal_values == {
    x: sp.Rational(1, 2),
    y: sp.Rational(1, 2),
}

power = minimize_result(x**20 - 2 * x**10 + 3, variables=x)
assert power.certified and power.optimum_value == 2
assert argmin(x**20 - 2 * x**10 + 3, variables=x) == ({x: -1}, {x: 1})

radial = minimize_result((x * x + y * y - 3) ** 2 + 2, variables=(x, y))
assert radial.certified and radial.optimum_value == 2
point = radial.best_candidate.primal_values
assert sp.simplify(point[x] ** 2 + point[y] ** 2 - 3) == 0
```

The power rule reduces the polynomial degree before root isolation. The radial rule optimizes over a scalar squared radius. The constraint on the new scalar is explicit, so a reduced rational objective cannot acquire spurious negative-radius poles.

## Boxes and piecewise linear objectives

```python
box = [x >= -1, x <= 2, y >= 1, y <= 3]
corner = minimize_result(x * y + x - 2 * y, box, variables=(x, y))
assert corner.certified and corner.optimum_value == -10
assert corner.best_candidate.primal_values == {x: -1, y: 3}

median = minimize_result(abs(x - 1) + abs(x - 3), variables=x)
assert median.certified and median.optimum_value == 2
assert median.structural_result.metadata["median_interval"] == (1, 3)

epigraph = minimize_result(sp.Max(x + y, 2 - x - y), [x >= 0, y >= 0], variables=(x, y))
assert epigraph.certified and epigraph.optimum_value == 1
```

Coupled constraints disable a box rule unless an independent reduction proves the needed image. Denominator exclusions remain active after cancellation. Open boxes can have excluded minimizing vertices: a feasible attaining point is checked separately, and unresolved attainment remains unknown. Vertex enumeration is limited to eight variables.

## Matrix and parameter formulas

```python
spectral = minimize_result(
    (2 * x * x + 2 * x * y + 2 * y * y) / (x * x + y * y), variables=(x, y)
)
assert spectral.certified and spectral.optimum_value == 1

radius = sp.Symbol("radius", positive=True)
ellipsoid = minimize_result(3 * x + 4 * y, x * x + y * y <= radius**2, variables=(x, y))
assert ellipsoid.certified and ellipsoid.optimum_value == -5 * radius

a, b = sp.symbols("a b", positive=True)
amgm = minimize_result(a * x + b / x, x > 0, variables=x)
assert amgm.certified and sp.simplify(amgm.optimum_value - 2 * sp.sqrt(a * b)) == 0

allocation = minimize_result(
    1 / x + 4 / y, [x > 0, y > 0, sp.Eq(x + y, 3)], variables=(x, y)
)
assert allocation.certified and allocation.optimum_value == 3
assert allocation.best_candidate.primal_values == {x: 1, y: 2}

convex = minimize_result(sp.exp(x) + sp.exp(-x), variables=x)
assert convex.certified and convex.optimum_value == 2
```

The direct unconstrained quadratic route uses exact linear algebra before numerical proposals. For degenerate constrained LP/QP candidates, a sufficient convex KKT check can recover a proof even when a particular active-set reconstruction leaves multiplier freedom. Rational multiplier feasibility may use the exact simplex backend; every returned multiplier and residual is still checked independently.

## Budgets and fallback controls

The preprocessor limits expression operations, estimated expansion size, matrix dimensions, multiplier count and vertex enumeration. It makes at most one selected coordinate reduction per invocation and allows three nested reductions. Explicit LP/QP recognition retains priority for ordinary closed affine quadratic problems. No numerical sign guess discharges a proof condition, and inexact coefficients remain with the precision-aware routes.

```python
ordinary = minimize_result(
    x**6 - 2 * x * x + 3, variables=x, options={"structural": {"enabled": False}}
)
assert ordinary.certified
```

This option disables the early recognizers. Established LP/QP, univariate, semialgebraic and interval methods remain available. A structural proof found before complexity planning can leave both `kkt_system` and `solver_plan` absent; the structural proof artifact records the route and its domain conditions. Convexity certification of a subsequently found candidate remains a proof check in the ordinary pipeline.

General function decomposition, arbitrary parameter-dependent ranks, non-PD Rayleigh metrics and exhaustive searches for escaping curves are outside these recognizers. See {doc}`polynomial-rational` for the exact univariate and graph-projection fallbacks.

Recognition matches and refusals use a bounded cache keyed by the exact objective, constraints, decision variables and reduction depth. Returned proof metadata and points are independent copies. Inner solver results are not cached by this layer. `clear_symbolic_caches()` clears recognition and internal coordinate caches as well as the shared polynomial caches.

The sufficient KKT checker requires differentiability at the candidate. It declines absolute-value kinks and tied maxima; symbolic default derivatives at those points are not certified subgradients. Affine Abs/Max objectives use the epigraph or weighted-median rules.
