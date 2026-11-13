# Optimization in model fitting, allocation and design

These examples show how a mathematical model reaches a value, point or bound. Every Python block on this page belongs to one executable session. Numerical proposals can be useful during exploration, but a global claim needs the corresponding exact or verified result.

## Calibrate a linear sensor

Suppose a sensor response is modeled as `a*t + b`. Three exact observations admit a least-squares calibration. A quadratic objective can return both coefficients exactly.

```python
import sympy as sp
import symbopt as so

a, b, x, y = sp.symbols("a b x y", real=True)
data = [(0, 1), (1, 3), (2, 5)]
loss = sum((a * t + b - observed) ** 2 for t, observed in data)
calibration = so.minimize_result(loss, variables=(a, b))
assert calibration.certified and calibration.optimum_value == 0
assert calibration.best_candidate.original_variable_values == {a: 2, b: 1}
assert loss.subs({a: 2, b: 1}) == 0
```

This verifies the fitted model for the stated data. It does not quantify measurement uncertainty; noisy data call for a statistical model and uncertainty analysis in addition to optimization.

## Allocate two production resources

Two products earn 3 and 2 units per item. Total capacity is four items and the first product has capacity two. Linear constraints preserve the physical feasible region.

```python
allocation = so.maximize_result(
    3 * x + 2 * y, [x >= 0, y >= 0, x + y <= 4, x <= 2], variables=(x, y)
)
assert allocation.certified and allocation.optimum_value == 10
assert allocation.best_candidate.original_variable_values == {x: 2, y: 2}
```

This is continuous allocation. Integer batch sizes require an integer model; the general continuous pipeline rejects integer decision symbols.

## Bound a nonconvex design cost

A bounded design with cost `x**2 - y**2` has an indefinite Hessian. Native stationary-face enumeration checks the bounded rational box and supplies exact replayable evidence.

```python
design = so.minimize_quadratic(
    x * x - y * y,
    variables=(x, y),
    bounds={x: (-1, 1), y: (-1, 1)},
    backend="faces",
    certified=True,
)
assert design.certified and design.objective_value == -1
assert so.verify_quadratic_certificate(
    x * x - y * y, design.certificate, [x >= -1, x <= 1, y >= -1, y <= 1], (x, y)
)
```

A positive gap or an exhausted face/node budget would retain bounds without claiming an exact optimum. Curved quadratic constraints use the spatial route within its finite-domain assumptions.

## Interpret an open operating domain

If a response is modeled by `1/x` for positive `x`, no finite operating point attains its infimum. A value-only answer and a minimizing point answer therefore differ.

```python
response = so.minimize_result(1 / x, x > 0, variables=x)
assert response.certified and response.optimum_value == 0
assert response.attained is False
assert so.argmin(1 / x, x > 0, variables=x) == ()
```

## Replay a polynomial inequality

An SOS Gram identity proves a polynomial lower bound. Replay binds the source polynomial and bound, and checks matrix positivity independently of stored summary flags.

```python
from dataclasses import replace

squares = so.sos_decompose((x - y) ** 2, (x, y))
proof = squares.certificate
assert proof.verify(polynomial=(x - y) ** 2, lower_bound=0)
assert not replace(proof, lower_bound=1).verified
```

Use {doc}`../methods/sum-of-squares` for constrained multiplier identities and {doc}`../guide/certification` for the distinction between discovery and verification.
