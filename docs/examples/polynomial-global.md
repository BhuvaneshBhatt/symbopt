# Polynomial global optimization

Consider

\[
f(x)=(x^2-1)^2.
\]

```python
import sympy as sp
from symbopt import minimize_result

x = sp.symbols("x", real=True)
result = minimize_result((x**2 - 1) ** 2)
```

Candidate generation finds the stationary structure. Polynomial global methods can establish the lower bound

\[
f(x)\ge0,
\]

and feasible points \(x=\pm1\) attain that bound.

Inspect:

```python
result.sos_result
result.sos_lower_bound
result.certified
result.certificate
```

If exact SOS recovery succeeds, the lower bound is a symbolic certificate rather than just an approximate SDP observation.
