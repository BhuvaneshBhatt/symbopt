# High-level value, optimizer, and parameter semantics

This example collects the public convenience API in one place.

```python
import sympy as sp
from symbopt import argmin, argmax, maximize, maxval, minimize, minval

x = sp.symbols("x", real=True)
```

## Full results

```python
rmin = minimize((x - 2)**2, variables=[x])
rmax = maximize(-(x - 1)**2, variables=[x])

rmin.optimum_value
rmax.optimum_value
```

## Value-only and optimizer-only queries

```python
minval((x - 2)**2, variables=[x])
argmin((x - 2)**2, variables=[x])
maxval(-(x - 1)**2, variables=[x])
argmax(-(x - 1)**2, variables=[x])
```

## Unattained extrema

```python
minval(x, [x > 0], variables=[x])
# 0

argmin(x, [x > 0], variables=[x])
# ()
```

The global value is an infimum, but there is no minimizing point.

## Parameters are not decision variables

```python
a = sp.symbols("a", positive=True)
result = minimize(a * (x - 2)**2, variables=[x])
```

`x` is optimized while `a` remains symbolic. This distinction is important for
parametric expressions and prevents accidental optimization over every free
symbol.
