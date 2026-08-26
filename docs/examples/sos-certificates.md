# SOS certificates

A direct SOS decomposition example is

\[
p(x)=(x^2-1)^2.
\]

```python
import sympy as sp
from symbopt import sos_decompose

x = sp.symbols("x", real=True)
result = sos_decompose((x**2 - 1)**2, [x])

print(result.success)
print(result.certificate.exact_residual_zero)
print(result.certificate.exact_psd_checks)
```

The important distinction is between the numerical SDP solution and the
recovered exact certificate. Rigorous success requires the polynomial residual
to vanish exactly and all Gram matrices to pass exact PSD verification.
