# Moment extraction

Moment matrices can encode finitely supported minimizer measures.

```python
import sympy as sp
from symbopt import build_moment_matrix, extract_moment_atoms

x = sp.symbols("x", real=True)

moments = {
    (0,): sp.Integer(1),
    (1,): sp.Integer(2),
    (2,): sp.Integer(4),
}
matrix = build_moment_matrix(
    [x],
    order=1,
    moment_values=moments,
)

atoms = extract_moment_atoms(
    matrix.matrix,
    matrix.basis,
    [x],
)
```

These moments describe a rank-one Dirac measure at \(x=2\). When flatness and
multiplication-matrix conditions are suitable, the atom extractor recovers the
support point and weights and then verifies moment residuals.
