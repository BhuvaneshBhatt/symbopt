# Moment methods

Moment relaxations are dual to SOS-style polynomial positivity relaxations.

## Moment matrix

For a moment sequence \(y_\alpha\), the moment matrix contains entries

\[
M_d(y)_{\alpha,\beta}=y_{\alpha+\beta}.
\]

`symbopt.build_moment_matrix()` constructs this matrix using exponent tuples.

## Localizing matrices

For a polynomial constraint \(g(x)\ge0\), the localizing matrix incorporates the
coefficients of \(g\) into shifted moments.

## Flatness and atoms

When moment matrices satisfy an appropriate flat-rank pattern, minimizers can
sometimes be recovered as atoms of a finite measure. The package builds
multiplication matrices and extracts candidate atoms/weights.

## Example

```python
import sympy as sp
from symbopt import build_moment_matrix

x = sp.symbols("x", real=True)
matrix = build_moment_matrix([x], order=2)
print(matrix.matrix)
```

The higher-level moment bridge can compare candidate points with moment data and
support the polynomial global-optimization pipeline.

## Further reading

See Lasserre and Laurent in {doc}`../reference/references` for the mathematical background.

