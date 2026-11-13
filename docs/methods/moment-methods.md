# Moment methods

Moment relaxations are dual to SOS-style polynomial positivity relaxations.

## Moment matrix

For a moment sequence \(y_\alpha\), the moment matrix contains entries

\[
M_d(y)_{\alpha,\beta}=y_{\alpha+\beta}.
\]

`symbopt.build_moment_matrix()` constructs this matrix using exponent tuples.

## Localizing matrices

For a polynomial constraint \(g(x)\ge0\), the localizing matrix incorporates the coefficients of \(g\) into shifted moments.

## Flatness and atoms

When moment matrices satisfy an appropriate flat-rank pattern, minimizers can sometimes be recovered as atoms of a finite measure. The package builds multiplication matrices and extracts candidate atoms/weights.

## Example

```python
import sympy as sp
from symbopt import build_moment_matrix

x = sp.symbols("x", real=True)
matrix = build_moment_matrix([x], order=2)
print(matrix.matrix)
```

The higher-level moment bridge can compare candidate points with moment data and support the polynomial global-optimization pipeline.

## Further reading

See Lasserre and Laurent in {doc}`../reference/references` for the mathematical background.

## Verification boundaries

For an order-d matrix, the previous-order submatrix contains exactly the basis monomials of degree at most d-1. Removing a fixed number of final rows gives the wrong submatrix in several variables. The reported flatness flag uses numerical ranks and remains a candidate-generation condition.

An extracted measure is marked verified only when its weights are nonnegative, its coordinates are real, and it reproduces every entry of the complete symmetric moment matrix exactly. Numerical atom and weight reconstruction supplies proposals; verification compares the original stored moments without rounding them to nearby rational values. This verifies the moment representation, not feasibility or global optimality for an unrelated optimization problem.

## Multivariate atom recovery

The extractor selects independent monomials from the previous-order moment matrix using pivoted QR. Their Gram matrix defines the inner product for joint diagonalization. A separating combination of coordinate operators resolves atoms even when one coordinate has repeated values. The supplied monomial basis may be reordered or rescaled by nonzero concrete real coefficients.

```python
from symbopt import extract_moment_atoms

x, y = sp.symbols("x y", real=True)
points = [(0, 0), (0, 1), (1, 0), (1, 1)]
values = {
    (i, j): sum(sp.Rational(1, 4) * a**i * b**j for a, b in points)
    for i in range(7)
    for j in range(7 - i)
}
data = build_moment_matrix((x, y), 3, values)
atoms = extract_moment_atoms(data.matrix, data.basis, (x, y))
assert atoms.flat and atoms.verified
assert {(atom[x], atom[y]) for atom in atoms.atoms} == set(points)
assert atoms.weights == [sp.Rational(1, 4)] * 4
```

`max_separating_attempts` bounds the deterministic search and defaults to 16. Exhaustion leaves the result unverified. Rank decisions and atom reconstruction are numerical proposals, so ill-conditioned data or atoms too close for the selected tolerance can remain unresolved. Only exact reconstruction of a positive real measure against the original full matrix sets `verified=True`; flatness alone does not establish that representation or certify an optimization problem.
