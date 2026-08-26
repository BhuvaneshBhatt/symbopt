# Linear and quadratic optimization API

## Linear programming

```{autofunction} symbopt.solve_lp
```

The advanced standard-form simplex routines are also exported from the package:

```{autofunction} symbopt.simplex
```

```{autofunction} symbopt.revised_simplex
```

## Convex quadratic programming

```{autofunction} symbopt.solve_convex_qp
```

The convex-QP routine uses free variables by default and rejects a
non-positive-semidefinite quadratic form.

See {doc}`../methods/linear-programming` and
{doc}`../methods/quadratic-programming`.
