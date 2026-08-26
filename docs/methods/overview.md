# Optimization methods

`symbopt` combines several solver families because no single symbolic method
works well for every continuous optimization problem.

The main families are:

- symbolic critical-point and degeneracy analysis,
- KKT and Karush--John constrained stationary systems,
- algebraization and elimination,
- linear programming,
- convex quadratic programming,
- exact structure-aware routes for linear-fractional, trigonometric, transcendental, and separable problems,
- semidefinite programming,
- sum-of-squares polynomial optimization,
- moment relaxations and atom extraction,
- sparsity/symmetry exploitation,
- verified interval branch-and-bound.

Each method page describes the recognized problem class, mathematical
convention, return semantics, limitations, and a worked example.
