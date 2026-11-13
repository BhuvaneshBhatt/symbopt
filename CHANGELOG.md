## 0.2.0

Version 0.2.0 expands polynomial optimization capabilities, improves integration with semialgebraic reasoning, and introduces stricter validation of mathematical assumptions and optimization results.

### Method and Certification Improvements

- Expand sum-of-squares (SOS) and semidefinite programming (SDP) methods, including improved relaxation planning, cost estimation, and certificate search.
- Improve exact refinement of optimization candidates and verification of polynomial nonnegativity certificates.
- Strengthen Karush–Kuhn–Tucker (KKT) analysis, including constraint handling, multiplier calculations, and classification of stationary points.
- Extend exact polynomial optimisation and infeasibility reasoning using algebraic certificates and Positivstellensatz-based methods.

### Solver Architecture and Performance

- Refactor optimization pipeline into more clearly separated solver orchestration, execution, and result-assembly components.
- Introduce structured solver-option validation and more consistent propagation of configuration settings through nested optimisation problems.
- Improve cost-aware selection of exact algebraic, numerical, interval, and SOS-based methods.
- Refine structural preprocessing and optimization routes for separable, piecewise, and supported mixed or periodic problems.
- Improve SOS relaxation cost estimates to account for polynomial and constraint structure.

### Correctness and Reliability

- Strengthened validation of decision variables, constraints, variable bounds, and symbolic parameters.
- Fix cases where strict inequalities or disequality constraints could be incorrectly handled or omitted during KKT analysis.
- Correct maximization transformations involving Lagrangians, multipliers, Hessians, eigenvalues, and local extremum classifications.
- Improve handling of unsupported solver inputs, preventing inappropriate numerical methods from being applied to symbolic problems.

### Testing and Validation

- Expand regression tests covering solver options, KKT constraints, maximization transformations, symbolic parameters, and certification semantics.
- Add tests for previously problematic mathematical edge cases and unsupported inputs.
- Expand testing of optimization routes, structural reductions, and solver result consistency.
- Replace inherited reference-case identifiers and source-file metadata with descriptive pytest IDs and repository-owned corpus checksums.
- Preserve all 779 mathematical reference problems, expected values, feasibility checks and capability classifications while simplifying test organization.
