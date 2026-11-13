# Advanced optimization and certification API

The following public objects expose local optimization, exact refinement, solver planning, external SDP proposals, SOS search, and Positivstellensatz certification:

- `symbopt.LocalModeResult`
- `symbopt.ConstrainedLocalResult`
- `symbopt.ExactRefinementResult`
- `symbopt.minimize_result`
- `symbopt.maximize_result`
- `symbopt.minimize_local`
- `symbopt.maximize_local`
- `symbopt.minimize_local_constrained`
- `symbopt.maximize_local_constrained`
- `symbopt.refine_exact_system`
- `symbopt.project_affine_exact`
- `symbopt.reconstruct_exact_vector`
- `symbopt.ProblemComplexity`
- `symbopt.SolverPlan`
- `symbopt.analyze_problem`
- `symbopt.plan_problem`
- `symbopt.CriticalConeResult`
- `symbopt.analyze_critical_cone`
- `symbopt.available_external_sdp_solvers`
- `symbopt.cvxpy_available`
- `symbopt.solve_sdp_cvxpy`
- `symbopt.SOSCertificateSearchResult`
- `symbopt.SOSCostEstimate`
- `symbopt.SOSSearchPlan`
- `symbopt.estimate_sos_cost`
- `symbopt.plan_sos_search`
- `symbopt.search_sos_certificate`
- `symbopt.PreorderingGenerator`
- `symbopt.PositivstellensatzCertificate`
- `symbopt.PositivstellensatzResult`
- `symbopt.preordering_generators`
- `symbopt.search_infeasibility_certificate`
- `symbopt.verify_infeasibility_certificate`

Numerical SDP results, including results produced through the optional CVXPY adapter, are proposal data. Exact SOS and Positivstellensatz status is granted only after symbolic identity and PSD verification.
