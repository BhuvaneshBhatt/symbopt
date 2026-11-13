import symbopt

PUBLIC = [
    "minimize",
    "minimize_local",
    "maximize",
    "maximize_local",
    "minval",
    "maxval",
    "argmin",
    "argmax",
    "infer_box_bounds",
    "solve_kkt_candidates",
    "solve_exact_candidates",
    "algebraize",
    "algebraize_cases",
    "critical_points",
    "solve_lp",
    "solve_convex_qp",
    "interval_optimize",
    "sos_decompose",
    "sos_lower_bound",
    "sos_scalar_relaxation",
    "sos_poly_relaxation",
    "search_sos_hierarchy",
    "solve_sdp",
    "solve_moment_dual",
    "link_sos_moments",
    "build_sparsity_graph",
    "build_symmetry_report",
]


def test_public_api_symbols_exist():
    for name in PUBLIC:
        assert hasattr(symbopt, name), name


def test_project_layout_imports():
    assert symbopt.minimize.__name__ == "minimize"


def test_root_export_ownership():
    from symbopt.api import __all__ as public_api

    assert symbopt.__all__ == public_api
    assert "solver_pipeline" not in public_api
    assert "fro_inner" not in public_api
    assert "reduce_relations" not in public_api
    assert "PipelineResult" in public_api
