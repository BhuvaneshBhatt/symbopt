from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import sympy as sp

from symbopt import (
    build_moment_matrix,
    interval_optimize,
    minimize,
    minimize_quadratic,
    minimize_result,
)
from symbopt.constraint_reduction import reduce_relations
from symbopt.convex_routing import _postprocess_direct
from symbopt.intervals import _precondition_forms
from symbopt.precision import outward_float
from symbopt.quadratic_candidates import sampled_points
from symbopt.quadratic_models import quadratic_problem
from symbopt.sdp import SDPProblem, solve_sdp_dense
from symbopt.sos_program import compile_sos_relaxation


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    group: str
    run: Callable[[], object]
    description: str


def benchmark_cases() -> list[BenchmarkCase]:
    x, y, z = sp.symbols("x y z", real=True)
    quartic = (x**2 + y**2 + z**2 + 1) ** 2 + (x - y) ** 2

    sdp_problem = SDPProblem.single_block(
        C=np.eye(3),
        A=[
            np.diag([1.0, 0.0, 0.0]),
            np.diag([0.0, 1.0, 0.0]),
            np.diag([0.0, 0.0, 1.0]),
        ],
        b=np.array([1.0, 2.0, 3.0]),
    )

    repeated_relations = [
        sp.Ge((x - k) ** 2 + y * y, 0, evaluate=False) for k in range(4)
    ] * 8
    sliced_problem = quadratic_problem(
        x + y, [sp.Eq(x * x + y * y, 1), sp.Eq(y, x * x)], (x, y)
    )
    precondition = (x * x - 1) / (x - 1) + sp.sin(x) ** 2 + sp.cos(x) ** 2

    rational_values = [sp.Rational(i, 4097) for i in range(1, 65)]
    return [
        BenchmarkCase(
            "rational_outward_bounds",
            "preprocessing",
            lambda: tuple(
                outward_float(value, side)
                for value in rational_values
                for side in (-1, 1)
            ),
            "Directed rational-to-binary endpoint conversion using integer comparisons.",
        ),
        BenchmarkCase(
            "constraint_reduction",
            "preprocessing",
            lambda: reduce_relations(repeated_relations),
            "Simplification and structural deduplication of repeated constraints.",
        ),
        BenchmarkCase(
            "quadratic_equality_slices",
            "preprocessing",
            lambda: tuple(sampled_points(sliced_problem, ((-2, 2), (-2, 2)))),
            "Exact linear and quadratic coordinate recovery for candidate proposals.",
        ),
        BenchmarkCase(
            "interval_preconditioning",
            "preprocessing",
            lambda: _precondition_forms.__wrapped__(precondition),
            "Uncached equivalent-form compilation with source-domain ownership.",
        ),
        BenchmarkCase(
            "active_kkt_recovery",
            "preprocessing",
            lambda: _postprocess_direct(
                "qp",
                (x - 2) ** 2 + (y - 3) ** 2,
                [x >= 0, y >= 0, x + y <= 5, x >= 2],
                (x, y),
                {x: 2.0, y: 3.0},
                0.0,
            ),
            "Active-constraint preparation and exact primal/dual reconstruction.",
        ),
        BenchmarkCase(
            "nonconvex_faces",
            "quadratic",
            lambda: minimize_quadratic(
                x * x - y * y,
                [x >= 0, y >= 0, x + y <= 1],
                (x, y),
                backend="faces",
                certified=True,
            ),
            "Complete exact stationary-face search with certificate replay.",
        ),
        BenchmarkCase(
            "nonconvex_spatial",
            "quadratic",
            lambda: minimize_quadratic(
                -x * y,
                [x >= 0, y >= 0, x + y <= 1],
                (x, y),
                backend="spatial",
                max_nodes=8,
                certified=True,
            ),
            "Exact product relaxation, spatial partition and proof replay.",
        ),
        BenchmarkCase(
            "qp_dispatch",
            "pipeline",
            lambda: minimize((x - 3) ** 2 + (y + 2) ** 2),
            "High-level recognition and solution of a small convex QP.",
        ),
        BenchmarkCase(
            "power_substitution",
            "structural",
            lambda: minimize_result(x**20 - 2 * x**10 + 3, variables=x, use_sos=False),
            "Degree reduction with both real inverse branches.",
        ),
        BenchmarkCase(
            "weighted_median",
            "structural",
            lambda: minimize_result(
                abs(x - 1) + 2 * abs(x - 2) + abs(x - 5), variables=x, use_sos=False
            ),
            "Exact weighted median without piecewise branch enumeration.",
        ),
        BenchmarkCase(
            "generalized_rayleigh",
            "structural",
            lambda: minimize_result(
                (3 * x * x + 2 * x * y + 3 * y * y) / (x * x + y * y),
                variables=(x, y),
                use_sos=False,
            ),
            "Small exact spectral bound with a real eigenvector witness.",
        ),
        BenchmarkCase(
            "unmatched_convex",
            "fallback",
            lambda: minimize_result(sp.exp(x) - 2 * x, variables=x, use_sos=False),
            "Recognizer refusal followed by the scalar transcendental route.",
        ),
        BenchmarkCase(
            "sos_compile_quartic",
            "sos",
            lambda: compile_sos_relaxation(quartic, [x, y, z], 2),
            "Direct coefficient assembly for a three-variable quartic SOS SDP.",
        ),
        BenchmarkCase(
            "moment_matrix_order3",
            "moments",
            lambda: build_moment_matrix([x, y, z], 3),
            "Exponent-based construction of a three-variable order-3 moment matrix.",
        ),
        BenchmarkCase(
            "interval_polynomial",
            "interval",
            lambda: interval_optimize(
                (x - 0.3) ** 2 + (y + 0.2) ** 2,
                [],
                {x: (-1.0, 1.0), y: (-1.0, 1.0)},
                max_boxes=40,
                tol=1e-3,
            ),
            "Verified polynomial branch-and-bound on a two-dimensional box.",
        ),
        BenchmarkCase(
            "sdp_small_dense",
            "sdp",
            lambda: solve_sdp_dense(sdp_problem, max_iter=80, tol=1e-8),
            "Small dense SDP with reusable operator/factorization workspace.",
        ),
    ]
