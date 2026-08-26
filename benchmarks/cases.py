from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import sympy as sp

from symbopt import build_moment_matrix, interval_optimize, minimize
from symbopt.sos_program import compile_sos_relaxation
from symbopt.sdp import SDPProblem, solve_sdp_dense


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

    return [
        BenchmarkCase(
            "qp_dispatch",
            "pipeline",
            lambda: minimize((x - 3) ** 2 + (y + 2) ** 2),
            "High-level recognition and solution of a small convex QP.",
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
