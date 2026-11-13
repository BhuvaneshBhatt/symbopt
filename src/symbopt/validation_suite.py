from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import sympy as sp

from .chordal import suggest_psd_blocks
from .lp import solve_lp
from .moment import build_moment_matrix, extract_moment_atoms
from .moments_dual import solve_moment_dual
from .qp import solve_convex_qp
from .sdp import SDPProblem, solve_sdp_dense
from .sos import sos_decompose, sos_lower_bound, sos_scalar_relaxation


def run_validation_suite(verbose: bool = False):
    """Run the independent numerical validation battery and return structured results."""
    out = {}

    C = np.eye(2)
    A = [np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [0.0, 1.0]])]
    b = np.array([1.0, 2.0])
    res = solve_sdp_dense(
        SDPProblem.single_block(C=C, A=A, b=b), max_iter=80, tol=1e-8, verbose=verbose
    )
    out["diag_trace"] = {
        "kind": "sdp",
        "passed": abs(res.primal_objective - 3.0) <= 2e-1,
        "status": res.status,
        "primal_objective": res.primal_objective,
    }

    M = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    dec = suggest_psd_blocks(M)
    out["chordal_structure"] = {
        "kind": "chordal",
        "passed": len(dec.cliques) >= 1,
        "num_cliques": len(dec.cliques),
    }

    x, y = sp.symbols("x y", real=True)
    decsos = sos_decompose((x**2 + y**2 + 1) ** 2 + (x - y) ** 2, [x, y])
    out["sos_decompose"] = {
        "kind": "sos",
        "passed": bool(decsos.success and sp.expand(decsos.exact_residual) == 0),
        "success": decsos.success,
    }
    lb, dec2 = sos_lower_bound(
        sp.expand((x - 1) ** 2 + (y + 2) ** 2 + 3),
        [x, y],
        search_lo=-10,
        search_hi=10,
        max_iter=10,
    )
    out["sos_lower_bound"] = {
        "kind": "sos_lb",
        "passed": bool(dec2.success and abs(float(sp.N(lb, 30)) - 3.0) <= 1e-8),
        "lower_bound": float(sp.N(lb, 30)),
    }
    relax = sos_scalar_relaxation(
        x**2 + y**2,
        [x, y],
        [x, y],
        search_lo=-1,
        search_hi=2,
        max_iter=10,
        multiplier_degree=1,
    )
    out["poly_multiplier_relax"] = {
        "kind": "sos_relax",
        "passed": bool(relax.success),
        "lower_bound": relax.lower_bound_numeric,
    }

    x1 = sp.symbols("x1", real=True)
    mm1 = build_moment_matrix(
        [x1], 1, {(0,): sp.Integer(1), (1,): sp.Integer(2), (2,): sp.Integer(4)}
    )
    atoms1 = extract_moment_atoms(mm1.matrix, mm1.basis, [x1])
    out["moment_atom_extraction_rank1"] = {
        "kind": "moment",
        "passed": atoms1.flat and len(atoms1.atoms) == 1,
        "atoms": str(atoms1.atoms),
    }
    mm2 = build_moment_matrix(
        [x1],
        2,
        {
            (0,): sp.Integer(1),
            (1,): sp.Integer(1),
            (2,): sp.Integer(2),
            (3,): sp.Integer(4),
            (4,): sp.Integer(8),
        },
    )
    atoms2 = extract_moment_atoms(mm2.matrix, mm2.basis, [x1])
    out["moment_atom_extraction_multi"] = {
        "kind": "moment",
        "passed": atoms2.flat and len(atoms2.atoms) >= 1,
        "atoms": str(atoms2.atoms),
    }
    dmp = solve_moment_dual(
        x1**2,
        [],
        [x1],
        1,
        moment_values={(0,): sp.Integer(1), (1,): sp.Integer(2), (2,): sp.Integer(4)},
    )
    out["solve_moment_dual"] = {
        "kind": "moments_dual",
        "passed": dmp.flat_extension_detected and len(dmp.candidate_points) == 1,
        "candidate_points": str(dmp.candidate_points),
    }

    lpres = solve_lp(
        np.array([1.0, 2.0]),
        A_ub=np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        b_ub=np.array([4.0, 3.0, 3.0]),
        maximize=False,
    )
    out["lp_optimization"] = {
        "kind": "lp",
        "passed": lpres.status == "optimal",
        "objective": lpres.objective,
    }

    qpres = solve_convex_qp(
        np.array([[2.0, 0.0], [0.0, 2.0]]),
        np.array([-2.0, -4.0]),
        A_ub=np.array([[1.0, 1.0]]),
        b_ub=np.array([3.0]),
    )
    out["qp_optimization"] = {
        "kind": "qp",
        "passed": qpres.status == "optimal",
        "objective": qpres.objective,
    }
    return out


def write_validation_summary(
    path: str | Path, results: dict[str, dict[str, object]]
) -> Path:
    """Write the numerical validation results as a JSON report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in results.values() for k in row} | {"name"})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, row in results.items():
            rec = {"name": name}
            rec.update(row)
            writer.writerow(rec)
    return path
