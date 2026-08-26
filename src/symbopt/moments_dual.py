from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import sympy as sp
from .moment import (
    MomentRelaxation,
    build_moment_relaxation,
    extract_moment_atoms,
    MomentMatrix,
)


@dataclass
class DualMomentResult:
    relaxation: MomentRelaxation
    flat_extension_detected: bool
    atom_extraction: Optional[object]
    candidate_points: List[Dict[sp.Symbol, sp.Expr]]
    verified: bool = False
    message: str = ""

    def pretty(self) -> str:
        return "\n".join(
            [
                f"flat_extension_detected: {self.flat_extension_detected}",
                f"candidate_points: {self.candidate_points}",
                f"verified: {self.verified}",
                f"message: {self.message}",
            ]
        )


def solve_moment_dual(
    objective: sp.Expr,
    inequalities: Sequence[sp.Expr],
    vars: Sequence[sp.Symbol],
    order: int,
    moment_values: Optional[Dict[Tuple[int, ...], sp.Expr]] = None,
) -> DualMomentResult:
    """Solve the implemented dual moment relaxation and return moment/certificate diagnostics."""
    rel = build_moment_relaxation(objective, inequalities, vars, order)
    if moment_values is None:
        return DualMomentResult(
            rel,
            False,
            None,
            [],
            False,
            "Compiled dual moment relaxation only; no moments supplied.",
        )
    mm = (
        MomentMatrix(
            basis=rel.moment_matrix.basis,
            matrix=sp.Matrix(
                [
                    [
                        moment_values[tuple(int(x) for x in sym.name.split("_")[1:])]
                        if sym.free_symbols == {sym}
                        else sym
                        for sym in row
                    ]
                    for row in rel.moment_matrix.matrix.tolist()
                ]
            ),
        )
        if rel.moment_matrix.matrix.free_symbols
        else rel.moment_matrix
    )
    atoms = extract_moment_atoms(mm.matrix, mm.basis, vars)
    return DualMomentResult(
        rel,
        atoms.flat,
        atoms,
        atoms.atoms,
        atoms.verified,
        "Compiled dual moment relaxation and ran flatness / atom extraction.",
    )
