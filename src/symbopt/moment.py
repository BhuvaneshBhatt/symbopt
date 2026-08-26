from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import math

import numpy as np
import sympy as sp

from .monomials import (
    Exponent,
    add_exponents,
    add_three,
    basis_data,
    coefficient_map,
    moment_symbol,
    monomial_basis as cached_basis,
)


@dataclass
class MomentMatrix:
    basis: List[sp.Expr]
    matrix: sp.Matrix


@dataclass
class LocalizingMatrix:
    polynomial: sp.Expr
    basis: List[sp.Expr]
    matrix: sp.Matrix


@dataclass
class MomentRelaxation:
    order: int
    variables: List[sp.Symbol]
    basis: List[sp.Expr]
    moment_matrix: MomentMatrix
    localizing_matrices: List[LocalizingMatrix]
    objective_linear_form: sp.Expr


@dataclass
class AtomExtractionResult:
    flat: bool
    rank_m: int
    rank_m_minus_1: Optional[int]
    atoms: List[Dict[sp.Symbol, sp.Expr]]
    weights: List[sp.Expr]
    message: str
    multiplication_matrices: Dict[str, sp.Matrix] | None = None
    verified: bool = False
    residuals: Dict[str, sp.Expr] | None = None

    def pretty(self) -> str:
        return "\n".join(
            [
                f"flat: {self.flat}",
                f"rank_m: {self.rank_m}",
                f"rank_m_minus_1: {self.rank_m_minus_1}",
                f"atoms: {self.atoms}",
                f"weights: {self.weights}",
                f"verified: {self.verified}",
                f"residuals: {self.residuals}",
                f"message: {self.message}",
            ]
        )


def monomial_basis(
    vars: Sequence[sp.Symbol],
    degree: int,
) -> List[sp.Expr]:
    return list(cached_basis(tuple(vars), degree))


def _moment_value(
    powers: Exponent,
    values: Optional[Dict[Exponent, sp.Expr]],
) -> sp.Expr:
    if values is None:
        return moment_symbol(powers)
    return values[powers]


def build_moment_matrix(
    vars: Sequence[sp.Symbol],
    order: int,
    moment_values: Optional[Dict[Exponent, sp.Expr]] = None,
) -> MomentMatrix:
    """Construct the truncated moment matrix for the requested variables and order.

    Symbolic moment entries are created when ``moment_values`` is omitted; a
    supplied exponent-to-value mapping instead produces a populated matrix.
    """
    variables = (vars,) if isinstance(vars, sp.Symbol) else tuple(vars)
    basis = tuple(cached_basis(variables, order))
    data = basis_data(basis, variables)
    powers = [entry[1] for entry in data]
    size = len(basis)

    matrix = sp.MutableDenseMatrix.zeros(size, size)
    for row, row_exp in enumerate(powers):
        for col in range(row, size):
            powers_sum = add_exponents(row_exp, powers[col])
            value = _moment_value(powers_sum, moment_values)
            matrix[row, col] = value
            if row != col:
                matrix[col, row] = value

    return MomentMatrix(list(basis), sp.Matrix(matrix))


def build_localizing_matrix(
    poly: sp.Expr,
    vars: Sequence[sp.Symbol],
    order: int,
    moment_values: Optional[Dict[Exponent, sp.Expr]] = None,
) -> LocalizingMatrix:
    """Construct the localizing matrix associated with a polynomial inequality.

    The basis order is reduced according to the polynomial degree so every
    matrix entry refers only to moments available at the requested relaxation.
    """
    variables = (vars,) if isinstance(vars, sp.Symbol) else tuple(vars)
    terms = coefficient_map(poly, variables)
    degree = sp.Poly(poly, *variables).total_degree()
    base_order = max(0, order - math.ceil(degree / 2))
    basis = tuple(cached_basis(variables, base_order))
    data = basis_data(basis, variables)
    powers = [entry[1] for entry in data]
    size = len(basis)

    matrix = sp.MutableDenseMatrix.zeros(size, size)
    for row, row_exp in enumerate(powers):
        for col in range(row, size):
            col_exp = powers[col]
            value = sp.Integer(0)
            for poly_exp, coeff in terms.items():
                moment_exp = add_three(row_exp, col_exp, poly_exp)
                value += coeff * _moment_value(moment_exp, moment_values)
            value = sp.expand(value)
            matrix[row, col] = value
            if row != col:
                matrix[col, row] = value

    return LocalizingMatrix(
        polynomial=sp.expand(poly),
        basis=list(basis),
        matrix=sp.Matrix(matrix),
    )


def build_moment_relaxation(
    objective: sp.Expr,
    inequalities: Sequence[sp.Expr],
    vars: Sequence[sp.Symbol],
    order: int,
) -> MomentRelaxation:
    """Assemble a moment relaxation from an objective and inequality residuals.

    The returned object contains the moment matrix, one localizing matrix per
    inequality, and the symbolic objective linear form for downstream solvers.
    """
    variables = (vars,) if isinstance(vars, sp.Symbol) else tuple(vars)
    if isinstance(inequalities, sp.Basic):
        inequalities = (inequalities,)
    else:
        inequalities = tuple(inequalities)
    moment_matrix = build_moment_matrix(variables, order)
    localizing = [
        build_localizing_matrix(poly, variables, order) for poly in inequalities
    ]
    return MomentRelaxation(
        order=order,
        variables=list(variables),
        basis=moment_matrix.basis,
        moment_matrix=moment_matrix,
        localizing_matrices=localizing,
        objective_linear_form=sp.expand(objective),
    )


def _numeric_rank(matrix: sp.Matrix, tol: float = 1e-8) -> int:
    array = np.array(matrix.evalf().tolist(), dtype=float)
    singular = np.linalg.svd(array, compute_uv=False)
    scale = singular[0] if len(singular) else 1.0
    return int(np.sum(singular > tol * max(1.0, scale)))


def _basis_exponents(
    basis: Sequence[sp.Expr],
    vars: Sequence[sp.Symbol],
) -> List[Exponent]:
    variables = tuple(vars)
    return [powers for _, powers in basis_data(tuple(basis), variables)]


def _solve_weights(
    atoms: List[Dict[sp.Symbol, sp.Expr]],
    basis: Sequence[sp.Expr],
    moment_matrix: sp.Matrix,
) -> List[sp.Expr]:
    if not atoms:
        return []

    row_count = max(1, len(atoms))
    rows = []
    rhs = []
    for index, term in enumerate(basis[:row_count]):
        rows.append([sp.nsimplify(sp.expand(term.subs(atom))) for atom in atoms])
        rhs.append(sp.nsimplify(moment_matrix[index, 0]))

    matrix = sp.Matrix(rows)
    target = sp.Matrix(rhs)
    try:
        solution = matrix.gauss_jordan_solve(target)[0]
        return [sp.nsimplify(value) for value in solution]
    except (TypeError, ValueError, ArithmeticError, NotImplementedError):
        weight = sp.nsimplify(moment_matrix[0, 0] / len(atoms))
        return [weight for _ in atoms]


def _dedupe_atoms(
    atoms: List[Dict[sp.Symbol, sp.Expr]],
    vars: Sequence[sp.Symbol],
) -> List[Dict[sp.Symbol, sp.Expr]]:
    unique = []
    for atom in atoms:
        duplicate = any(
            all(sp.simplify(atom[var] - other[var]) == 0 for var in vars)
            for other in unique
        )
        if not duplicate:
            unique.append(atom)
    return unique


def extract_moment_atoms(
    moment_matrix: sp.Matrix,
    basis: Sequence[sp.Expr],
    vars: Sequence[sp.Symbol],
    tol: float = 1e-8,
) -> AtomExtractionResult:
    """Recover finitely supported optimizer candidates from a flat moment matrix.

    Extraction first checks the numerical flat-rank condition, then constructs
    multiplication matrices whose joint spectral information yields atoms and
    weights when the finite-support assumptions are satisfied.
    """
    variables = list(vars)
    rank_m = _numeric_rank(moment_matrix, tol)
    rank_prev = None

    if len(basis) > 1:
        prev_size = max(1, len(basis) - len(variables))
        rank_prev = _numeric_rank(
            moment_matrix[:prev_size, :prev_size],
            tol,
        )

    flat = rank_prev is not None and rank_prev == rank_m
    atoms: List[Dict[sp.Symbol, sp.Expr]] = []
    weights: List[sp.Expr] = []
    mults: Dict[str, sp.Matrix] = {}

    if not flat:
        return AtomExtractionResult(
            False,
            rank_m,
            rank_prev,
            atoms,
            weights,
            "Flatness failed.",
            None,
            False,
            None,
        )

    numeric = np.array(moment_matrix.evalf().tolist(), dtype=float)
    rank = max(1, rank_m)
    basis_exps = _basis_exponents(basis, variables)
    exp_map = {powers: index for index, powers in enumerate(basis_exps)}
    indices = list(range(min(rank, len(basis))))
    base = numeric[np.ix_(indices, indices)]
    inverse = np.linalg.pinv(base, rcond=tol)

    for var_pos, variable in enumerate(variables):
        mult = np.zeros((len(indices), len(indices)))
        for col, basis_index in enumerate(indices):
            powers = list(basis_exps[basis_index])
            powers[var_pos] += 1
            target_exp = tuple(powers)
            target_index = exp_map.get(target_exp)
            if target_index is None:
                continue
            column = numeric[np.ix_(indices, [target_index])]
            mult[:, col] = (inverse @ column).ravel()
        mults[str(variable)] = sp.Matrix(mult)

    if len(variables) == 1 and mults:
        variable = variables[0]
        matrix = np.array(
            mults[str(variable)].evalf().tolist(),
            dtype=float,
        )
        eigenvalues = np.linalg.eigvals(matrix)
        atoms = [{variable: sp.nsimplify(value.real)} for value in eigenvalues]
        atoms = _dedupe_atoms(atoms, variables)
        weights = _solve_weights(atoms, basis, moment_matrix)
        verified, residuals = verify_atom_measure(
            atoms,
            weights,
            basis,
            moment_matrix,
            variables,
        )
        return AtomExtractionResult(
            True,
            rank_m,
            rank_prev,
            atoms,
            weights,
            "Extracted atoms from a one-dimensional multiplication matrix.",
            mults,
            verified,
            residuals,
        )

    if len(variables) >= 2 and mults:
        first = np.array(
            mults[str(variables[0])].evalf().tolist(),
            dtype=float,
        )
        eigenvalues, vectors = np.linalg.eig(first)
        for index, eigenvalue in enumerate(eigenvalues):
            vector = vectors[:, index]
            atom = {variables[0]: sp.nsimplify(np.real(eigenvalue))}
            for variable in variables[1:]:
                matrix = np.array(
                    mults[str(variable)].evalf().tolist(),
                    dtype=float,
                )
                num = np.vdot(vector, matrix @ vector)
                den = np.vdot(vector, vector)
                atom[variable] = sp.nsimplify(np.real(num / den))
            atoms.append(atom)

        atoms = _dedupe_atoms(atoms, variables)
        weights = _solve_weights(atoms, basis, moment_matrix)
        verified, residuals = verify_atom_measure(
            atoms,
            weights,
            basis,
            moment_matrix,
            variables,
        )
        return AtomExtractionResult(
            True,
            rank_m,
            rank_prev,
            atoms,
            weights,
            "Extracted atoms from multiplication matrices.",
            mults,
            verified,
            residuals,
        )

    verified, residuals = (
        verify_atom_measure(
            atoms,
            weights,
            basis,
            moment_matrix,
            variables,
        )
        if atoms and weights
        else (False, None)
    )
    return AtomExtractionResult(
        flat,
        rank_m,
        rank_prev,
        atoms,
        weights,
        "Flatness detected but atom extraction was unavailable.",
        mults,
        verified,
        residuals,
    )


def verify_atom_measure(
    atoms,
    weights,
    basis,
    moment_matrix,
    vars,
):
    if not atoms or not weights:
        return False, {"message": sp.Symbol("no_atoms")}

    if len(weights) != len(atoms):
        return False, {"message": sp.Symbol("weight_count_mismatch")}

    residuals = {}
    verified = True
    size = min(len(basis), moment_matrix.rows, moment_matrix.cols)
    for row in range(size):
        for col in range(row, size):
            term = sp.expand(basis[row] * basis[col])
            lhs = sp.nsimplify(
                sum(
                    weight * sp.expand(term.subs(atom))
                    for weight, atom in zip(weights, atoms)
                )
            )
            rhs = sp.nsimplify(moment_matrix[row, col])
            residual = sp.simplify(lhs - rhs)
            key = f"{sp.sstr(basis[row])}*{sp.sstr(basis[col])}"
            residuals[key] = residual
            verified = verified and residual == 0

    return verified, residuals
