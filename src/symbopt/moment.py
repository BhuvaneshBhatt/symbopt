from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import sympy as sp
from scipy.linalg import eigh, qr

from .monomials import (
    Exponent,
    add_exponents,
    add_three,
    basis_data,
    coefficient_map,
    moment_symbol,
)
from .monomials import (
    monomial_basis as cached_basis,
)


@dataclass
class MomentMatrix:
    """A truncated moment matrix with its basis and indexed moment symbols."""

    basis: list[sp.Expr]
    matrix: sp.Matrix


@dataclass
class LocalizingMatrix:
    """A polynomial-weighted localizing matrix on a truncated monomial basis."""

    polynomial: sp.Expr
    basis: list[sp.Expr]
    matrix: sp.Matrix


@dataclass
class MomentRelaxation:
    """Moment and localizing matrices for a polynomial optimization relaxation."""

    order: int
    variables: list[sp.Symbol]
    basis: list[sp.Expr]
    moment_matrix: MomentMatrix
    localizing_matrices: list[LocalizingMatrix]
    objective_linear_form: sp.Expr


@dataclass
class AtomExtractionResult:
    """Recovered support points, weights and exact or numerical moment-verification evidence."""

    flat: bool
    rank_m: int
    rank_m_minus_1: int | None
    atoms: list[dict[sp.Symbol, sp.Expr]]
    weights: list[sp.Expr]
    message: str
    multiplication_matrices: dict[str, sp.Matrix] | None = None
    verified: bool = False
    residuals: dict[str, sp.Expr] | None = None

    def pretty(self) -> str:
        """Format the stored mathematical result and diagnostic metadata for inspection."""
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
) -> list[sp.Expr]:
    """Return monomials of bounded total degree in the canonical exponent order."""
    return list(cached_basis(tuple(vars), degree))


def _moment_value(
    powers: Exponent,
    values: dict[Exponent, sp.Expr] | None,
) -> sp.Expr:
    if values is None:
        return moment_symbol(powers)
    return values[powers]


def build_moment_matrix(
    vars: Sequence[sp.Symbol],
    order: int,
    moment_values: dict[Exponent, sp.Expr] | None = None,
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
    moment_values: dict[Exponent, sp.Expr] | None = None,
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
) -> list[Exponent]:
    variables = tuple(vars)
    return [powers for _, powers in basis_data(tuple(basis), variables)]


def _solve_weights(atoms, basis, moment_matrix, indices, constant_index):
    """Recover weights in the same independent monomial basis used for extraction."""
    if not atoms:
        return []
    constant = basis[constant_index]
    rows = [
        [sp.expand(basis[i].subs(atom, simultaneous=True)) for atom in atoms]
        for i in indices
    ]
    rhs = [moment_matrix[i, constant_index] / constant for i in indices]
    try:
        solution = sp.Matrix(rows).gauss_jordan_solve(sp.Matrix(rhs))[0]
        return [sp.nsimplify(value) for value in solution]
    except (TypeError, ValueError, ArithmeticError, NotImplementedError):
        mass = moment_matrix[constant_index, constant_index] / constant**2
        return [sp.nsimplify(mass / len(atoms)) for _ in atoms]


def _dedupe_atoms(
    atoms: list[dict[sp.Symbol, sp.Expr]],
    vars: Sequence[sp.Symbol],
) -> list[dict[sp.Symbol, sp.Expr]]:
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
    *,
    max_separating_attempts: int = 16,
) -> AtomExtractionResult:
    """Recover real atoms and positive weights from numerically flat moment data.

    Pivoted QR selects an independent previous-order monomial basis. Joint
    diagonalization uses a separating linear combination of coordinate
    operators, so individual coordinates may have repeated values. Numerical
    proposals are verified against the complete original moment matrix exactly.
    A failed separation or reconstruction returns an unverified result.
    """
    if (
        isinstance(max_separating_attempts, bool)
        or not isinstance(max_separating_attempts, int)
        or max_separating_attempts < 1
    ):
        raise ValueError("max_separating_attempts must be a positive integer")
    variables = [vars] if isinstance(vars, sp.Symbol) else list(vars)
    basis = tuple(map(sp.sympify, basis))
    if moment_matrix.shape != (len(basis), len(basis)):
        raise ValueError("the moment matrix must match the supplied basis")
    data = basis_data(tuple(basis), tuple(variables))
    powers = [exponents for _, exponents in data]
    degree = max((sum(exponents) for exponents in powers), default=0)
    previous = [i for i, exponents in enumerate(powers) if sum(exponents) < degree]
    rank_m = _numeric_rank(moment_matrix, tol)
    rank_prev = (
        _numeric_rank(moment_matrix.extract(previous, previous), tol)
        if previous
        else None
    )
    flat = rank_prev is not None and rank_prev == rank_m
    if not flat:
        return AtomExtractionResult(
            False, rank_m, rank_prev, [], [], "Flatness failed."
        )
    constant_index = next(
        (i for i, exponents in enumerate(powers) if not any(exponents)), None
    )
    if constant_index is None:
        return AtomExtractionResult(
            True, rank_m, rank_prev, [], [], "The basis must include a constant."
        )
    if rank_m == 0:
        atoms = [dict.fromkeys(variables, sp.S.Zero)]
        verified, residuals = verify_atom_measure(
            atoms, [sp.S.Zero], basis, moment_matrix, variables
        )
        return AtomExtractionResult(
            True,
            rank_m,
            rank_prev,
            atoms,
            [sp.S.Zero],
            "Zero moment measure.",
            {},
            verified,
            residuals,
        )

    numeric = np.array(moment_matrix.evalf().tolist(), dtype=float)
    previous_matrix = numeric[np.ix_(previous, previous)]
    _, _, pivots = qr(previous_matrix, pivoting=True, mode="economic")
    indices = [previous[i] for i in pivots[:rank_m]]
    base = numeric[np.ix_(indices, indices)]
    exp_map = {exponents: i for i, exponents in enumerate(powers)}
    shifted = []
    mults = {}
    for var_pos, variable in enumerate(variables):
        columns = []
        for index in indices:
            exponents = list(powers[index])
            exponents[var_pos] += 1
            target = exp_map.get(tuple(exponents))
            if target is None:
                return AtomExtractionResult(
                    True,
                    rank_m,
                    rank_prev,
                    [],
                    [],
                    "The basis lacks a required shifted monomial.",
                )
            scale = float(sp.N(data[index][0] / data[target][0]))
            columns.append(scale * numeric[indices, target])
        matrix = np.column_stack(columns)
        shifted.append(matrix)
        try:
            mults[str(variable)] = sp.Matrix(np.linalg.solve(base, matrix))
        except np.linalg.LinAlgError:
            return AtomExtractionResult(
                True,
                rank_m,
                rank_prev,
                [],
                [],
                "The selected moment basis is numerically singular.",
            )

    atoms, weights, residuals = [], [], None
    for attempt in range(1, max_separating_attempts + 1):
        coefficients = np.array(
            [attempt**i for i in range(len(variables))], dtype=float
        )
        coefficients /= np.max(coefficients)
        combination = sum(c * matrix for c, matrix in zip(coefficients, shifted))
        try:
            eigenvalues, vectors = eigh(
                (combination + combination.T) / 2, (base + base.T) / 2
            )
        except np.linalg.LinAlgError:
            break
        if len(eigenvalues) > 1 and np.min(np.diff(eigenvalues)) <= tol * max(
            1.0, np.max(np.abs(eigenvalues))
        ):
            continue
        atoms = []
        for column in range(rank_m):
            vector = vectors[:, column]
            norm = vector @ base @ vector
            atom = {}
            for variable, matrix in zip(variables, shifted):
                value = float(vector @ matrix @ vector / norm)
                # Rounding produces candidates only; exact verification below
                # rejects any genuine small atom that this proposal loses.
                atom[variable] = (
                    sp.S.Zero
                    if abs(value) <= tol
                    else sp.nsimplify(value, tolerance=min(tol, 1e-12))
                )
            atoms.append(atom)
        atoms = _dedupe_atoms(atoms, variables)
        weights = _solve_weights(atoms, basis, moment_matrix, indices, constant_index)
        verified, residuals = verify_atom_measure(
            atoms, weights, basis, moment_matrix, variables
        )
        if verified:
            return AtomExtractionResult(
                True,
                rank_m,
                rank_prev,
                atoms,
                weights,
                "Extracted and verified a joint atomic measure.",
                mults,
                True,
                residuals,
            )
    return AtomExtractionResult(
        True,
        rank_m,
        rank_prev,
        atoms,
        weights,
        "Joint atom extraction remained unverified.",
        mults,
        False,
        residuals,
    )


def verify_atom_measure(
    atoms,
    weights,
    basis,
    moment_matrix,
    vars,
):
    """Verify a positive real atomic measure against every supplied moment.

    Numerical reconstruction generates proposals during extraction. Verification
    compares the original stored moments without rounding them to nearby exact
    values, and requires a complete symmetric matrix for the supplied basis.
    """
    if not atoms or not weights:
        return False, {"message": sp.Symbol("no_atoms")}
    if len(weights) != len(atoms):
        return False, {"message": sp.Symbol("weight_count_mismatch")}
    size = len(basis)
    if not size or moment_matrix.shape != (size, size):
        return False, {"message": sp.Symbol("moment_shape_mismatch")}
    if moment_matrix != moment_matrix.T:
        return False, {"message": sp.Symbol("nonsymmetric_moments")}
    if any(sp.sympify(weight).is_nonnegative is not True for weight in weights):
        return False, {"message": sp.Symbol("nonpositive_measure")}
    if any(
        variable not in atom or sp.sympify(atom[variable]).is_real is not True
        for atom in atoms
        for variable in vars
    ):
        return False, {"message": sp.Symbol("nonreal_or_incomplete_atoms")}

    residuals = {}
    verified = True
    for row in range(size):
        for col in range(row, size):
            term = sp.expand(basis[row] * basis[col])
            lhs = sum(
                weight * term.subs(atom, simultaneous=True)
                for weight, atom in zip(weights, atoms)
            )
            residual = sp.simplify(lhs - moment_matrix[row, col])
            key = f"{sp.sstr(basis[row])}*{sp.sstr(basis[col])}"
            residuals[key] = residual
            verified = verified and residual == 0
    return verified, residuals
