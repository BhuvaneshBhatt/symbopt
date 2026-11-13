from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from sympy.polys.polyerrors import PolynomialError

from .chordal import SparsityGraph as IndexedSparsityGraph
from .chordal import chordal_extension
from .monomials import (
    Exponent,
    add_exponents,
    basis_data,
    coefficient_map,
    exponent_tuples,
)
from .monomials import (
    monomial_basis as cached_basis,
)
from .polynomial_structure import build_sparsity_graph, sparse_supported_basis
from .sdp import SDPProblem


@dataclass
class PSDGramBlock:
    """A polynomial basis and nonnegative multiplier for one PSD Gram block."""

    name: str
    basis: list[sp.Expr]
    multiplier: sp.Expr | None = None
    variables: list[sp.Symbol] = field(default_factory=list)
    basis_terms: tuple[tuple[sp.Expr, Exponent], ...] = field(default_factory=tuple)


@dataclass
class EqualityMultiplierBlock:
    """A polynomial equality and a basis for its unrestricted multiplier."""

    name: str
    basis: list[sp.Expr]
    polynomial: sp.Expr
    variables: list[sp.Symbol] = field(default_factory=list)
    basis_terms: tuple[tuple[sp.Expr, Exponent], ...] = field(default_factory=tuple)


@dataclass
class SOSCompilation:
    """Polynomial identity, scaled bases and coefficient constraints for an SOS search."""

    polynomial: sp.Expr
    variables: list[sp.Symbol]
    degree_main: int
    fixed_gamma: sp.Expr
    normalization_scale: sp.Expr
    normalized_polynomial: sp.Expr
    main_gram_count: int = 1
    variable_blocks: list[list[sp.Symbol]] = field(default_factory=list)
    clique_separators: list[list[sp.Symbol]] = field(default_factory=list)
    sparsity_mode: str = "dense"
    gram_blocks: list[PSDGramBlock] = field(default_factory=list)
    equality_blocks: list[EqualityMultiplierBlock] = field(default_factory=list)
    scalar_names: list[str] = field(default_factory=list)
    sdp_problem: SDPProblem | None = None

    def pretty(self) -> str:
        """Format the stored mathematical result and diagnostic metadata for inspection."""
        return "\n".join(
            [
                f"degree_main: {self.degree_main}",
                f"variables: {[sp.sstr(v) for v in self.variables]}",
                f"gram_blocks: {[block.name for block in self.gram_blocks]}",
                f"equality_blocks: {[block.name for block in self.equality_blocks]}",
                f"num_unrestricted_scalars: {len(self.scalar_names)}",
                f"fixed_gamma: {sp.sstr(self.fixed_gamma)}",
                f"normalization_scale: {sp.sstr(self.normalization_scale)}",
            ]
        )


def monomial_exponent_tuples(
    size: int,
    max_degree: int,
) -> list[Exponent]:
    """Return exponent tuples ordered by total degree and then lexicographically."""
    return list(exponent_tuples(size, max_degree))


def monomial_basis(
    vars: Sequence[sp.Symbol],
    max_degree: int,
) -> list[sp.Expr]:
    """Return monomials of bounded total degree in the canonical exponent order."""
    return list(cached_basis(tuple(vars), max_degree))


def _support_degree_bounds(
    exprs: Sequence[sp.Expr],
    vars: Sequence[sp.Symbol],
) -> tuple[int, ...]:
    bounds = [0] * len(vars)
    for expr in exprs:
        try:
            for powers in coefficient_map(expr, vars):
                for index, power in enumerate(powers):
                    bounds[index] = max(bounds[index], power)
        except (
            PolynomialError,
            TypeError,
            ValueError,
            ArithmeticError,
            NotImplementedError,
        ):
            continue
    return tuple(bounds)


def _safe_box_prune_basis(
    basis: Sequence[sp.Expr],
    exprs: Sequence[sp.Expr],
    vars: Sequence[sp.Symbol],
) -> list[sp.Expr]:
    """Prune outside a conservative coordinatewise half-Newton box."""
    bounds = _support_degree_bounds(exprs, vars)
    if not any(bounds):
        return list(basis)

    data = basis_data(tuple(basis), tuple(vars))
    pruned = [
        term
        for term, (_, powers) in zip(basis, data)
        if all(2 * power <= bound for power, bound in zip(powers, bounds))
    ]
    return pruned if pruned else [sp.Integer(1)]


def sparse_monomial_basis(
    exprs: Sequence[sp.Expr],
    vars: Sequence[sp.Symbol],
    max_degree: int,
) -> list[sp.Expr]:
    """Prune a polynomial basis using variable components and coordinate support bounds."""
    raw = monomial_basis(vars, max_degree)
    graph = build_sparsity_graph(exprs, vars)
    components = graph.connected_components()
    component_basis = sparse_supported_basis(raw, components, vars)
    return _safe_box_prune_basis(component_basis, exprs, vars)


def poly_coeff_dict(
    expr: sp.Expr,
    vars: Sequence[sp.Symbol],
) -> dict[Exponent, sp.Expr]:
    """Map each exponent tuple to its exact polynomial coefficient."""
    return coefficient_map(expr, vars)


def basis_exponents(
    basis: Sequence[sp.Expr],
    vars: Sequence[sp.Symbol],
) -> list[Exponent]:
    """Extract each basis term's exponent tuple in the supplied variable order."""
    return [powers for _, powers in basis_data(tuple(basis), tuple(vars))]


def max_abs_coeff(
    expr: sp.Expr,
    vars: Sequence[sp.Symbol],
) -> sp.Expr:
    """Estimate the largest coefficient magnitude for numerical normalization."""
    try:
        coeffs = list(coefficient_map(expr, vars).values())
        if not coeffs:
            return sp.Integer(1)
        values = [abs(complex(sp.N(coeff, 40))) for coeff in coeffs]
        maximum = max(values)
        return sp.nsimplify(maximum) if maximum > 0 else sp.Integer(1)
    except (
        PolynomialError,
        TypeError,
        ValueError,
        ArithmeticError,
        NotImplementedError,
    ):
        return sp.Integer(1)


def basis_scaling_weights(
    expr: sp.Expr,
    basis: Sequence[sp.Expr],
    vars: Sequence[sp.Symbol],
) -> list[float]:
    """Choose numerical diagonal basis weights from the polynomial coefficients."""
    coeffs = coefficient_map(expr, vars)
    weights = []
    for _, powers in basis_data(tuple(basis), tuple(vars)):
        doubled = tuple(2 * power for power in powers)
        coeff = coeffs.get(doubled, sp.Integer(1))
        value = abs(complex(sp.N(coeff, 30)))
        scale = value**0.5 if value > 0 else 1.0
        weights.append(float(1.0 / max(1.0, scale)))
    return weights


def scaled_basis(
    basis: Sequence[sp.Expr],
    weights: Sequence[float],
) -> list[sp.Expr]:
    """Apply the supplied diagonal weights to the monomial basis."""
    return [sp.nsimplify(weight) * term for term, weight in zip(basis, weights)]


def qmat_expr(name: str, size: int) -> sp.Matrix:
    """Construct a symmetric symbolic Gram matrix with one symbol per upper-triangle entry."""
    matrix = sp.MutableDenseMatrix.zeros(size, size)
    for row in range(size):
        for col in range(row, size):
            symbol = sp.Symbol(f"{name}_{row}_{col}", real=True)
            matrix[row, col] = symbol
            matrix[col, row] = symbol
    return sp.Matrix(matrix)


def sos_expr_from_blocks(
    gram_blocks: Sequence[tuple[list[sp.Expr], sp.Matrix, sp.Expr | None]],
) -> sp.Expr:
    """Assemble the polynomial identity contributed by Gram blocks and their multipliers."""
    expression = sp.Integer(0)
    for basis, matrix, multiplier in gram_blocks:
        vector = sp.Matrix(list(basis))
        term = (vector.T * matrix * vector)[0]
        expression += term if multiplier is None else sp.expand(multiplier * term)
    return sp.expand(expression)


def coeff_blocks_to_poly(
    block: EqualityMultiplierBlock,
    coeff_pos: Sequence[sp.Symbol],
    coeff_neg: Sequence[sp.Symbol],
) -> sp.Expr:
    """Assemble a polynomial from exponent-indexed coefficient blocks."""
    expression = sp.Integer(0)
    for term, pos, neg in zip(block.basis, coeff_pos, coeff_neg):
        expression += (pos - neg) * term
    return sp.expand(expression)


def _gram_coeff_rows(
    basis: Sequence[sp.Expr],
    multiplier: sp.Expr | None,
    vars: Sequence[sp.Symbol],
    data: Sequence[tuple[sp.Expr, Exponent]] | None = None,
) -> dict[Exponent, np.ndarray]:
    variables = tuple(vars)
    data = tuple(data) if data is not None else basis_data(tuple(basis), variables)
    zero = (0,) * len(variables)
    multiplier_terms = (
        {zero: sp.Integer(1)}
        if multiplier is None
        else coefficient_map(multiplier, variables)
    )
    rows: dict[Exponent, np.ndarray] = {}
    size = len(basis)

    for row, (row_coeff, row_exp) in enumerate(data):
        for col in range(row, size):
            col_coeff, col_exp = data[col]
            base_exp = add_exponents(row_exp, col_exp)
            base_coeff = row_coeff * col_coeff

            for mult_exp, mult_coeff in multiplier_terms.items():
                powers = add_exponents(base_exp, mult_exp)
                matrix = rows.setdefault(
                    powers,
                    np.zeros((size, size), dtype=float),
                )
                value = -float(sp.N(base_coeff * mult_coeff, 50))
                matrix[row, col] += value
                if row != col:
                    matrix[col, row] += value

    return rows


def _equality_columns(
    blocks: Sequence[EqualityMultiplierBlock],
    vars: Sequence[sp.Symbol],
):
    columns = []
    names = []
    variables = tuple(vars)

    for block_index, block in enumerate(blocks, start=1):
        poly_terms = coefficient_map(block.polynomial, variables)
        block_data = block.basis_terms or basis_data(tuple(block.basis), variables)
        for term_index, (basis_coeff, basis_exp) in enumerate(block_data):
            positive: dict[Exponent, float] = {}
            negative: dict[Exponent, float] = {}
            for poly_exp, poly_coeff in poly_terms.items():
                powers = add_exponents(basis_exp, poly_exp)
                value = float(sp.N(basis_coeff * poly_coeff, 50))
                positive[powers] = positive.get(powers, 0.0) - value
                negative[powers] = negative.get(powers, 0.0) + value

            pos_name = f"U{block_index}_{term_index}_pos"
            neg_name = f"U{block_index}_{term_index}_neg"
            names.extend([pos_name, neg_name])
            columns.extend([positive, negative])

    return names, columns


def _active_components(
    exprs: Sequence[sp.Expr],
    vars: Sequence[sp.Symbol],
) -> list[list[sp.Symbol]]:
    """Return nonempty disconnected correlative variable components."""
    graph = build_sparsity_graph(exprs, vars)
    components = graph.connected_components()
    used = set().union(*(expr.free_symbols for expr in exprs))
    active = [[var for var in component if var in used] for component in components]
    active = [component for component in active if component]
    return active or [list(vars)]


def _chordal_cliques(
    exprs: Sequence[sp.Expr],
    vars: Sequence[sp.Symbol],
) -> tuple[list[list[sp.Symbol]], list[list[sp.Symbol]]]:
    """Return maximal cliques of a chordal extension of correlative sparsity.

    Unlike disconnected-component splitting, these cliques may overlap.  Every
    original polynomial monomial support is a clique in the interaction graph
    and is therefore contained in at least one maximal clique after extension.
    """
    variables = list(vars)
    graph = build_sparsity_graph(exprs, variables)
    index = {v: i for i, v in enumerate(variables)}
    edges = {(min(index[a], index[b]), max(index[a], index[b])) for a, b in graph.edges}
    decomposition = chordal_extension(IndexedSparsityGraph(len(variables), edges))
    cliques = [[variables[i] for i in clique] for clique in decomposition.cliques]
    separators = [[variables[i] for i in sep] for sep in decomposition.separators]
    used = set().union(*(expr.free_symbols for expr in exprs)) if exprs else set()
    cliques = [[v for v in clique if v in used] for clique in cliques]
    cliques = [clique for clique in cliques if clique]
    if not cliques:
        cliques = [variables]
    maximal = []
    for clique in sorted(cliques, key=lambda c: (-len(c), tuple(map(sp.sstr, c)))):
        if not any(set(clique).issubset(set(other)) for other in maximal):
            maximal.append(clique)
    maximal.sort(key=lambda c: tuple(index[v] for v in c))
    return maximal, separators


def _component_terms(
    expr: sp.Expr,
    component: Sequence[sp.Symbol],
    vars: Sequence[sp.Symbol],
) -> sp.Expr:
    """Extract polynomial terms whose variable support lies in one component."""
    component_set = set(component)
    try:
        poly = sp.Poly(sp.expand(expr), *vars)
    except (
        PolynomialError,
        TypeError,
        ValueError,
        ArithmeticError,
        NotImplementedError,
    ):
        return expr if expr.free_symbols.issubset(component_set) else sp.Integer(0)
    pieces = []
    for powers, coeff in poly.terms():
        support = {var for var, power in zip(vars, powers) if power}
        if support.issubset(component_set):
            term = coeff
            for var, power in zip(vars, powers):
                if power:
                    term *= var**power
            pieces.append(term)
    return sp.expand(sum(pieces, sp.Integer(0)))


def _component_basis(
    component: Sequence[sp.Symbol],
    degree: int,
    exprs: Sequence[sp.Expr],
    vars: Sequence[sp.Symbol],
) -> list[sp.Expr]:
    raw = monomial_basis(component, degree)
    return _safe_box_prune_basis(raw, exprs, vars)


def _support_component(
    expr: sp.Expr,
    components: Sequence[Sequence[sp.Symbol]],
) -> list[sp.Symbol] | None:
    support = set(expr.free_symbols)
    if not support:
        return []
    for component in components:
        if support.issubset(set(component)):
            return list(component)
    return None


def compile_sos_relaxation(
    polynomial: sp.Expr,
    vars: Sequence[sp.Symbol],
    degree_main: int,
    inequalities: Sequence[sp.Expr] = (),
    equalities: Sequence[sp.Expr] = (),
    fixed_gamma: sp.Expr = 0,
    ineq_mult_degree: int = 1,
    eq_mult_degree: int = 1,
) -> SOSCompilation:
    """Compile a polynomial nonnegativity claim into sparse SOS/SDP blocks."""
    variables = list(vars)
    polynomial = sp.expand(polynomial)
    scale = max_abs_coeff(polynomial, variables)
    normalized = sp.expand(polynomial / scale)
    target = sp.expand(normalized - fixed_gamma / scale)

    sparse_exprs = [polynomial, *inequalities, *equalities]
    components = _active_components(sparse_exprs, variables)
    cliques, clique_separators = _chordal_cliques(sparse_exprs, variables)
    use_chordal = len(cliques) > 1 and any(
        set(cliques[i]).intersection(cliques[j])
        for i in range(len(cliques))
        for j in range(i)
    )
    main_blocks = cliques if len(cliques) > 1 else components
    sparsity_mode = (
        "chordal-overlap"
        if use_chordal
        else "disconnected"
        if len(main_blocks) > 1
        else "dense"
    )

    gram_blocks = []
    if len(main_blocks) > 1:
        for block_index, component in enumerate(main_blocks, start=1):
            component_exprs = [
                local
                for expr in sparse_exprs
                if (local := _component_terms(expr, component, variables)) != 0
            ]
            if not component_exprs:
                # A fill clique can be needed for PSD completion even when it
                # owns no original term; retain a constant/local basis.
                component_exprs = [sp.Integer(1)]
            raw = _component_basis(component, degree_main, component_exprs, variables)
            basis = scaled_basis(
                raw,
                basis_scaling_weights(normalized, raw, variables),
            )
            gram_blocks.append(
                PSDGramBlock(f"sigma0_{block_index}", basis, None, list(component))
            )
    else:
        main_raw = sparse_monomial_basis(sparse_exprs, variables, degree_main)
        main_basis = scaled_basis(
            main_raw,
            basis_scaling_weights(normalized, main_raw, variables),
        )
        gram_blocks.append(PSDGramBlock("sigma0", main_basis, None, list(variables)))

    main_gram_count = len(gram_blocks)

    for index, inequality in enumerate(inequalities, start=1):
        component = _support_component(inequality, components)
        basis_vars = variables if component is None or not component else component
        local_exprs = [inequality]
        raw = monomial_basis(basis_vars, ineq_mult_degree)
        raw = _safe_box_prune_basis(raw, local_exprs, variables)
        basis = scaled_basis(
            raw,
            basis_scaling_weights(normalized, raw, variables),
        )
        gram_blocks.append(
            PSDGramBlock(
                f"sigma{index}",
                basis,
                sp.expand(inequality),
                list(basis_vars),
            )
        )

    equality_blocks = []
    for index, equality in enumerate(equalities, start=1):
        component = _support_component(equality, components)
        basis_vars = variables if component is None or not component else component
        raw = monomial_basis(basis_vars, eq_mult_degree)
        raw = _safe_box_prune_basis(raw, [equality], variables)
        basis = scaled_basis(
            raw,
            basis_scaling_weights(normalized, raw, variables),
        )
        equality_blocks.append(
            EqualityMultiplierBlock(
                f"tau{index}",
                basis,
                sp.expand(equality),
            )
        )

    for block in gram_blocks:
        block.basis_terms = basis_data(tuple(block.basis), tuple(variables))
    for block in equality_blocks:
        block.variables = list(variables)
        block.basis_terms = basis_data(tuple(block.basis), tuple(variables))

    target_coeffs = coefficient_map(target, variables)
    gram_rows = [
        _gram_coeff_rows(
            block.basis,
            block.multiplier,
            variables,
            block.basis_terms,
        )
        for block in gram_blocks
    ]
    scalar_names, scalar_cols = _equality_columns(
        equality_blocks,
        variables,
    )

    powers = set(target_coeffs)
    for rows in gram_rows:
        powers.update(rows)
    for column in scalar_cols:
        powers.update(column)
    row_powers = sorted(powers, key=lambda value: (sum(value), value))

    a_blocks: list[list[np.ndarray]] = []
    b_rows: list[float] = []
    B = (
        np.zeros((len(row_powers), len(scalar_cols)), dtype=float)
        if scalar_cols
        else None
    )

    block_sizes = [len(block.basis) for block in gram_blocks]
    for row_index, powers in enumerate(row_powers):
        matrices = []
        for size, rows in zip(block_sizes, gram_rows):
            matrices.append(
                rows.get(
                    powers,
                    np.zeros((size, size), dtype=float),
                )
            )
        a_blocks.append(matrices)

        target_coeff = target_coeffs.get(powers, sp.Integer(0))
        b_rows.append(-float(sp.N(target_coeff, 50)))

        if B is not None:
            for col_index, column in enumerate(scalar_cols):
                B[row_index, col_index] = column.get(powers, 0.0)

    problem = SDPProblem(
        C_blocks=[np.zeros((size, size), dtype=float) for size in block_sizes],
        A_blocks=a_blocks,
        b=np.array(b_rows, dtype=float),
        B=B,
        c=(np.zeros(len(scalar_cols), dtype=float) if scalar_cols else None),
        name="sos_compilation",
    )

    return SOSCompilation(
        polynomial=polynomial,
        variables=variables,
        degree_main=degree_main,
        fixed_gamma=fixed_gamma,
        normalization_scale=scale,
        normalized_polynomial=normalized,
        main_gram_count=main_gram_count,
        variable_blocks=[list(component) for component in main_blocks],
        clique_separators=[list(separator) for separator in clique_separators],
        sparsity_mode=sparsity_mode,
        gram_blocks=gram_blocks,
        equality_blocks=equality_blocks,
        scalar_names=scalar_names,
        sdp_problem=problem,
    )
