from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import sympy as sp
from sympy.polys.numberfields import to_number_field
from sympy.polys.polyerrors import PolynomialError
from sympy.solvers.solveset import NonlinearError


@dataclass(frozen=True)
class ExactRefinementResult:
    """Exact solution recovered near a numerical root of symbolic equations."""

    point: Mapping[sp.Symbol, sp.Expr] | None
    method: str | None
    verified: bool
    distance: float


def _exact_zero(expr: sp.Expr) -> bool:
    try:
        value = sp.cancel(sp.together(sp.expand(expr)))
    except (TypeError, ValueError, ArithmeticError):
        value = sp.simplify(expr)
    return value == 0 or value.is_zero is True


def _numeric_vector(point, variables):
    try:
        values = np.asarray(
            [float(sp.N(point[var], 17)) for var in variables], dtype=float
        )
    except (TypeError, ValueError, OverflowError):
        return None
    return values if np.all(np.isfinite(values)) else None


def _verified(candidate, equations) -> bool:
    return all(_exact_zero(sp.sympify(eq).subs(candidate)) for eq in equations)


def _rational_numerators(equations, variables):
    numerators = []
    for equation in equations:
        expr = sp.cancel(sp.together(sp.sympify(equation)))
        numerator, denominator = sp.fraction(expr)
        if denominator == 0:
            return None
        numerator = sp.expand(numerator)
        if not numerator.is_polynomial(*variables):
            return None
        numerators.append(numerator)
    return numerators


def _choose_nearest(solutions, variables, numeric_point, equations):
    target = np.asarray(numeric_point, dtype=float)
    best = None
    best_distance = math.inf
    for solution in solutions:
        if isinstance(solution, Mapping):
            candidate = {
                var: sp.simplify(solution[var]) for var in variables if var in solution
            }
            if len(candidate) != len(variables):
                continue
        else:
            values = (
                tuple(solution) if isinstance(solution, (tuple, list, sp.Tuple)) else ()
            )
            if len(values) != len(variables):
                continue
            candidate = {
                var: sp.simplify(value) for var, value in zip(variables, values)
            }
        numeric = _numeric_vector(candidate, variables)
        if numeric is None or not _verified(candidate, equations):
            continue
        distance = float(np.linalg.norm(numeric - target))
        if distance < best_distance:
            best = candidate
            best_distance = distance
    return best, best_distance


def reconstruct_exact_vector(
    values: Sequence[object],
    *,
    tolerances: Sequence[float] = (1e-12, 1e-10, 1e-8, 1e-6),
    max_denominator: int = 10**6,
    common_number_field: bool = True,
):
    """Generate increasingly permissive exact reconstructions of a numeric vector.

    Reconstruction is simultaneous in the sense that each pass uses one common
    tolerance/denominator policy for the entire vector.  Candidates are only
    proposals; callers must verify their defining equations exactly.
    """
    raw = tuple(sp.sympify(value) for value in values)
    seen = set()
    for tolerance in tolerances:
        candidate = []
        for value in raw:
            try:
                exact = sp.nsimplify(value, tolerance=tolerance, full=True)
            except (TypeError, ValueError, AssertionError):
                exact = value
            if isinstance(exact, sp.Float):
                try:
                    exact = sp.Rational(str(exact)).limit_denominator(max_denominator)
                except (TypeError, ValueError, OverflowError):
                    pass
            candidate.append(sp.simplify(exact))
        key = tuple(sp.srepr(item) for item in candidate)
        if key not in seen:
            seen.add(key)
            yield tuple(candidate), f"simultaneous_reconstruction_{tolerance:g}"

        algebraic = [item for item in candidate if item.is_Rational is not True]
        use_common_field = (
            common_number_field
            and 0 < len(algebraic) <= 4
            and sum(int(sp.count_ops(item)) for item in algebraic) <= 80
            and all(item.is_algebraic is True for item in algebraic)
        )
        if use_common_field:
            try:
                field = to_number_field(algebraic)
                field_candidate = []
                for item in candidate:
                    if item.is_Rational is True:
                        field_candidate.append(item)
                    else:
                        field_candidate.append(to_number_field(item, field).as_expr())
                key = tuple(sp.srepr(item) for item in field_candidate)
                if key not in seen:
                    seen.add(key)
                    yield tuple(field_candidate), "common_number_field_reconstruction"
            except (NotImplementedError, PolynomialError, TypeError, ValueError):
                pass


class _AffineProjection:
    """Retain an affine system and lazily prepare its exact orthogonal projector."""

    def __init__(self, equations, variables):
        self.equations = tuple(equations)
        self.variables = tuple(variables)
        self.matrix, self.rhs = sp.linear_eq_to_matrix(self.equations, self.variables)

    @cached_property
    def projector(self):
        """Use the Gram inverse for independent rows and a pseudoinverse otherwise."""
        try:
            return self.matrix.T * (self.matrix * self.matrix.T).inv()
        except ValueError:
            try:
                return self.matrix.pinv()
            except (TypeError, ValueError, NotImplementedError):
                return None

    def project(self, candidate):
        """Project one proposal and independently check the original equations."""
        if isinstance(candidate, Mapping):
            vector = sp.Matrix([candidate[var] for var in self.variables])
        else:
            vector = sp.Matrix(list(candidate))
        if vector.rows != len(self.variables):
            raise ValueError("candidate must provide one value per variable")
        if self.matrix.rows == 0:
            return {v: sp.simplify(value) for v, value in zip(self.variables, vector)}
        residual = self.rhs - self.matrix * vector
        if all(sp.simplify(value) == 0 for value in residual):
            projected = vector
        else:
            if self.projector is None:
                return None
            projected = sp.simplify(vector + self.projector * residual)
        result = {v: sp.simplify(projected[i]) for i, v in enumerate(self.variables)}
        return result if _verified(result, self.equations) else None


def _affine_projection(equations, variables):
    try:
        return _AffineProjection(equations, variables)
    except (NonlinearError, TypeError, ValueError):
        return None


def project_affine_exact(
    equations: Sequence[sp.Expr],
    variables: Sequence[sp.Symbol],
    candidate: Sequence[sp.Expr] | Mapping[sp.Symbol, sp.Expr],
):
    """Project a candidate onto an affine system and verify its equations exactly.

    Return None for nonlinear or inconsistent systems. Repeated proposals within
    refine_exact_system share their coefficient matrix and projection operator.
    """
    prepared = _affine_projection(tuple(map(sp.sympify, equations)), variables)
    return None if prepared is None else prepared.project(candidate)


def refine_exact_system(
    equations: Sequence[sp.Expr],
    variables: Sequence[sp.Symbol],
    numeric_point: Sequence[float],
    *,
    tolerance: float = 1e-8,
    max_variables: int = 7,
    max_degree_product: int = 4096,
) -> ExactRefinementResult:
    """Recover and exactly verify a symbolic root near ``numeric_point``.

    The routine first tries direct algebraic reconstruction, then polynomial or
    rational-system solving. Numerical proximity only ranks candidates; exact
    substitution into the original equations is the acceptance criterion.
    """
    variables = tuple(variables)
    equations = tuple(sp.sympify(eq) for eq in equations)
    raw_numeric_point = tuple(sp.sympify(value) for value in numeric_point)
    numeric_point = tuple(float(sp.N(value, 30)) for value in raw_numeric_point)
    if len(variables) != len(numeric_point):
        raise ValueError("numeric_point must provide one value per variable.")
    if len(variables) > max_variables:
        return ExactRefinementResult(None, None, False, math.inf)

    projection = None
    projection_ready = False
    tolerances = tuple(dict.fromkeys((min(tolerance, 1e-12), tolerance, 1e-6)))
    for values, method in reconstruct_exact_vector(
        raw_numeric_point, tolerances=tolerances
    ):
        candidate = {var: value for var, value in zip(variables, values)}
        if _verified(candidate, equations):
            numeric = _numeric_vector(candidate, variables)
            distance = (
                math.inf
                if numeric is None
                else float(
                    np.linalg.norm(numeric - np.asarray(numeric_point, dtype=float))
                )
            )
            return ExactRefinementResult(candidate, method, True, distance)
        if not projection_ready:
            projection = _affine_projection(equations, variables)
            projection_ready = True
        projected = None if projection is None else projection.project(candidate)
        if projected is not None:
            numeric = _numeric_vector(projected, variables)
            distance = (
                math.inf
                if numeric is None
                else float(
                    np.linalg.norm(numeric - np.asarray(numeric_point, dtype=float))
                )
            )
            return ExactRefinementResult(
                projected, "exact_affine_projection", True, distance
            )

    polys = _rational_numerators(equations, variables)
    if polys is None:
        return ExactRefinementResult(None, None, False, math.inf)
    try:
        degree_product = 1
        for polynomial in polys:
            degree_product *= max(
                1, int(sp.Poly(polynomial, *variables).total_degree())
            )
        if degree_product > max_degree_product:
            return ExactRefinementResult(None, None, False, math.inf)
    except (PolynomialError, TypeError, ValueError):
        return ExactRefinementResult(None, None, False, math.inf)

    candidates = []
    method = "polynomial_solve"
    try:
        solved = sp.solve(polys, variables, dict=True, simplify=False)
        candidates.extend(solved)
    except (
        NotImplementedError,
        PolynomialError,
        TypeError,
        ValueError,
        ZeroDivisionError,
    ):
        pass

    if not candidates:
        try:
            solution_set = sp.nonlinsolve(polys, variables)
            if isinstance(solution_set, sp.FiniteSet):
                candidates.extend(list(solution_set))
                method = "nonlinsolve"
        except (
            NotImplementedError,
            PolynomialError,
            TypeError,
            ValueError,
            ZeroDivisionError,
        ):
            pass

    best, distance = _choose_nearest(candidates, variables, numeric_point, equations)
    return ExactRefinementResult(
        best, method if best is not None else None, best is not None, distance
    )
