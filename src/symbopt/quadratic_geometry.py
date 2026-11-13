"""Exact equality reduction and feasible recession witnesses for quadratic models."""

from dataclasses import dataclass

import sympy as sp

from .quadratic_faces import FaceCertificate, verify_face_certificate
from .quadratic_lp import affine_system, exact_linear_minimize
from .quadratic_models import GlobalQuadraticResult
from .symbolic_cache import cached_gradient, cached_hessian


@dataclass(frozen=True)
class EqualityCertificate:
    """A stationary affine-subspace minimum or an explicit feasible escaping ray."""

    problem: object
    kind: str
    anchor: tuple
    direction: tuple = ()


def negative_direction(matrix):
    """Find rational negative curvature by symmetric Schur-complement elimination.

    Positive pivots eliminate a coordinate by minimizing in that coordinate.
    A zero diagonal with a nonzero off-diagonal supplies a negative two-coordinate
    direction, so semidefinite and singular matrices need no floating eigenvalues.
    """
    matrix = sp.Matrix(matrix)
    basis = sp.eye(matrix.rows)
    while matrix.rows:
        n = matrix.rows
        for i in range(n):
            if matrix[i, i] < 0:
                return tuple(basis[:, i])
        for i in range(n):
            if matrix[i, i] != 0:
                continue
            for j in range(n):
                if i != j and matrix[i, j] != 0:
                    vector = sp.zeros(n, 1)
                    vector[i] = 1
                    vector[j] = -matrix[i, j] / (matrix[j, j] + 1)
                    return tuple(basis * vector)
        pivot = next((i for i in range(n) if matrix[i, i] > 0), None)
        if pivot is None:
            return None
        rest = [i for i in range(n) if i != pivot]
        row = matrix.extract([pivot], rest)
        column = matrix.extract(rest, [pivot])
        basis = (
            basis.extract(list(range(basis.rows)), rest)
            - basis[:, pivot] * row / matrix[pivot, pivot]
        )
        matrix = matrix.extract(rest, rest) - column * row / matrix[pivot, pivot]
    return None


def verify_equality_certificate(problem, certificate):
    """Check stationarity/PSD or ray descent on the original feasible domain."""
    if (
        not isinstance(certificate, EqualityCertificate)
        or certificate.problem != problem
        or not problem.rational
        or problem.integer_variables
    ):
        return False
    system = affine_system(problem.constraints, problem.variables)
    if system is None or not system.feasible(certificate.anchor):
        return False
    n = len(problem.variables)
    if certificate.kind == "ray":
        d = certificate.direction
        if len(d) != n or any(sp.sympify(v).is_Rational is not True for v in d):
            return False
        if any(sum(a * x for a, x in zip(row, d)) < 0 for row in system.inequalities):
            return False
        if any(sum(a * x for a, x in zip(row, d)) != 0 for row in system.equalities):
            return False
        t = sp.Dummy("ray", positive=True)
        substitution = dict(
            zip(problem.variables, [p + t * v for p, v in zip(certificate.anchor, d)])
        )
        poly = sp.Poly(problem.objective.subs(substitution), t)
        return poly.degree() > 0 and poly.LC() < 0
    if certificate.kind != "minimum" or system.inequalities or certificate.direction:
        return False
    a = sp.Matrix(system.equalities) if system.equalities else sp.zeros(0, n)
    vectors = a.nullspace()
    if not vectors:
        return True
    basis = sp.Matrix.hstack(*vectors)
    hessian = sp.Matrix(cached_hessian(problem.objective, problem.variables))
    reduced = basis.T * hessian * basis
    if negative_direction(reduced) is not None:
        return False
    point = dict(zip(problem.variables, certificate.anchor))
    gradient = sp.Matrix(
        [v.subs(point) for v in cached_gradient(problem.objective, problem.variables)]
    )
    return all(v == 0 for v in basis.T * gradient)


def solve_equalities(problem, controls):
    """Solve unrestricted rational quadratics on affine subspaces exactly."""
    if not problem.rational or problem.integer_variables:
        return None
    system = affine_system(problem.constraints, problem.variables)
    if system is None:
        return None
    feasible = exact_linear_minimize((sp.S.Zero,) * system.size, system)
    if feasible.status == "infeasible":
        proof = FaceCertificate(
            problem,
            "infeasible",
            sp.oo,
            sp.oo,
            complete=True,
            infeasibility=feasible.proof,
        )
        verified = verify_face_certificate(problem, proof)
        return GlobalQuadraticResult(
            "infeasible",
            problem.variables,
            lower_bound=sp.oo,
            backend="equalities",
            certified=controls.certified and verified,
            bounds_verified=verified,
            attained=False,
            certificate=proof,
        )
    if feasible.status != "optimal":
        return None
    n = system.size
    a = sp.Matrix(system.equalities) if system.equalities else sp.zeros(0, n)
    basis = sp.Matrix.hstack(*a.nullspace()) if a.nullspace() else sp.zeros(n, 0)
    hessian = sp.Matrix(cached_hessian(problem.objective, problem.variables))
    reduced = basis.T * hessian * basis
    direction = negative_direction(reduced)
    directions = [tuple(basis * v) for v in reduced.nullspace()]
    if direction is not None:
        directions.append(tuple(basis * sp.Matrix(direction)))
    for vector in basis.columnspace():
        directions.append(tuple(vector))
    if basis.cols:
        directions.append(tuple(basis * sp.ones(basis.cols, 1)))
    # A failed ray search is only a refusal; it never proves boundedness.
    for d in directions:
        for sign in (1, -1):
            ray = EqualityCertificate(
                problem, "ray", feasible.point, tuple(sign * v for v in d)
            )
            if verify_equality_certificate(problem, ray):
                return GlobalQuadraticResult(
                    "unbounded",
                    problem.variables,
                    -sp.oo,
                    (),
                    -sp.oo,
                    -sp.oo,
                    "equalities",
                    certified=controls.certified,
                    bounds_verified=True,
                    attained=False,
                    certificate=ray,
                )
    if system.inequalities:
        return None
    from .quadratic_faces import face_system

    _, _, solution, free, _ = face_system(problem, system, ())
    if not solution:
        return None
    zero = dict.fromkeys(free, sp.S.Zero)
    point = tuple(sp.cancel(v.subs(zero)) for v in solution[:n])
    proof = EqualityCertificate(problem, "minimum", point)
    if not verify_equality_certificate(problem, proof):
        return None
    value = problem.value(point)
    return GlobalQuadraticResult(
        "optimal",
        problem.variables,
        value,
        point,
        value,
        value,
        "equalities",
        certified=controls.certified,
        bounds_verified=True,
        attained=True,
        certificate=proof,
    )
