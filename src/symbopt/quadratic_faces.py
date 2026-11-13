"""Exact global quadratic optimization by complete stationary-face enumeration."""

from dataclasses import dataclass
from itertools import combinations
from time import monotonic

import sympy as sp

from .quadratic_lp import (
    LinearProof,
    affine_system,
    exact_linear_minimize,
    verify_linear_proof,
)
from .quadratic_models import GlobalQuadraticResult
from .symbolic_cache import cached_gradient, cached_hessian


@dataclass(frozen=True)
class CoordinateBounds:
    """Finite rational bounds justified by exact linear dual witnesses."""

    intervals: tuple
    proofs: tuple


@dataclass(frozen=True)
class FaceRecord:
    """Evidence for the stationary system on one independent active face."""

    active: tuple
    kind: str
    solution: tuple = ()
    value: sp.Expr | None = None
    proof: LinearProof | None = None


@dataclass(frozen=True)
class FaceCertificate:
    """Complete face coverage or a verified interval bound for an incomplete search."""

    problem: object
    kind: str
    lower_bound: sp.Expr
    upper_bound: sp.Expr
    point: tuple = ()
    bounds: CoordinateBounds | None = None
    faces: tuple = ()
    complete: bool = False
    infeasibility: LinearProof | None = None


def dot(row, point):
    """Return the scalar product of two coordinate sequences."""
    return sum(a * x for a, x in zip(row, point))


def coordinate_bounds(system):
    """Find finite coordinate bounds; direct axis rows avoid unnecessary LPs."""
    intervals, proofs = [], []
    for i in range(system.size):
        coordinate = []
        for sign in (1, -1):
            objective = tuple(
                sp.S(sign) if j == i else sp.S.Zero for j in range(system.size)
            )
            options = []
            for k, (row, b) in enumerate(zip(system.inequalities, system.lower)):
                if row[i] * sign > 0 and all(
                    a == 0 for j, a in enumerate(row) if j != i
                ):
                    weights = tuple(
                        sign / row[i] if j == k else sp.S.Zero
                        for j in range(len(system.inequalities))
                    )
                    proof = LinearProof(
                        "bound", weights, (sp.S.Zero,) * len(system.equalities)
                    )
                    options.append((sign * b / row[i], proof))
            for k, (row, b) in enumerate(zip(system.equalities, system.rhs)):
                if row[i] != 0 and all(a == 0 for j, a in enumerate(row) if j != i):
                    weights = tuple(
                        sign / row[i] if j == k else sp.S.Zero
                        for j in range(len(system.equalities))
                    )
                    options.append(
                        (
                            sign * b / row[i],
                            LinearProof(
                                "bound",
                                (sp.S.Zero,) * len(system.inequalities),
                                weights,
                            ),
                        )
                    )
            if options:
                value, proof = max(options, key=lambda pair: pair[0])
            else:
                result = exact_linear_minimize(objective, system)
                if result.status != "optimal":
                    return None
                value, proof = result.value, result.proof
            if not verify_linear_proof(system, objective, value, proof):
                return None
            coordinate.append(value)
            proofs.append(proof)
        intervals.append((coordinate[0], -coordinate[1]))
    return CoordinateBounds(tuple(intervals), tuple(proofs))


def interval_polynomial(expression, variables, bounds):
    """Enclose a quadratic polynomial using exact rational term intervals."""
    poly = sp.Poly(expression, *variables)
    lower = upper = sp.S.Zero
    for powers, coefficient in poly.terms():
        left = right = sp.S.One
        for (lo, hi), power in zip(bounds, powers):
            if power == 0:
                continue
            if power == 2:
                a = 0 if lo <= 0 <= hi else min(lo * lo, hi * hi)
                b = max(lo * lo, hi * hi)
            else:
                a, b = lo, hi
            products = (left * a, left * b, right * a, right * b)
            left, right = min(products), max(products)
        a, b = coefficient * left, coefficient * right
        lower += min(a, b)
        upper += max(a, b)
    return lower, upper


def independent_faces(system):
    """Enumerate all independent inequality normals modulo the equality row space."""
    n = system.size
    rank = sp.Matrix(system.equalities).rank() if system.equalities else 0
    for count in range(min(n - rank, len(system.inequalities)) + 1):
        for active in combinations(range(len(system.inequalities)), count):
            rows = (*system.equalities, *(system.inequalities[i] for i in active))
            if not rows or sp.Matrix(rows).rank() == rank + count:
                yield active


def face_system(problem, system, active):
    """Parameterize a face's linear KKT system and its primal feasible slice."""
    n = system.size
    rows = (*system.equalities, *(system.inequalities[i] for i in active))
    rhs = (*system.rhs, *(system.lower[i] for i in active))
    count = len(rows)
    hessian = sp.Matrix(cached_hessian(problem.objective, problem.variables))
    zero = dict.fromkeys(problem.variables, sp.S.Zero)
    linear = sp.Matrix(
        [g.subs(zero) for g in cached_gradient(problem.objective, problem.variables)]
    )
    a = sp.Matrix(rows) if rows else sp.zeros(0, n)
    matrix = hessian.row_join(a.T).col_join(a.row_join(sp.zeros(count, count)))
    vector = (-linear).col_join(sp.Matrix(rhs) if rhs else sp.zeros(0, 1))
    unknowns = (
        *problem.variables,
        *(sp.Dummy(f"multiplier{i}", real=True) for i in range(count)),
    )
    solutions = sp.linsolve((matrix, vector), unknowns)
    if solutions is sp.EmptySet:
        return matrix, vector, (), (), None
    solution = tuple(next(iter(solutions)))
    used = set().union(*(v.free_symbols for v in solution[:n]))
    free = tuple(v for v in unknowns if v in used)
    unused = dict.fromkeys((v for v in unknowns if v not in used), sp.S.Zero)
    solution = tuple(v.xreplace(unused) for v in solution)
    if not free:
        return matrix, vector, solution, free, None
    substitution = dict(zip(problem.variables, solution[:n]))
    transformed = []
    for relation in problem.constraints:
        residual = sp.expand(
            (relation.lhs - relation.rhs).subs(substitution, simultaneous=True)
        )
        transformed.append(relation.func(residual, 0, evaluate=False))
    return matrix, vector, solution, free, affine_system(transformed, free)


def solve_face(problem, system, active):
    """Check stationary feasibility, including singular families via an exact LP."""
    matrix, vector, solution, free, slice_system = face_system(problem, system, active)
    if not solution:
        return FaceRecord(active, "no_stationary")
    if free:
        feasibility = exact_linear_minimize((sp.S.Zero,) * len(free), slice_system)
        if feasibility.status == "infeasible":
            return FaceRecord(active, "empty", proof=feasibility.proof)
        if feasibility.status != "optimal":
            return FaceRecord(active, "unknown")
        mapping = dict(zip(free, feasibility.point))
        solution = tuple(
            sp.cancel(v.subs(mapping, simultaneous=True)) for v in solution
        )
    if not system.feasible(solution[: system.size]):
        return (
            FaceRecord(active, "outside") if not free else FaceRecord(active, "unknown")
        )
    if matrix * sp.Matrix(solution) != vector:
        return FaceRecord(active, "unknown")
    return FaceRecord(
        active, "candidate", solution, problem.value(solution[: system.size])
    )


def verify_face_certificate(problem, certificate):
    """Replay exact dual bounds, face coverage and stationary feasibility evidence."""
    if (
        not isinstance(certificate, FaceCertificate)
        or certificate.problem != problem
        or not problem.rational
        or problem.integer_variables
    ):
        return False
    system = affine_system(problem.constraints, problem.variables)
    if system is None:
        return False
    if certificate.kind == "infeasible":
        return (
            certificate.complete
            and certificate.lower_bound == sp.oo
            and certificate.upper_bound == sp.oo
            and not certificate.point
            and certificate.infeasibility is not None
            and certificate.infeasibility.kind == "infeasible"
            and verify_linear_proof(
                system, (sp.S.Zero,) * system.size, 0, certificate.infeasibility
            )
        )
    if certificate.kind != "faces":
        return False
    bounds = certificate.bounds
    if (
        not isinstance(bounds, CoordinateBounds)
        or len(bounds.intervals) != system.size
        or len(bounds.proofs) != 2 * system.size
    ):
        return False
    for i, (lo, hi) in enumerate(bounds.intervals):
        if not (lo.is_Rational and hi.is_Rational and lo <= hi):
            return False
        for sign, value, proof in (
            (1, lo, bounds.proofs[2 * i]),
            (-1, -hi, bounds.proofs[2 * i + 1]),
        ):
            objective = tuple(
                sp.S(sign) if j == i else sp.S.Zero for j in range(system.size)
            )
            if not verify_linear_proof(system, objective, value, proof):
                return False
    if (
        not system.feasible(certificate.point)
        or problem.value(certificate.point) != certificate.upper_bound
    ):
        return False
    if not certificate.complete:
        return (
            certificate.lower_bound
            == interval_polynomial(
                problem.objective, problem.variables, bounds.intervals
            )[0]
        )
    expected = tuple(independent_faces(system))
    if tuple(record.active for record in certificate.faces) != expected:
        return False
    if certificate.lower_bound != certificate.upper_bound:
        return False
    found = False
    for record in certificate.faces:
        matrix, vector, solution, free, slice_system = face_system(
            problem, system, record.active
        )
        if record.kind == "no_stationary":
            if solution:
                return False
        elif record.kind == "outside":
            if not solution or free or system.feasible(solution[: system.size]):
                return False
        elif record.kind == "empty":
            if (
                not free
                or record.proof is None
                or record.proof.kind != "infeasible"
                or not verify_linear_proof(
                    slice_system, (sp.S.Zero,) * len(free), 0, record.proof
                )
            ):
                return False
        elif record.kind == "candidate":
            if (
                len(record.solution) != matrix.cols
                or any(sp.sympify(v).is_Rational is not True for v in record.solution)
                or matrix * sp.Matrix(record.solution) != vector
            ):
                return False
            point = record.solution[: system.size]
            if (
                not system.feasible(point)
                or problem.value(point) != record.value
                or record.value < certificate.lower_bound
            ):
                return False
            found |= record.value == certificate.lower_bound
        else:
            return False
    return found


def solve_faces(problem, controls):
    """Exhaust stationary faces on a bounded rational polytope, with honest limits."""
    start = monotonic()
    if not problem.rational or problem.integer_variables:
        return GlobalQuadraticResult(
            "unsupported",
            problem.variables,
            backend="faces",
            metadata={"reason": "rational continuous coefficients required"},
        )
    system = affine_system(problem.constraints, problem.variables)
    if system is None:
        return GlobalQuadraticResult(
            "unsupported",
            problem.variables,
            backend="faces",
            metadata={"reason": "closed affine constraints required"},
        )
    feasibility = exact_linear_minimize((sp.S.Zero,) * system.size, system)
    if feasibility.status == "infeasible":
        proof = FaceCertificate(
            problem,
            "infeasible",
            sp.oo,
            sp.oo,
            complete=True,
            infeasibility=feasibility.proof,
        )
        verified = verify_face_certificate(problem, proof)
        return GlobalQuadraticResult(
            "infeasible",
            problem.variables,
            lower_bound=sp.oo,
            backend="faces",
            certified=controls.certified and verified,
            bounds_verified=verified,
            attained=False,
            certificate=proof,
        )
    if feasibility.status != "optimal":
        return GlobalQuadraticResult(
            "unknown",
            problem.variables,
            backend="faces",
            metadata={"reason": "linear feasibility could not be verified"},
        )
    bounds = coordinate_bounds(system)
    if bounds is None:
        return GlobalQuadraticResult(
            "unsupported",
            problem.variables,
            backend="faces",
            point=feasibility.point,
            objective_value=problem.value(feasibility.point),
            upper_bound=problem.value(feasibility.point),
            metadata={"reason": "finite domain bounds could not be proved"},
        )
    point = feasibility.point
    best = problem.value(point)
    lower = interval_polynomial(problem.objective, problem.variables, bounds.intervals)[
        0
    ]
    records = []
    complete = True
    status = "optimal"
    stationary = False
    for active in independent_faces(system):
        if len(records) >= controls.max_faces:
            complete = False
            status = "face_limit"
            break
        if (
            controls.time_limit is not None
            and monotonic() - start >= controls.time_limit
        ):
            complete = False
            status = "time_limit"
            break
        record = solve_face(problem, system, active)
        records.append(record)
        if record.kind == "unknown":
            complete = False
            status = "unknown"
            break
        if record.kind == "candidate":
            stationary = True
            if record.value < best:
                best = record.value
                point = record.solution[: system.size]
    if complete and stationary:
        lower = best
    elif complete:
        complete = False
        status = "unknown"
    if lower == best:
        status = "optimal"
    proof = FaceCertificate(
        problem, "faces", lower, best, point, bounds, tuple(records), complete
    )
    verified = verify_face_certificate(problem, proof) if controls.certified else True
    return GlobalQuadraticResult(
        status,
        problem.variables,
        best,
        point,
        lower,
        best,
        "faces",
        len(records),
        controls.certified and verified,
        verified,
        True if status == "optimal" else None,
        proof,
        {"complete": complete, "faces": len(records)},
    )
