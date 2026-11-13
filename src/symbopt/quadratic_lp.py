"""Rational LP subproblems with independently checked dual and Farkas witnesses."""

from dataclasses import dataclass

import sympy as sp
from sympy.solvers.simplex import InfeasibleLPError, UnboundedLPError, lpmin


@dataclass(frozen=True)
class LinearSystem:
    """Rational rows A*x >= b and E*x = h, with free real coordinates."""

    size: int
    inequalities: tuple = ()
    lower: tuple = ()
    equalities: tuple = ()
    rhs: tuple = ()

    def feasible(self, point):
        """Check exact feasibility of an ordered coordinate tuple in the stored system."""
        return (
            len(point) == self.size
            and all(sp.sympify(v).is_Rational for v in point)
            and all(
                sum(a * x for a, x in zip(row, point)) >= b
                for row, b in zip(self.inequalities, self.lower)
            )
            and all(
                sum(a * x for a, x in zip(row, point)) == b
                for row, b in zip(self.equalities, self.rhs)
            )
        )


@dataclass(frozen=True)
class LinearProof:
    """Nonnegative inequality and unrestricted equality weights for a bound or contradiction."""

    kind: str
    inequalities: tuple
    equalities: tuple


@dataclass(frozen=True)
class LinearResult:
    """An LP result whose decisive statuses require checked exact evidence."""

    status: str
    value: sp.Expr | None = None
    point: tuple = ()
    proof: LinearProof | None = None


def verify_linear_proof(system, objective, value, proof):
    """Verify weak duality or a Farkas contradiction using rational arithmetic only."""
    if not isinstance(proof, LinearProof):
        return False
    weights, free = proof.inequalities, proof.equalities
    if len(weights) != len(system.inequalities) or len(free) != len(system.equalities):
        return False
    if any(sp.sympify(v).is_Rational is not True for v in (*weights, *free)):
        return False
    if any(w < 0 for w in weights):
        return False
    combined = tuple(
        sum(w * row[i] for w, row in zip(weights, system.inequalities))
        + sum(w * row[i] for w, row in zip(free, system.equalities))
        for i in range(system.size)
    )
    constant = sum(w * b for w, b in zip(weights, system.lower)) + sum(
        w * b for w, b in zip(free, system.rhs)
    )
    if proof.kind == "infeasible":
        return all(v == 0 for v in combined) and constant > 0
    return proof.kind == "bound" and combined == tuple(objective) and constant == value


def _conditions(system, variables):
    return [
        sp.Ge(sum(a * x for a, x in zip(row, variables)), b)
        for row, b in zip(system.inequalities, system.lower)
    ] + [
        sp.Eq(sum(a * x for a, x in zip(row, variables)), b)
        for row, b in zip(system.equalities, system.rhs)
    ]


def _weights(system, objective, value, kind):
    """Find a dual/Farkas witness; a simplex claim alone never proves a status."""
    count = len(system.inequalities)
    variables = tuple(
        sp.Dummy(f"weight{i}", real=True) for i in range(count + len(system.equalities))
    )
    if not variables:
        return None
    weights, free = variables[:count], variables[count:]
    columns = [
        sum(w * row[i] for w, row in zip(weights, system.inequalities))
        + sum(w * row[i] for w, row in zip(free, system.equalities))
        for i in range(system.size)
    ]
    constant = sum(w * b for w, b in zip(weights, system.lower)) + sum(
        w * b for w, b in zip(free, system.rhs)
    )
    conditions = [sp.Ge(w, 0) for w in weights]
    conditions += [sp.Eq(a, b) for a, b in zip(columns, objective)]
    conditions += (
        [sp.Ge(constant, 1)] if kind == "infeasible" else [sp.Eq(constant, value)]
    )
    if any(c is sp.false for c in conditions):
        return None
    try:
        _, solution = lpmin(0, [c for c in conditions if c is not sp.true])
    except (InfeasibleLPError, UnboundedLPError, ValueError, TypeError):
        return None
    proof = LinearProof(
        kind,
        tuple(solution.get(v, sp.S.Zero) for v in weights),
        tuple(solution.get(v, sp.S.Zero) for v in free),
    )
    return proof if verify_linear_proof(system, objective, value, proof) else None


def exact_linear_minimize(objective, system):
    """Minimize a rational affine form, checking primal and dual evidence separately.

    Infeasibility is accepted only with a verified Farkas certificate. An LP
    backend exception or an unchecked unboundedness claim returns unknown.
    """
    objective = tuple(map(sp.sympify, objective))
    if len(objective) != system.size:
        raise ValueError("LP objective dimension mismatch")
    if system.size >= 3:
        proposed = _linear_proposal(objective, system)
        if proposed is not None:
            return proposed
    variables = tuple(sp.Dummy(f"coordinate{i}", real=True) for i in range(system.size))
    conditions = [c for c in _conditions(system, variables) if c is not sp.true]
    try:
        value, solution = lpmin(
            sum(c * x for c, x in zip(objective, variables)), conditions
        )
    except InfeasibleLPError:
        proof = _weights(system, (sp.S.Zero,) * system.size, sp.S.Zero, "infeasible")
        return (
            LinearResult("infeasible", proof=proof)
            if proof is not None
            else LinearResult("unknown")
        )
    except (UnboundedLPError, ValueError, TypeError):
        return LinearResult("unknown")
    point = tuple(solution.get(v, sp.S.Zero) for v in variables)
    if (
        not system.feasible(point)
        or sum(c * x for c, x in zip(objective, point)) != value
    ):
        return LinearResult("unknown")
    if all(c == 0 for c in objective):
        proof = LinearProof(
            "bound",
            (sp.S.Zero,) * len(system.inequalities),
            (sp.S.Zero,) * len(system.equalities),
        )
    else:
        proof = _weights(system, objective, value, "bound")
    if proof is None or not verify_linear_proof(system, objective, value, proof):
        return LinearResult("unknown")
    return LinearResult("optimal", value, point, proof)


def _linear_proposal(objective, system):
    """Reconstruct a numerical LP basis and accept only matching rational evidence.

    HiGHS supplies a useful basis proposal for product relaxations. Its floating
    status is never used as proof: exact primal feasibility and a reconstructed
    nonnegative dual witness must agree on the same objective value.
    """
    import numpy as np
    from scipy.optimize import linprog

    try:
        result = linprog(
            np.asarray(objective, dtype=float),
            A_ub=-np.asarray(system.inequalities, dtype=float)
            if system.inequalities
            else None,
            b_ub=-np.asarray(system.lower, dtype=float)
            if system.inequalities
            else None,
            A_eq=np.asarray(system.equalities, dtype=float)
            if system.equalities
            else None,
            b_eq=np.asarray(system.rhs, dtype=float) if system.equalities else None,
            bounds=[(None, None)] * system.size,
            method="highs",
        )
    except (ValueError, OverflowError):
        return None
    if not result.success:
        return None
    active = [i for i, value in enumerate(result.ineqlin.marginals) if value < -1e-9]
    columns = [sp.Matrix(system.inequalities[i]) for i in active]
    columns.extend(sp.Matrix(row) for row in system.equalities)
    if columns:
        solutions = sp.linsolve((sp.Matrix.hstack(*columns), sp.Matrix(objective)))
        if solutions is sp.EmptySet:
            return None
        solution = next(iter(solutions))
        free = set().union(*(v.free_symbols for v in solution))
        weights = tuple(v.subs(dict.fromkeys(free, sp.S.Zero)) for v in solution)
    elif any(objective):
        return None
    else:
        weights = ()
    inequalities = [sp.S.Zero] * len(system.inequalities)
    for i, w in zip(active, weights):
        inequalities[i] = w
    proof = LinearProof("bound", tuple(inequalities), weights[len(active) :])
    value = sum(w * b for w, b in zip(proof.inequalities, system.lower)) + sum(
        w * b for w, b in zip(proof.equalities, system.rhs)
    )
    if not verify_linear_proof(system, objective, value, proof):
        return None
    # Reconstruct the primal from all numerically active rows, with an exact
    # rational approximation used only to select any remaining free coordinates.
    rows = [
        row
        for row, slack in zip(system.inequalities, result.ineqlin.residual)
        if abs(slack) <= 1e-8
    ]
    rhs = [
        b
        for b, slack in zip(system.lower, result.ineqlin.residual)
        if abs(slack) <= 1e-8
    ]
    rows.extend(system.equalities)
    rhs.extend(system.rhs)
    variables = sp.symbols(f"p0:{system.size}")
    if rows:
        solutions = sp.linsolve((sp.Matrix(rows), sp.Matrix(rhs)), variables)
        if solutions is sp.EmptySet:
            return None
        proposed = next(iter(solutions))
    else:
        proposed = variables
    mapping = {
        v: sp.Rational(float(x)).limit_denominator(10**6)
        for v, x in zip(variables, result.x)
    }
    point = tuple(v.subs(mapping) for v in proposed)
    if (
        not system.feasible(point)
        or sum(a * x for a, x in zip(objective, point)) != value
    ):
        return None
    return LinearResult("optimal", value, point, proof)


def affine_system(constraints, variables):
    """Extract closed rational affine relations without changing their domains."""
    from .symbolic_cache import affine_coefficients

    inequalities, lower, equalities, rhs = [], [], [], []
    for relation in constraints:
        if isinstance(
            relation, (sp.StrictGreaterThan, sp.StrictLessThan, sp.Unequality)
        ):
            return None
        data = affine_coefficients(relation.lhs - relation.rhs, tuple(variables))
        if data is None:
            return None
        coefficients, constant = data
        if any(c.is_Rational is not True for c in (*coefficients, constant)):
            return None
        if isinstance(relation, sp.Equality):
            equalities.append(coefficients)
            rhs.append(-constant)
        elif isinstance(relation, (sp.GreaterThan, sp.LessThan)):
            sign = -1 if isinstance(relation, sp.LessThan) else 1
            inequalities.append(tuple(sign * c for c in coefficients))
            lower.append(-sign * constant)
        else:
            return None
    return LinearSystem(
        len(variables), tuple(inequalities), tuple(lower), tuple(equalities), tuple(rhs)
    )
