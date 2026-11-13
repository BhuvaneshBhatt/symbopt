"""Spatial branch-and-bound with rational McCormick relaxations and proof replay."""

from dataclasses import dataclass
from heapq import heappop, heappush
from time import monotonic

import sympy as sp

from .quadratic_candidates import improve_incumbent, numerical_proposals, sampled_points
from .quadratic_faces import CoordinateBounds, coordinate_bounds, interval_polynomial
from .quadratic_lp import (
    LinearProof,
    affine_system,
    exact_linear_minimize,
    verify_linear_proof,
)
from .quadratic_models import GlobalQuadraticResult
from .rational_optimization import _compare
from .spectral_rules import _ellipsoid


@dataclass(frozen=True)
class SpatialDomain:
    """A rational search box covering all feasible points or every improving point."""

    kind: str
    intervals: tuple
    linear_bounds: CoordinateBounds | None = None
    relation: int | None = None
    witness: tuple = ()


@dataclass(frozen=True)
class SpatialNode:
    """A domain partition, its verified relaxation bound, and optional children."""

    incoming: tuple
    box: tuple
    lower: sp.Expr
    kind: str
    proof: LinearProof | None = None
    children: tuple = ()
    split: int | None = None
    midpoint: sp.Expr | None = None


@dataclass(frozen=True)
class SpatialCertificate:
    """A complete partition and exact leaf bounds, including any unprocessed leaves."""

    problem: object
    domain: SpatialDomain
    nodes: tuple
    lower_bound: sp.Expr
    upper_bound: sp.Expr
    point: tuple = ()


def linear_part(problem):
    """Extract the closed affine constraints of a quadratic model."""
    variables = problem.variables
    relations = [
        r
        for r in problem.constraints
        if sp.Poly(r.lhs - r.rhs, *variables).total_degree() <= 1
    ]
    return affine_system(relations, variables)


def ellipsoid_box(relation, variables):
    """Outward-round exact ellipsoid coordinate bounds to rational integers."""
    data = _ellipsoid(relation, variables)
    if data is None:
        return None
    matrix, center, level, _ = data
    inverse = matrix.inv()
    intervals = []
    for i in range(len(variables)):
        radius = sp.sqrt(level * inverse[i, i])
        lo, hi = sp.floor(center[i] - radius), sp.ceiling(center[i] + radius)
        if not (lo.is_Integer and hi.is_Integer):
            return None
        intervals.append((lo, hi))
    return tuple(intervals)


def spatial_domain(problem, incumbent):
    """Prove a finite domain, or a safe objective-sublevel reduction with a witness."""
    system = linear_part(problem)
    bounds = coordinate_bounds(system) if system is not None else None
    if bounds is not None:
        return SpatialDomain("linear", bounds.intervals, bounds)
    for i, relation in enumerate(problem.constraints):
        bounds = ellipsoid_box(relation, problem.variables)
        if bounds is not None:
            return SpatialDomain("ellipsoid", bounds, relation=i)
    if incumbent:
        bounds = ellipsoid_box(
            sp.Le(problem.objective, problem.value(incumbent), evaluate=False),
            problem.variables,
        )
        if bounds is not None:
            return SpatialDomain("sublevel", bounds, witness=incumbent)
    return None


def verify_domain(problem, domain):
    """Replay the finite search-domain reduction on the original quadratic problem."""
    if not isinstance(domain, SpatialDomain) or len(domain.intervals) != len(
        problem.variables
    ):
        return False
    if any(
        len(pair) != 2
        or any(sp.sympify(v).is_Rational is not True for v in pair)
        or pair[0] > pair[1]
        for pair in domain.intervals
    ):
        return False
    if domain.kind == "linear":
        bounds = domain.linear_bounds
        if (
            not isinstance(bounds, CoordinateBounds)
            or bounds.intervals != domain.intervals
            or len(bounds.proofs) != 2 * len(problem.variables)
        ):
            return False
        system = linear_part(problem)
        if system is None:
            return False
        for i, (lo, hi) in enumerate(bounds.intervals):
            for sign, value, proof in (
                (1, lo, bounds.proofs[2 * i]),
                (-1, -hi, bounds.proofs[2 * i + 1]),
            ):
                objective = tuple(
                    sp.S(sign) if j == i else sp.S.Zero for j in range(system.size)
                )
                if not verify_linear_proof(system, objective, value, proof):
                    return False
        return True
    if domain.kind == "ellipsoid":
        if not isinstance(domain.relation, int) or not 0 <= domain.relation < len(
            problem.constraints
        ):
            return False
        expected = ellipsoid_box(
            problem.constraints[domain.relation], problem.variables
        )
    elif domain.kind == "sublevel":
        if not problem.feasible(domain.witness):
            return False
        expected = ellipsoid_box(
            sp.Le(problem.objective, problem.value(domain.witness), evaluate=False),
            problem.variables,
        )
    else:
        return False
    return expected == domain.intervals


def propagate(problem, box):
    """Tighten rational bounds by necessary affine row implications."""
    system = linear_part(problem)
    values = [list(pair) for pair in box]
    rows = [
        *zip(system.inequalities, system.lower),
        *zip(system.equalities, system.rhs),
        *(
            (tuple(-a for a in row), -b)
            for row, b in zip(system.equalities, system.rhs)
        ),
    ]
    for _ in range(2):
        for row, rhs in rows:
            for i, a in enumerate(row):
                if a == 0:
                    continue
                maximum = sum(
                    c * (values[j][1] if c >= 0 else values[j][0])
                    for j, c in enumerate(row)
                    if j != i
                )
                endpoint = (rhs - maximum) / a
                if a > 0:
                    values[i][0] = max(values[i][0], endpoint)
                else:
                    values[i][1] = min(values[i][1], endpoint)
                if values[i][0] > values[i][1]:
                    return None
    return tuple(tuple(pair) for pair in values)


def relaxation(problem, box):
    """Build a linear outer approximation using shared square/product coordinates."""
    variables = problem.variables
    pairs = sorted(
        {
            tuple(i for i, power in enumerate(powers) for _ in range(power))
            for expression in problem.expressions
            for powers, _ in sp.Poly(expression, *variables).terms()
            if sum(powers) == 2
        }
    )
    products = {
        pair: sp.Dummy(f"product{k}", real=True) for k, pair in enumerate(pairs)
    }

    def linearize(expression):
        result = sp.S.Zero
        for powers, coefficient in sp.Poly(expression, *variables).terms():
            indices = tuple(i for i, power in enumerate(powers) for _ in range(power))
            result += coefficient * (
                products[indices]
                if len(indices) == 2
                else variables[indices[0]]
                if indices
                else 1
            )
        return result

    relations = []
    for relation in problem.constraints:
        relations.append(
            relation.func(linearize(relation.lhs - relation.rhs), 0, evaluate=False)
        )
    for variable, (lo, hi) in zip(variables, box):
        relations.extend(
            (sp.Ge(variable, lo, evaluate=False), sp.Le(variable, hi, evaluate=False))
        )
    for (i, j), product_variable in products.items():
        x, y = variables[i], variables[j]
        lo, hi = box[i]
        left, right = box[j]
        relations.extend(
            (
                sp.Ge(product_variable, lo * y + left * x - lo * left, evaluate=False),
                sp.Ge(
                    product_variable, hi * y + right * x - hi * right, evaluate=False
                ),
                sp.Le(product_variable, hi * y + left * x - hi * left, evaluate=False),
                sp.Le(
                    product_variable, lo * y + right * x - lo * right, evaluate=False
                ),
            )
        )
        if i == j:
            midpoint = (lo + hi) / 2
            relations.append(
                sp.Ge(product_variable, 2 * midpoint * x - midpoint**2, evaluate=False)
            )
            relations.append(
                sp.Ge(
                    product_variable,
                    0 if lo <= 0 <= hi else min(lo * lo, hi * hi),
                    evaluate=False,
                )
            )
    all_variables = (*variables, *products.values())
    system = affine_system(relations, all_variables)
    expression = linearize(problem.objective)
    coefficients = tuple(expression.coeff(v) for v in all_variables)
    constant = expression.subs(dict.fromkeys(all_variables, sp.S.Zero))
    return system, coefficients, constant, pairs


def interval_infeasible(problem, box):
    """Prove that a rational box violates at least one original quadratic constraint."""
    for relation in problem.constraints:
        lo, hi = interval_polynomial(
            relation.lhs - relation.rhs, problem.variables, box
        )
        if isinstance(relation, sp.Equality) and not lo <= 0 <= hi:
            return True
        if isinstance(relation, sp.GreaterThan) and hi < 0:
            return True
        if isinstance(relation, sp.LessThan) and lo > 0:
            return True
    return False


def node_bound(problem, box):
    """Compute only bounds backed by rational dual/Farkas or interval evidence."""
    if interval_infeasible(problem, box):
        return sp.oo, "interval_infeasible", None, (), ()
    system, objective, constant, pairs = relaxation(problem, box)
    result = exact_linear_minimize(objective, system)
    if result.status == "infeasible":
        return sp.oo, "lp_infeasible", result.proof, (), pairs
    if result.status == "optimal":
        return result.value + constant, "lp", result.proof, result.point, pairs
    return (
        interval_polynomial(problem.objective, problem.variables, box)[0],
        "interval",
        None,
        (),
        pairs,
    )


def verify_spatial_certificate(problem, certificate):
    """Replay every partition and exact leaf bound without trusting solver statuses."""
    if (
        not isinstance(certificate, SpatialCertificate)
        or certificate.problem != problem
        or not problem.rational
        or problem.integer_variables
    ):
        return False
    if not verify_domain(problem, certificate.domain) or not certificate.nodes:
        return False
    if certificate.point:
        if (
            not problem.feasible(certificate.point)
            or problem.value(certificate.point) != certificate.upper_bound
        ):
            return False
    elif certificate.upper_bound != sp.oo:
        return False
    nodes = certificate.nodes
    stack = [(0, certificate.domain.intervals, None)]
    seen = set()
    leaves = []
    while stack:
        index, incoming, parent_bound = stack.pop()
        if not isinstance(index, int) or not 0 <= index < len(nodes) or index in seen:
            return False
        seen.add(index)
        node = nodes[index]
        if node.incoming != incoming:
            return False
        if node.kind == "pending":
            expected = (
                interval_polynomial(problem.objective, problem.variables, incoming)[0]
                if parent_bound is None
                else parent_bound
            )
            if node.box != incoming or node.lower != expected or node.children:
                return False
            leaves.append(node.lower)
            continue
        effective = propagate(problem, incoming)
        if node.kind == "propagation_infeasible":
            if effective is not None or node.lower != sp.oo or node.children:
                return False
            leaves.append(sp.oo)
            continue
        if effective != node.box:
            return False
        if node.kind == "interval_infeasible":
            if (
                node.children
                or not interval_infeasible(problem, effective)
                or node.lower != sp.oo
            ):
                return False
        elif node.kind == "interval":
            if (
                node.lower
                != interval_polynomial(problem.objective, problem.variables, effective)[
                    0
                ]
            ):
                return False
        elif node.kind in {"lp", "lp_infeasible"}:
            system, objective, constant, _ = relaxation(problem, effective)
            if node.kind == "lp":
                if (
                    node.proof is None
                    or node.proof.kind != "bound"
                    or not verify_linear_proof(
                        system, objective, node.lower - constant, node.proof
                    )
                ):
                    return False
            elif (
                node.children
                or node.proof is None
                or node.proof.kind != "infeasible"
                or node.lower != sp.oo
                or not verify_linear_proof(
                    system, (sp.S.Zero,) * system.size, 0, node.proof
                )
            ):
                return False
        else:
            return False
        if node.children:
            i, midpoint = node.split, node.midpoint
            if (
                not isinstance(i, int)
                or not 0 <= i < len(effective)
                or sp.sympify(midpoint).is_Rational is not True
                or not effective[i][0] < midpoint < effective[i][1]
                or len(node.children) != 2
            ):
                return False
            first = list(effective)
            second = list(effective)
            first[i] = (effective[i][0], midpoint)
            second[i] = (midpoint, effective[i][1])
            stack.extend(
                (
                    (node.children[0], tuple(first), node.lower),
                    (node.children[1], tuple(second), node.lower),
                )
            )
        else:
            leaves.append(node.lower)
    if len(seen) != len(nodes):
        return False
    lower = min([certificate.upper_bound, *leaves])
    return lower == certificate.lower_bound


def solve_spatial(problem, controls, initial_point=()):
    """Globally bound a rational closed quadratic problem using a finite partition.

    Limits are checked between exact subproblems. A positive verified gap is a
    bound certificate, not an exact optimality claim. Finite boxes must follow
    from constraints or a checked coercive objective sublevel.
    """
    start = monotonic()
    if (
        not problem.rational
        or problem.integer_variables
        or any(
            isinstance(r, (sp.StrictGreaterThan, sp.StrictLessThan, sp.Unequality))
            for r in problem.constraints
        )
    ):
        return GlobalQuadraticResult(
            "unsupported",
            problem.variables,
            backend="spatial",
            metadata={"reason": "closed rational continuous quadratic models required"},
        )
    point = improve_incumbent(problem, [initial_point] if initial_point else [])
    point = improve_incumbent(problem, sampled_points(problem), point)
    if not point:
        point = improve_incumbent(problem, numerical_proposals(problem), point)
    domain = spatial_domain(problem, point)
    if domain is None:
        return GlobalQuadraticResult(
            "unsupported",
            problem.variables,
            problem.value(point) if point else None,
            point,
            upper_bound=problem.value(point) if point else sp.oo,
            backend="spatial",
            metadata={"reason": "no finite global domain reduction was proved"},
        )
    point = improve_incumbent(problem, sampled_points(problem, domain.intervals), point)
    seeds = [point] if point else []
    for i, (lo, hi) in enumerate(domain.intervals):
        seeds.append(
            tuple(
                hi if j == i else (a + b) / 2
                for j, (a, b) in enumerate(domain.intervals)
            )
        )
    for seed in seeds[:4]:
        point = improve_incumbent(
            problem, numerical_proposals(problem, domain.intervals, seed), point
        )
    upper = problem.value(point) if point else sp.oo
    lower = interval_polynomial(problem.objective, problem.variables, domain.intervals)[
        0
    ]
    nodes = [SpatialNode(domain.intervals, domain.intervals, lower, "pending")]
    heap = [(lower, 0)]
    processed = 0
    status = "node_limit"
    while heap:
        lower, index = heappop(heap)
        if upper != sp.oo:
            gap = upper - lower
            if (
                _compare(
                    gap, controls.absolute_gap + controls.relative_gap * abs(upper)
                )
                <= 0
            ):
                heappush(heap, (lower, index))
                status = "optimal" if gap == 0 else "gap_limit"
                break
        if processed >= controls.max_nodes:
            heappush(heap, (lower, index))
            status = "node_limit"
            break
        if (
            controls.time_limit is not None
            and monotonic() - start >= controls.time_limit
        ):
            heappush(heap, (lower, index))
            status = "time_limit"
            break
        incoming = nodes[index].incoming
        box = propagate(problem, incoming)
        processed += 1
        if box is None:
            nodes[index] = SpatialNode(
                incoming, incoming, sp.oo, "propagation_infeasible"
            )
            continue
        bound, kind, proof, relaxed, pairs = node_bound(problem, box)
        nodes[index] = SpatialNode(incoming, box, bound, kind, proof)
        if bound == sp.oo:
            continue
        if relaxed:
            point = improve_incumbent(
                problem, [relaxed[: len(problem.variables)]], point
            )
            upper = problem.value(point) if point else sp.oo
        if upper != sp.oo and _compare(bound, upper) >= 0:
            continue
        widths = [b - a for a, b in box]
        if max(widths) == 0:
            continue
        if relaxed and pairs:
            violations = [
                (
                    abs(relaxed[len(problem.variables) + k] - relaxed[i] * relaxed[j]),
                    i,
                    j,
                )
                for k, (i, j) in enumerate(pairs)
            ]
            _, i, j = max(violations)
            split = i if widths[i] >= widths[j] else j
            if widths[split] == 0:
                split = max(range(len(widths)), key=widths.__getitem__)
        else:
            split = max(range(len(widths)), key=widths.__getitem__)
        midpoint = sum(box[split]) / 2
        first = list(box)
        second = list(box)
        first[split] = (box[split][0], midpoint)
        second[split] = (midpoint, box[split][1])
        children = (len(nodes), len(nodes) + 1)
        nodes[index] = SpatialNode(
            incoming, box, bound, kind, proof, children, split, midpoint
        )
        for child, child_box in zip(children, (tuple(first), tuple(second))):
            nodes.append(SpatialNode(child_box, child_box, bound, "pending"))
            heappush(heap, (bound, child))
    else:
        status = "infeasible" if not point else "optimal"
    leaves = [node.lower for node in nodes if not node.children]
    lower = min([upper, *leaves])
    if lower == upper and point:
        status = "optimal"
    elif status == "optimal":
        status = "gap_limit" if point else "infeasible"
    certificate = SpatialCertificate(problem, domain, tuple(nodes), lower, upper, point)
    verified = (
        verify_spatial_certificate(problem, certificate) if controls.certified else True
    )
    return GlobalQuadraticResult(
        status,
        problem.variables,
        problem.value(point) if point else None,
        point,
        lower,
        upper,
        "spatial",
        processed,
        controls.certified and verified,
        verified,
        True
        if status == "optimal" and point
        else False
        if status == "infeasible"
        else None,
        certificate,
        {"complete_partition": True, "remaining_nodes": len(heap)},
    )
