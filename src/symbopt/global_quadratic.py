"""Global quadratic optimization over exact or numerical coefficient data."""

from dataclasses import replace

import sympy as sp

from .quadratic_faces import FaceCertificate, solve_faces, verify_face_certificate
from .quadratic_geometry import (
    EqualityCertificate,
    solve_equalities,
    verify_equality_certificate,
)
from .quadratic_models import (
    GlobalQuadraticResult,
    quadratic_controls,
    quadratic_problem,
)
from .quadratic_scip import scip_available, solve_scip
from .quadratic_spatial import (
    SpatialCertificate,
    solve_spatial,
    verify_spatial_certificate,
)


def verify_quadratic_certificate(
    objective,
    certificate,
    constraints=None,
    variables=None,
    *,
    bounds=None,
    region=None,
    use_symbol_assumptions=True,
):
    """Replay a native quadratic certificate against the original problem.

    Model coefficients, domain coverage, primal feasibility and lower-bound
    evidence are checked independently of the search result's status. Numerical
    SCIP results have no exact certificate and return False.
    """
    try:
        problem = quadratic_problem(
            objective,
            constraints,
            variables,
            bounds,
            use_symbol_assumptions=use_symbol_assumptions,
            region=region,
        )
        if isinstance(certificate, FaceCertificate):
            return bool(verify_face_certificate(problem, certificate))
        if isinstance(certificate, EqualityCertificate):
            return bool(verify_equality_certificate(problem, certificate))
        if isinstance(certificate, SpatialCertificate):
            return bool(verify_spatial_certificate(problem, certificate))
    except (
        ValueError,
        TypeError,
        IndexError,
        AttributeError,
        KeyError,
        ArithmeticError,
        NotImplementedError,
    ):
        return False
    return False


def minimize_quadratic(
    objective,
    constraints=None,
    variables=None,
    *,
    bounds=None,
    region=None,
    backend="auto",
    max_faces=4096,
    max_nodes=512,
    time_limit=None,
    absolute_gap=1e-8,
    relative_gap=1e-8,
    certified=False,
    initial_point=None,
    use_symbol_assumptions=True,
):
    """Globally minimize a fixed-coefficient quadratic with quadratic constraints.

    ``faces`` exhausts stationary faces of a bounded rational polytope. ``spatial``
    uses rational McCormick relaxations on a proved finite domain. ``scip`` uses
    the optional PySCIPOpt backend, including integer decision symbols, and
    reports numerical bounds without exact certification. ``auto`` first tries
    exact equality/ray and face methods, then spatial search for rational
    continuous models, or SCIP for other fixed real data.

    ``bounds`` specifies actual domain constraints, with free coordinates by
    default. Native solvers require closed rational continuous models; strict
    constraints and excluded boundaries return unsupported natively. SCIP also
    accepts strict rational lattice rows over integer variables. Search limits
    preserve partial bounds. A positive gap is not an exact optimum. Set
    ``certified=True`` to replay native proof evidence before returning.
    ``region`` intersects geometric membership with the other constraints.
    """
    controls = quadratic_controls(
        backend=backend,
        max_faces=max_faces,
        max_nodes=max_nodes,
        time_limit=time_limit,
        absolute_gap=absolute_gap,
        relative_gap=relative_gap,
        certified=certified,
    )
    problem = quadratic_problem(
        objective,
        constraints,
        variables,
        bounds,
        use_symbol_assumptions=use_symbol_assumptions,
        region=region,
    )
    if backend == "off":
        return GlobalQuadraticResult("disabled", problem.variables)
    if problem.integer_variables and backend != "scip":
        raise NotImplementedError(
            "Integer quadratic optimization requires backend='scip'"
        )
    if initial_point is None:
        initial = ()
    elif isinstance(initial_point, dict):
        initial = tuple(sp.sympify(initial_point[v]) for v in problem.variables)
    else:
        initial = tuple(map(sp.sympify, initial_point))
    if initial and not problem.feasible(initial):
        raise ValueError("initial_point must satisfy the original constraints exactly")
    if backend == "scip":
        return solve_scip(problem, controls)
    if backend == "spatial":
        return solve_spatial(problem, controls, initial)
    exact = solve_equalities(problem, controls)
    if exact is not None:
        return exact
    faces = solve_faces(problem, controls)
    if backend == "faces" or faces.exact:
        return faces
    if not problem.rational:
        return solve_scip(problem, controls)
    spatial = solve_spatial(problem, controls, initial or faces.point)
    if spatial.status != "unsupported":
        return spatial
    if faces.bounds_verified or faces.point:
        return faces
    if not certified and scip_available():
        return solve_scip(problem, controls)
    return replace(spatial, metadata={**spatial.metadata, "faces_status": faces.status})


def solve_global_qp(
    Q, c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, **options
):
    """Minimize ``x.T*Q*x/2 + c.T*x`` globally with affine matrix constraints.

    Inequalities use ``A_ub*x <= b_ub``; equalities use ``A_eq*x = b_eq``.
    Coordinates are free unless ``bounds`` supplies endpoints. Exact inputs
    retain their rational coefficients. See :func:`minimize_quadratic` for
    backends, proof replay and finite search controls.
    """
    matrix, linear = sp.Matrix(Q), sp.Matrix(c)
    n = matrix.rows
    if n == 0 or matrix.cols != n or linear.shape != (n, 1):
        raise ValueError("Q must be square and c must be a matching vector")
    variables = sp.symbols(f"x0:{n}", real=True)
    vector = sp.Matrix(variables)
    objective = (vector.T * matrix * vector)[0] / 2 + (linear.T * vector)[0]
    constraints = []
    for coefficients, values, relation in ((A_ub, b_ub, sp.Le), (A_eq, b_eq, sp.Eq)):
        if coefficients is None and values is None:
            continue
        if coefficients is None or values is None:
            raise ValueError("Each constraint matrix requires its right-hand side")
        rows, rhs = sp.Matrix(coefficients), sp.Matrix(values)
        if rows.cols != n or rhs.shape != (rows.rows, 1):
            raise ValueError("Constraint matrix and vector dimensions must match Q")
        constraints.extend(
            relation((rows * vector)[i], rhs[i], evaluate=False)
            for i in range(rows.rows)
        )
    return minimize_quadratic(
        objective, constraints, variables, bounds=bounds, **options
    )
