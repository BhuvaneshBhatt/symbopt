import numpy as np
import sympy as sp

from symbopt import (
    SDPProblem,
    build_moment_matrix,
    interval_optimize,
    solve_sdp_dense,
)
from symbopt.symbolic_cache import cache_info, clear_symbolic_caches
from symbopt.intervals import IntervalEvaluator
from symbopt.verified_intervals import enclose_expression
from symbopt.moment import monomial_basis as moment_basis
from symbopt.sos_program import (
    compile_sos_relaxation,
    monomial_basis as sos_basis,
)


def test_shared_monomial_cache():
    x, y = sp.symbols("x y")
    clear_symbolic_caches()
    assert moment_basis([x, y], 3) == sos_basis([x, y], 3)
    first = cache_info()["monomial_basis"]
    moment_basis([x, y], 3)
    second = cache_info()["monomial_basis"]
    assert second.hits > first.hits


def test_moment_matrix_powers():
    x, y = sp.symbols("x y")
    matrix = build_moment_matrix([x, y], 1)
    assert matrix.matrix[1, 2] == sp.Symbol("y_1_1", real=True)


def test_sos_direct_rows():
    x = sp.symbols("x", real=True)
    comp = compile_sos_relaxation(x**2 + 1, [x], 1)
    assert comp.sdp_problem is not None
    assert comp.sdp_problem.constraint_count >= 2
    assert comp.sdp_problem.C_blocks[0].shape == (2, 2)


def test_interval_eval_cache():
    x = sp.symbols("x", real=True)
    enclose_expression.cache_clear()
    evaluator = IntervalEvaluator.compile(sp.sin(x))
    from symbopt.problem_models import IntervalBox

    box = IntervalBox({x: (0.0, 1.0)})
    evaluator.evaluate(box)
    first = enclose_expression.cache_info()
    evaluator.evaluate(box)
    second = enclose_expression.cache_info()
    assert second.hits > first.hits


def test_interval_independent_rel():
    x, y = sp.symbols("x y", real=True)
    result = interval_optimize(
        x**2,
        [sp.Ge(y, 0)],
        {x: (-1.0, 1.0), y: (1.0, 2.0)},
        max_boxes=8,
        tol=0.1,
    )
    assert result.status == "success"


def test_sdp_workspace_result():
    objective = np.eye(2)
    rows = [
        np.array([[1.0, 0.0], [0.0, 0.0]]),
        np.array([[0.0, 0.0], [0.0, 1.0]]),
    ]
    problem = SDPProblem.single_block(
        C=objective,
        A=rows,
        b=np.array([1.0, 2.0]),
    )
    result = solve_sdp_dense(problem, max_iter=80, tol=1e-8)
    assert abs(result.primal_objective - 3.0) < 0.2
