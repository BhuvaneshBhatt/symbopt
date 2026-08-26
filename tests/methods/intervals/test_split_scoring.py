import sympy as sp

from symbopt.intervals import BoxState, IntervalEvaluator, _split_choice
from symbopt.problem_models import IntervalBox
from symbopt.verified_intervals import EvalConfig, compile_arb


def test_preconditioning_preserves_original_domain():
    x = sp.symbols("x", real=True)
    expr = (x**2 - 1) / (x - 1)
    evaluator = IntervalEvaluator.compile(expr)
    bound = evaluator.evaluate(IntervalBox({x: (0.0, 2.0)}))
    assert bound.rigorous
    assert bound.domain == "uncertain"


def test_split_scoring_prefers_high_influence_variable():
    x, y = sp.symbols("x y", real=True)
    config = EvalConfig(use_taylor=False)
    objective = IntervalEvaluator.compile(100 * x + y, config)
    box = IntervalBox({x: (0.0, 0.5), y: (0.0, 1.0)})
    state = BoxState(
        box,
        objective.evaluate(box),
        (),
    )
    derivs = {
        x: IntervalEvaluator.compile(sp.Integer(100), config),
        y: IntervalEvaluator.compile(sp.Integer(1), config),
    }
    variable, _ = _split_choice(
        state,
        [100 * x + y],
        derivs,
        {x: (), y: ()},
        [],
    )
    assert variable == x


def test_broader_arb_compiler_map():
    x = sp.symbols("x", real=True)
    expressions = [
        sp.sec(x),
        sp.erfi(x),
        sp.Ei(x),
        sp.Si(x),
        sp.loggamma(x),
        sp.zeta(x),
        sp.LambertW(x),
        sp.polylog(2, x),
        sp.besselj(0, x),
        sp.airyai(x),
        sp.fresnelc(x),
        sp.sinc(x),
        sp.polygamma(0, x),
    ]
    for expr in expressions:
        assert compile_arb(expr) is not None
