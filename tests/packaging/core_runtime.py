"""Exercise an installed core wheel with no optional symbolic backends."""

import importlib.util
import json
import sys
from dataclasses import replace

import sympy as sp

import symbopt as so


def main():
    """Check mathematical results and report import provenance in a core environment."""
    assert importlib.util.find_spec("funcprops") is None
    assert importlib.util.find_spec("semialg") is None
    x, y = sp.symbols("x y", real=True)
    assert so.minimize((x - 2) ** 2, variables=x) == [0, {x: 2}]
    assert so.minval(x**4 - 2 * x * x, variables=x) == -1
    assert so.argmax(-x * x, variables=x) == ({x: 0},)
    assert so.minimize(-x, region=sp.Interval(0, 1), variables=x) == [-1, {x: 1}]
    assert so.minval(x, region=sp.S.EmptySet, variables=x) == sp.oo
    assert "semialg" not in sys.modules
    lp = so.revised_simplex([-4, 0], [[2, -4], [3, -1], [0, 1]], [2, 5, 6])
    assert abs(lp.objective + 44 / 3) < 1e-12
    assert max(abs(a - b) for a, b in zip(lp.x, (11 / 3, 6))) < 1e-12
    polynomial = (x - 1) ** 2
    certificate = so.sos_decompose(polynomial, (x,)).certificate
    assert certificate.verify(polynomial=polynomial, lower_bound=0)
    negative = replace(
        certificate,
        polynomial=-polynomial,
        gram_matrices=[-matrix for matrix in certificate.gram_matrices],
    )
    assert not negative.verified
    interval = so.interval_optimize(
        x - sp.Rational(1, 10), (), {x: (sp.Rational(1, 10), 1)}, max_boxes=1
    )
    assert interval.lower_bound_rigorous and sp.Rational(interval.best_lower_bound) <= 0
    quadratic = so.minimize_quadratic(
        x * x - y * y, variables=(x, y), bounds={x: (-1, 1), y: (-1, 1)}, certified=True
    )
    assert quadratic.certified and quadratic.objective_value == -1
    unresolved = so.minimize_result(
        sp.cos(sp.I * sp.log(x), evaluate=False),
        [x >= -2, x <= -sp.Rational(1, 2)],
        variables=x,
        max_active_sets=0,
        use_sos=False,
    )
    assert not unresolved.infeasible
    assert not unresolved.certified or unresolved.optimum_value == -sp.Rational(5, 4)
    print(
        json.dumps(
            {
                "package": so.__file__,
                "optional_backends": False,
                "checks": "compact API, polynomial extrema, simplex, SOS replay, exact boxes, quadratic proofs, real-output refusal",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
