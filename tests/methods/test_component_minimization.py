from dataclasses import replace

import pytest
import sympy as sp

from symbopt import certify_global_minimum, finite_component_minimum

fp = pytest.importorskip("funcprops")

x = sp.Symbol("x", real=True)


@pytest.mark.parametrize(
    "f,D,value,points",
    [
        (x * x, sp.Union(sp.Interval(-3, -2), sp.Interval(1, 2)), 1, sp.FiniteSet(1)),
        (x * x, sp.Interval.open(1, 2), 1, sp.S.EmptySet),
        (x * x + 1 / x**2, sp.S.Reals, 2, sp.FiniteSet(-1, 1)),
        (1 / x, sp.S.Reals, -sp.oo, sp.S.EmptySet),
        (sp.Integer(3), sp.Interval.open(0, 1), 3, sp.Interval.open(0, 1)),
    ],
)
def test_component_values(f, D, value, points):
    result = finite_component_minimum(f, x, D, certified=True)
    assert result.complete and result.verify() and result.certified
    assert result.infimum == value and result.minimizers == points
    assert not finite_component_minimum(f, x, D).certified


def test_source_hole():
    raw = sp.Mul(x, sp.Pow(x, -1, evaluate=False), evaluate=False)
    result = finite_component_minimum(raw, x, certified=True)
    assert result.infimum == 1 and result.minimizers == sp.S.Reals - sp.FiniteSet(0)
    assert result.verify()
    assert not replace(result, infimum=-1).verify()
    assert not replace(result, minimizers=sp.S.Reals).verify()


def test_global_certificate():
    proof = certify_global_minimum(x * x, x, sp.Interval(-1, 1))
    assert proof.value == 0 and proof.point == 0 and proof.verify()
    assert not replace(proof, value=-1).verify()
    assert not replace(proof, point=1).verify()
    assert certify_global_minimum(x * x, x, sp.Interval.open(1, 2)) is None
