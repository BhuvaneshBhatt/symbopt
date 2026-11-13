"""Completeness and mathematical integrity of the executable reference corpus."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import sympy as sp

from .corpus.checks import decode_case, load_cases

ROOT = Path(__file__).with_name("corpus")
CASES = load_cases()


def test_reference_inventory_complete():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    identities = [case["case_id"] for case in CASES]
    assert len(identities) == len(set(identities)) == manifest["case_count"] == 779
    assert Counter(case["category"] for case in CASES) == manifest["categories"]
    signature = hashlib.sha256(
        ("\n".join(sorted(identities)) + "\n").encode()
    ).hexdigest()
    assert signature == manifest["case_ids_sha256"]
    corpus_digest = hashlib.sha256((ROOT / "cases.json").read_bytes()).hexdigest()
    assert corpus_digest == manifest["cases_sha256"]


def test_reference_expressions_reconstruct():
    for case in CASES:
        objective, constraints, variables = decode_case(case)
        assert isinstance(objective, sp.Expr), case["case_id"]
        assert len(variables) == len(set(variables)) > 0, case["case_id"]
        assert all(isinstance(v, sp.Symbol) for v in variables), case["case_id"]
        assert all(
            isinstance(
                r,
                (
                    sp.core.relational.Relational,
                    sp.And,
                    sp.Or,
                    sp.logic.boolalg.BooleanAtom,
                ),
            )
            for r in constraints
        ), case["case_id"]
        for value in case["expected_values"]:
            assert isinstance(sp.sympify(value), sp.Expr), case["case_id"]
        if case["kind"] in {"optimum", "branch_domain"}:
            assert len(case["expected_values"]) >= 1, case["case_id"]


def test_cubic_constraints_preserved():
    case = next(
        item
        for item in CASES
        if item["case_id"]
        == "global_optimization_minimize_result_x_power_2_2_y_power_2_3_z_power_2_c060e51e9b"
    )
    objective, constraints, (x, y, z) = decode_case(case)
    assert objective == x * x + 2 * y * y + 3 * z * z
    assert set(constraints) == {
        sp.Ge(x**3 - 2 * x * y + 3 * z**3, 0),
        sp.Ge(x - y + z * z, 0),
    }


def test_fractional_part_sign():
    case = next(
        item
        for item in CASES
        if item["case_id"]
        == "univariate_maximize_result_x_piecewise_floor_x_x_0_ceiling_x_true_2b99d70931"
    )
    expression, _, (x,) = decode_case(case)
    assert expression.subs(x, sp.Rational(-7, 4)) == -sp.Rational(3, 4)
    assert expression.subs(x, sp.Rational(7, 4)) == sp.Rational(3, 4)
