from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import sympy as sp
from sympy.core.relational import Relational


@dataclass
class ReductionReport:
    duplicate_rels: int = 0
    trivial_rels: int = 0
    unused_aux: int = 0
    duplicate_lifts: int = 0

    @property
    def changed(self) -> bool:
        return any(
            [
                self.duplicate_rels,
                self.trivial_rels,
                self.unused_aux,
                self.duplicate_lifts,
            ]
        )


def _relation_key(rel: Relational) -> str:
    return sp.srepr(rel)


def reduce_relations(relations: Sequence[Relational]):
    out: List[Relational] = []
    seen = set()
    duplicates = 0
    trivial = 0
    for rel in relations:
        try:
            simp = sp.simplify(rel)
        except (TypeError, ValueError, ArithmeticError, NotImplementedError):
            simp = rel
        if simp is sp.S.true or simp is True:
            trivial += 1
            continue
        # Preserve a false relation so downstream feasibility checks can detect an inconsistent problem.
        rel2 = rel if simp is sp.S.false or simp is False else simp
        key = _relation_key(rel2)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        out.append(rel2)
    return out, duplicates, trivial


def reduce_lifted_problem(problem):
    """Conservative post-lift reduction.

    Only removes duplicate/trivially-true relations, duplicate lift metadata, and
    auxiliary variables that occur nowhere in the effective problem. It does not
    perform substitutions or eliminate equations, so mathematical semantics and
    branch structure are preserved.
    """
    report = ReductionReport()

    fields = [
        ("lifted_constraints", problem.lifted_constraints),
        ("defining_equalities", problem.defining_equalities),
        ("branch_inequalities", problem.branch_inequalities),
        ("strict_inequalities", problem.strict_inequalities),
        ("graph_equalities", problem.graph_equalities),
        ("graph_inequalities", problem.graph_inequalities),
    ]
    reduced = {}
    for name, values in fields:
        items, dup, triv = reduce_relations(list(values))
        reduced[name] = items
        report.duplicate_rels += dup
        report.trivial_rels += triv

    problem.lifted_constraints = reduced["lifted_constraints"]
    problem.defining_equalities = reduced["defining_equalities"]
    problem.branch_inequalities = reduced["branch_inequalities"]
    problem.strict_inequalities = reduced["strict_inequalities"]
    problem.graph_equalities = reduced["graph_equalities"]
    problem.graph_inequalities = reduced["graph_inequalities"]

    used = set(problem.lifted_objective.free_symbols)
    for rel in problem.all_constraints():
        used.update(getattr(rel, "free_symbols", set()))

    original_aux = list(problem.auxiliary_variables)
    problem.auxiliary_variables = [a for a in original_aux if a in used]
    report.unused_aux = len(original_aux) - len(problem.auxiliary_variables)

    original_vars = [v for v in problem.all_variables if v not in set(original_aux)]
    problem.all_variables = original_vars + list(problem.auxiliary_variables)

    seen_records = set()
    new_records = []
    for rec in problem.lift_records:
        key = (rec.aux, sp.srepr(rec.original_expr), rec.kind)
        if key in seen_records:
            report.duplicate_lifts += 1
            continue
        seen_records.add(key)
        if rec.aux in problem.auxiliary_variables:
            new_records.append(rec)
    problem.lift_records = new_records

    if report.changed:
        problem.preprocessing_notes.append(
            "conservative reduction: "
            f"duplicates={report.duplicate_rels}, "
            f"trivial={report.trivial_rels}, "
            f"unused_aux={report.unused_aux}, "
            f"duplicate_lift_records={report.duplicate_lifts}"
        )
    if hasattr(problem, "refresh_metadata"):
        problem.refresh_metadata()
    return problem, report
