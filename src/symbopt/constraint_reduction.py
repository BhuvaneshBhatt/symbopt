from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import sympy as sp
from sympy.core.relational import Relational


@dataclass
class ReductionReport:
    """Reduced constraints, substitutions and the evidence used by preprocessing."""

    duplicate_rels: int = 0
    trivial_rels: int = 0
    unused_aux: int = 0
    duplicate_lifts: int = 0

    @property
    def changed(self) -> bool:
        """Whether preprocessing recorded any supported structural change."""
        return any(
            [
                self.duplicate_rels,
                self.trivial_rels,
                self.unused_aux,
                self.duplicate_lifts,
            ]
        )


def reduce_relations(relations: Sequence[Relational]):
    """Simplify and deduplicate relations, retaining exact contradictions."""
    return _reduce_relations(relations, {})


def _reduce_relations(relations, simplified):
    """Reuse simplification results across constraint groups while counting each occurrence."""
    out: list[Relational] = []
    seen = set()
    duplicates = 0
    trivial = 0
    for rel in relations:
        if rel not in simplified:
            try:
                simplified[rel] = sp.simplify(rel)
            except (TypeError, ValueError, ArithmeticError, NotImplementedError):
                simplified[rel] = rel
        simp = simplified[rel]
        if simp is sp.S.true or simp is True:
            trivial += 1
            continue
        # Keep the false relation because it is the exact infeasibility witness
        # consumed by downstream feasibility checks.
        rel2 = rel if simp is sp.S.false or simp is False else simp
        if rel2 in seen:
            duplicates += 1
            continue
        seen.add(rel2)
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
    simplified = {}
    for name, values in fields:
        items, dup, triv = _reduce_relations(values, simplified)
        reduced[name] = items
        report.duplicate_rels += dup
        report.trivial_rels += triv

    used = set(problem.lifted_objective.free_symbols)
    for name, _ in fields:
        for rel in reduced[name]:
            used.update(getattr(rel, "free_symbols", set()))

    original_aux = list(problem.auxiliary_variables)
    original_aux_set = set(original_aux)
    auxiliary_variables = [aux for aux in original_aux if aux in used]
    retained_aux = set(auxiliary_variables)
    report.unused_aux = len(original_aux) - len(auxiliary_variables)

    original_vars = [
        var for var in problem.all_variables if var not in original_aux_set
    ]
    all_variables = original_vars + auxiliary_variables

    seen_records = set()
    new_records = []
    for rec in problem.lift_records:
        key = (rec.aux, rec.original_expr, rec.kind)
        if key in seen_records:
            report.duplicate_lifts += 1
            continue
        seen_records.add(key)
        if rec.aux in retained_aux:
            new_records.append(rec)

    problem.update_structure(
        **reduced,
        auxiliary_variables=auxiliary_variables,
        all_variables=all_variables,
        lift_records=new_records,
    )

    if report.changed:
        problem.preprocessing_notes.append(
            "conservative reduction: "
            f"duplicates={report.duplicate_rels}, "
            f"trivial={report.trivial_rels}, "
            f"unused_aux={report.unused_aux}, "
            f"duplicate_lift_records={report.duplicate_lifts}"
        )
    return problem, report
