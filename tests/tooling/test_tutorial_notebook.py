from __future__ import annotations

import ast
import json
from pathlib import Path

import symbopt

NOTEBOOK = Path("notebooks/symbopt_tutorial.ipynb")


def _payload():
    return json.loads(NOTEBOOK.read_text())


def test_tutorial_root_imports_are_public():
    """Keep tutorial imports synchronized with the supported root API."""
    imported: set[str] = set()
    for cell in _payload()["cells"]:
        if cell.get("cell_type") != "code":
            continue
        tree = ast.parse("".join(cell.get("source", [])))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "symbopt":
                imported.update(alias.name for alias in node.names)
    missing = sorted(imported.difference(symbopt.__all__))
    assert not missing, f"Tutorial imports missing from root API: {missing}"


def test_tutorial_cells_have_ids():
    ids = [cell.get("id") for cell in _payload()["cells"]]
    assert all(ids)
    assert len(ids) == len(set(ids))


def test_tutorial_executes_and_its_core_claims_are_true():
    """Execute every code cell and assert the tutorial's semantic promises."""
    namespace = {"__name__": "__symbopt_tutorial_test__"}
    for index, cell in enumerate(_payload()["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        exec(  # noqa: S102 -- executing repository-owned tutorial cells is the test contract
            compile(source, f"{NOTEBOOK}:cell-{index}", "exec"), namespace
        )

    assert namespace["quickstart_result"] == [0, {namespace["x"]: 2}]
    assert namespace["detailed_result"].certified is True
    assert namespace["planner_example"].complexity.finite_box is True
    assert namespace["sos_demo"].certificate.verified is True
    assert namespace["critical_cone_demo"].classification == "IndefiniteCriticalCone"
    assert namespace["interval_demo"].rigorous is True
    assert namespace["diagnostic_demo"]["method"] != "unknown"
