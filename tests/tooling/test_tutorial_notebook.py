from __future__ import annotations

import ast
import json
from pathlib import Path

import symbopt


def test_tutorial_root_imports_are_public():
    """Keep tutorial imports synchronized with the supported root API."""
    notebook = Path("notebooks/symbopt_tutorial.ipynb")
    payload = json.loads(notebook.read_text())
    imported: set[str] = set()

    for cell in payload["cells"]:
        if cell.get("cell_type") != "code":
            continue
        tree = ast.parse("".join(cell.get("source", [])))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "symbopt":
                imported.update(alias.name for alias in node.names)

    missing = sorted(imported.difference(symbopt.__all__))
    assert not missing, f"Tutorial imports missing from root API: {missing}"


def test_tutorial_cells_have_ids():
    """Notebook cells use stable IDs required by current nbformat."""
    notebook = Path("notebooks/symbopt_tutorial.ipynb")
    payload = json.loads(notebook.read_text())
    ids = [cell.get("id") for cell in payload["cells"]]
    assert all(ids)
    assert len(ids) == len(set(ids))


def test_tutorial_sections_follow_learning_flow():
    """Keep tutorial sections ordered from modeling through certification."""
    notebook = Path("notebooks/symbopt_tutorial.ipynb")
    payload = json.loads(notebook.read_text())
    headings = []
    for cell in payload["cells"]:
        if cell.get("cell_type") != "markdown":
            continue
        for line in "".join(cell.get("source", [])).splitlines():
            if line.startswith("## ") and not line.startswith("### "):
                headings.append(line.strip())
                break

    expected = [
        "## 1. Algebraization and lifting",
        "## 2. Symbolic critical points and degeneracy",
        "## 3. Constrained KKT candidates",
        "## 4. High-level optimization API, decision variables, and result semantics",
        "## 5. Linear programming: simplex, revised simplex, and automatic routing",
        "## 6. Convex quadratic programming",
        "## 7. Structure-aware global routes",
        "## 8. Exact semialgebraic backend",
        "## 9. Unified routing and certificates",
        "## 10. Semidefinite programming",
        "## 11. Exact SOS decomposition",
        "## 12. SOS global lower bounds and multiplier certificates",
        "## 13. SOS hierarchy and exact certificate recovery",
        "## 14. Moment and localizing matrices",
        "## 15. Verified interval optimization and Arb",
        "## 16. Sparsity, overlapping chordal decomposition, separability, and symmetry",
        "## 17. Symbolic caching and precision provenance",
        "## 18. What counts as a proof?",
        "## References",
    ]
    assert headings == expected
