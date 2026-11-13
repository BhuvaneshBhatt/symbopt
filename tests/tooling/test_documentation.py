import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


def test_required_documentation_pages_exist():
    required = [
        "index.md",
        "getting-started/installation.md",
        "getting-started/quickstart.md",
        "getting-started/problem-formulation.md",
        "getting-started/understanding-results.md",
        "guide/solver-selection.md",
        "guide/certification.md",
        "reference/capability-matrix.md",
        "reference/limitations.md",
        "methods/critical-points.md",
        "methods/kkt-karush-john.md",
        "methods/algebraization.md",
        "methods/elimination.md",
        "methods/linear-programming.md",
        "methods/quadratic-programming.md",
        "methods/semidefinite-programming.md",
        "methods/sum-of-squares.md",
        "methods/moment-methods.md",
        "methods/sparse-structure.md",
        "methods/degeneracy.md",
        "methods/verified-intervals.md",
        "examples/index.md",
        "guide/how-symbopt-solved-this.md",
        "api/public-symbols.md",
    ]
    missing = [name for name in required if not (DOCS / name).exists()]
    assert not missing


def test_myst_internal_doc_targets_exist():
    missing = []
    for page in DOCS.rglob("*.md"):
        text = page.read_text()
        for match in re.finditer(r"\{doc\}`([^`]+)`", text):
            target = match.group(1)
            if target.startswith("http"):
                continue
            candidate = page.parent / target
            if candidate.suffix == "":
                candidate = candidate.with_suffix(".md")
            if not candidate.resolve().exists():
                missing.append((page.name, target))
    assert not missing


def test_readthedocs_and_sphinx_configuration_exist():
    assert (ROOT / ".readthedocs.yaml").exists()
    assert (DOCS / "conf.py").exists()


def test_public_api_documentation_ownership():
    from symbopt.api import __all__ as stable_api

    registry = DOCS / "api" / "public-symbols.md"
    pattern = re.compile(
        r"^\| `symbopt\.([^`]+)` \| \{doc\}`([^`]+)` \|$", re.MULTILINE
    )
    entries = pattern.findall(registry.read_text())
    registered = [name for name, _owner in entries]
    assert len(registered) == len(set(registered)), (
        "duplicate public API registry entries"
    )
    assert set(registered) == set(stable_api)

    bad_owners = []
    for name, owner in entries:
        page = DOCS / "api" / f"{owner}.md"
        if not page.exists() or f"symbopt.{name}" not in page.read_text():
            bad_owners.append((name, owner))
    assert not bad_owners


def test_reference_and_development_pages_exist():
    required = [
        "api/index.md",
        "reference/numerical-tolerances.md",
        "reference/performance.md",
        "reference/references.md",
        "development/architecture.md",
        "development/testing.md",
        "development/documentation.md",
    ]
    missing = [name for name in required if not (DOCS / name).exists()]
    assert not missing


def _resolve_dotted_object(path: str):
    """Import a dotted autodoc target and return the referenced object."""
    import importlib

    parts = path.split(".")
    for cut in range(len(parts), 0, -1):
        module_name = ".".join(parts[:cut])
        try:
            obj = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        for attr in parts[cut:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(path)


def test_myst_autodoc_targets_import():
    """Ensure documented functions/classes resolve before Sphinx runs autodoc."""
    missing = []
    pattern = re.compile(r"\{auto(?:function|class)\}\s+([^\s`]+)")
    for page in DOCS.rglob("*.md"):
        for target in pattern.findall(page.read_text()):
            try:
                _resolve_dotted_object(target)
            except (ImportError, AttributeError, ModuleNotFoundError) as exc:
                missing.append(
                    (str(page.relative_to(DOCS)), target, type(exc).__name__)
                )
    assert not missing
