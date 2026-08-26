import ast
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "symbopt"


def test_source_has_no_trailing_whitespace():
    for path in [
        *PACKAGE.rglob("*.py"),
        *ROOT.glob("*.md"),
        *(ROOT / "docs").rglob("*.md"),
    ]:
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            assert line == line.rstrip(), (
                f"{path}:{line_number} has trailing whitespace"
            )


def test_source_has_no_overridden_top_level_definitions():
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text())
        definitions = defaultdict(list)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                definitions[node.name].append(node.lineno)
        duplicates = {
            name: lines for name, lines in definitions.items() if len(lines) > 1
        }
        assert not duplicates, f"{path}: overridden definitions {duplicates}"


def test_local_and_argument_names_are_concise():
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text())
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.arg):
                names.add(node.arg)
        long_names = sorted(name for name in names if len(name) > 24)
        assert not long_names, f"{path}: identifiers over 24 characters: {long_names}"


def test_tests_do_not_patch_package_implementation():
    for path in (ROOT / "tests").rglob("*.py"):
        if path == Path(__file__):
            continue
        text = path.read_text()
        assert "monkeypatch" not in text, (
            f"{path}: use observable state or dependency injection"
        )
        assert ".setattr(" not in text, (
            f"{path}: use observable state or dependency injection"
        )


def test_result_mapping_does_not_use_deepcopy():
    for path in PACKAGE.rglob("*.py"):
        assert "deepcopy" not in path.read_text(), (
            f"{path}: use targeted dataclass replacement"
        )


def test_structural_routes_are_split_by_solver_family():
    assert not (PACKAGE / "special_routes.py").exists()
    for filename in (
        "linear_fractional.py",
        "trigonometric_optimization.py",
        "transcendental_analysis.py",
        "structural_routing.py",
    ):
        assert (PACKAGE / filename).exists(), filename
