"""Every advertised capability has a live link to checked example mathematics."""

import ast
import re
from pathlib import Path

import pytest

from tools.check_docs import python_blocks

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "docs/reference/capability-matrix.md"


def _rows():
    lines = MATRIX.read_text().splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("|") or re.fullmatch(r"[| :\-]+", line):
            continue
        if index + 1 < len(lines) and re.fullmatch(r"[| :\-]+", lines[index + 1]):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        yield cells[0], cells[-1]


@pytest.mark.parametrize("capability,link", list(_rows()), ids=lambda value: value)
def test_capability_example(capability, link):
    match = re.fullmatch(r"\[[^\]]+\]\(([^#)]+)#([^\)]+)\)", link)
    assert match, f"{capability}: link an executable example section"
    path = (MATRIX.parent / match[1]).resolve()
    assert path.is_relative_to(ROOT) and path.is_file(), capability
    # Ignore headings inside code fences, where Python comments can begin '#'.
    headings, fenced = [], False
    for line, text in enumerate(path.read_text().splitlines(), 1):
        if text.startswith("```"):
            fenced = not fenced
        heading = re.match(r"^(#{1,6}) (.+)$", text)
        if heading and not fenced:
            slug = re.sub(r"[^\w\s-]", "", heading[2].lower()).replace(" ", "-")
            headings.append((line, len(heading[1]), slug))
    selected = [(i, level) for i, level, slug in headings if slug == match[2]]
    assert len(selected) == 1, f"{capability}: missing or duplicate anchor {match[2]}"
    start, level = selected[0]
    end = next(
        (i for i, depth, _ in headings if i > start and depth <= level), float("inf")
    )
    examples = [source for line, source in python_blocks(path) if start < line < end]
    assert any(
        isinstance(node, ast.Assert)
        for source in examples
        for node in ast.walk(ast.parse(source))
    ), f"{capability}: example needs checked mathematics"
