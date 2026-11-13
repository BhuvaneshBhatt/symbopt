"""Every advertised Markdown Python session runs without a hidden namespace."""

import importlib.util

import pytest

from tools.check_docs import documents, execute_document, python_blocks

DEPENDENT_PAGES = {"ranges-and-extrema", "univariate-analysis"}
PAGES = [path for path in documents() if list(python_blocks(path))]


@pytest.mark.parametrize("path", PAGES, ids=lambda p: p.stem)
def test_document_session(path):
    if path.stem in DEPENDENT_PAGES and importlib.util.find_spec("funcprops") is None:
        pytest.skip("optional funcprops analysis examples")
    execute_document(path)


def test_broken_assertion(tmp_path):
    path = tmp_path / "broken.md"
    path.write_text("```python\nassert 2+2 == 5\n```\n")
    with pytest.raises(AssertionError):
        execute_document(path)


def test_page_namespace(tmp_path):
    first = tmp_path / "first.md"
    first.write_text("```python\nvalue = 3\n```\n\n```python\nassert value == 3\n```\n")
    execute_document(first)
    second = tmp_path / "second.md"
    second.write_text("```python\nassert value == 3\n```\n")
    with pytest.raises(NameError):
        execute_document(second)


def test_unclosed_fence(tmp_path):
    path = tmp_path / "broken.md"
    path.write_text("```python\nvalue = 3\n")
    with pytest.raises(ValueError):
        list(python_blocks(path))
