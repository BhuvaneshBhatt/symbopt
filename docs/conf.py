from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

with (ROOT / "pyproject.toml").open("rb") as handle:
    metadata = tomllib.load(handle)["project"]

project = "symbopt"
author = ", ".join(item["name"] for item in metadata.get("authors", []))
release = metadata["version"]
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
]

autosummary_generate = True
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
]
myst_heading_anchors = 4

html_theme = "sphinx_rtd_theme"
html_title = "symbopt documentation"
html_context = {
    "display_github": True,
    "github_user": "BhuvaneshBhatt",
    "github_repo": "symbopt",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
}

exclude_patterns = ["_build"]
