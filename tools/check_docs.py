"""Execute Python documentation sessions without an implicit shared setup."""

import contextlib
import io
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def documents(root=ROOT):
    """Return maintained Markdown pages, excluding generated build output."""
    return [
        root / "README.md",
        *sorted(p for p in (root / "docs").rglob("*.md") if "_build" not in p.parts),
    ]


def python_blocks(path):
    """Extract complete Python fences and reject unclosed or annotated fences."""
    opening = None
    lines = []
    for number, line in enumerate(path.read_text().splitlines(True), 1):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})([^\n]*)", line)
        if opening is None:
            if marker:
                opening = (marker[1], marker[2].strip(), number + 1)
                lines = []
        elif (
            marker
            and marker[1][0] == opening[0][0]
            and len(marker[1]) >= len(opening[0])
            and not marker[2].strip()
        ):
            if opening[1].split()[:1] in (["python"], ["py"]):
                if opening[1] not in {"python", "py"}:
                    raise ValueError(f"unsupported Python fence in {path}")
                yield opening[2], "".join(lines)
            opening = None
        else:
            lines.append(line)
    if opening is not None:
        raise ValueError(f"unclosed fence in {path}")


def execute_document(path):
    """Run each page as one fresh session and return its executed block count."""
    namespace = {"__name__": "__documentation__"}
    blocks = list(python_blocks(path))
    with contextlib.redirect_stdout(io.StringIO()):
        for line, source in blocks:
            exec(compile("\n" * (line - 1) + source, str(path), "exec"), namespace)  # noqa: S102
    return len(blocks)


def main():
    """Execute every Python example in maintained Markdown documentation."""
    total = sum(
        execute_document(path) for path in documents() if "_build" not in path.parts
    )
    print(f"Executed {total} Python documentation blocks")


if __name__ == "__main__":
    main()
