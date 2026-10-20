"""Import-only smoke test for the course notebooks.

Runs every import statement found in each notebook in a fresh interpreter. It needs no
API keys, so it can run in CI and catches broken or incompatible dependencies, e.g. a
library release that removes a symbol another library still imports.

Run with: uv run pytest tests/test_notebook_imports.py
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*/*.ipynb"))


def notebook_imports(path: Path) -> list[str]:
    """Return the import statements used in the code cells of a notebook."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    imports = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        # IPython magics and shell commands are not valid Python
        lines = [
            line
            for line in "".join(cell["source"]).splitlines()
            if not line.lstrip().startswith(("%", "!"))
        ]
        try:
            tree = ast.parse("\n".join(lines))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import | ast.ImportFrom) and not getattr(node, "level", 0):
                statement = ast.unparse(node)
                if statement not in imports:
                    imports.append(statement)
    return imports


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_notebook_imports(path: Path) -> None:
    imports = notebook_imports(path)
    assert imports, f"no imports found in {path}"
    result = subprocess.run(
        [sys.executable, "-c", "\n".join(imports)],
        cwd=path.parent,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-3000:]
