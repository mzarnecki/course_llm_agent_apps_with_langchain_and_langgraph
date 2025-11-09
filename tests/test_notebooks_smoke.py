import pathlib
import pytest

try:
    import nbformat  # noqa: F401
except Exception:  # pragma: no cover
    pytest.skip("nbformat not installed", allow_module_level=True)

NOTEBOOK_ROOT = pathlib.Path(__file__).resolve().parents[1] / "notebooks"

@pytest.mark.parametrize("nb_path", [
    p for p in NOTEBOOK_ROOT.rglob("*.ipynb") if "expensive" not in p.name
])
def test_notebook_exists(nb_path):
    assert nb_path.exists()
