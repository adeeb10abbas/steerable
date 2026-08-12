from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest


SOURCE = Path(__file__).parents[1] / "tools/run_v3e006_zero_model_preflight.py"
SPEC = importlib.util.spec_from_file_location("v3e006_preflight_harness", SOURCE)
assert SPEC is not None and SPEC.loader is not None
HARNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARNESS)


def test_interpreter_binding_preserves_venv_symlink_path(tmp_path: Path) -> None:
    target = tmp_path / "python-build" / "python3.11"
    target.parent.mkdir()
    target.write_bytes(b"interpreter")
    target.chmod(0o755)
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(target)

    bound = HARNESS.require_interpreter(venv_python)

    assert bound["path"] == str(venv_python)
    assert bound["resolved_path"] == str(target)
    assert bound["path"] != bound["resolved_path"]


@pytest.mark.parametrize("path_kind", ["relative", "non_executable"])
def test_interpreter_binding_fails_closed(tmp_path: Path, path_kind: str) -> None:
    if path_kind == "relative":
        path = Path("python")
    else:
        path = tmp_path / "python"
        path.write_bytes(b"interpreter")
        path.chmod(0o644)
        assert not os.access(path, os.X_OK)
    with pytest.raises(ValueError):
        HARNESS.require_interpreter(path)
