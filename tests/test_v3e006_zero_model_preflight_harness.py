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


def test_harness_retains_unique_writable_home() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert '("home", "xdg", "warp", "matplotlib", "tmp")' in source
    assert '"HOME": str(cache["home"])' in source
    assert '\n            "HOME",\n' in source
    validator = (SOURCE.parent / "validate_v3e006_infrastructure_evidence.py").read_text(encoding="utf-8")
    assert 'launch.get("environment", {}).get("HOME") != expected_home' in validator


def test_generated_app_launcher_argv_uses_accepted_e004_runtime_flags() -> None:
    argv = HARNESS.E004_APP_LAUNCHER_ARGV
    assert argv == (
        "--num-envs", "1",
        "--headless",
        "--rendering_mode", "balanced",
        "--device", "cuda:0",
        "--kit_args=--/rtx/verifyDriverVersion/enabled=false",
    )
    assert "--renderer" not in argv
    assert "--rendering-type" not in argv


def test_child_failure_is_retained_before_simulator_close() -> None:
    child_source = (SOURCE.parent / "v3e006_zero_model_runtime_preflight.py").read_text(encoding="utf-8")
    retained = child_source.index('"status": "infrastructure_invalid_zero_model_runtime_health_preflight"')
    close = child_source.index("simulation_app.close()")
    reraise = child_source.index("raise health_failure.with_traceback")
    assert retained < close < reraise
    assert '"traceback": traceback.format_exc()' in child_source
    assert '"model_request_count": 0' in child_source
    assert '"behavioral_episode_count": 0' in child_source
    assert '"state_candidate_count": 0' in child_source


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
