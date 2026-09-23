from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

from experiments.workshops.spatial_grounding_v1 import native_worker_entrypoint
from experiments.workshops.spatial_grounding_v1 import runtime


def test_configure_app_uses_qualified_single_env_realtime_options(
    monkeypatch,
) -> None:
    observed = {}

    class FakeLauncher:
        def __init__(self, args):
            observed.update(vars(args))
            self.app = self
            self.closed = False

        def close(self):
            self.closed = True

    app_module = types.ModuleType("isaaclab.app")
    app_module.AppLauncher = FakeLauncher
    isaaclab = types.ModuleType("isaaclab")
    isaaclab.app = app_module
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.app", app_module)
    monkeypatch.setenv("SGW01_SIMULATOR_DEVICE", "cuda:0")

    args = argparse.Namespace(
        device="cuda:99", headless=False, enable_cameras=False, num_envs=8, rendering_mode="quality"
    )
    launcher = native_worker_entrypoint._configure_app(args)

    assert observed["device"] == "cuda:0"
    assert observed["headless"] is True
    assert observed["enable_cameras"] is True
    assert observed["num_envs"] == 1
    assert observed["rendering_mode"] == "balanced"
    launcher.app.close()
    assert launcher.closed is True


def test_configure_app_requires_all_renderer_fields_and_close(monkeypatch) -> None:
    monkeypatch.setenv("SGW01_SIMULATOR_DEVICE", "cuda:0")
    args = argparse.Namespace(device="cuda:0", headless=False, enable_cameras=False)
    import pytest
    with pytest.raises(RuntimeError, match="renderer/device"):
        native_worker_entrypoint._configure_app(args)

    class NoCloseLauncher:
        app = object()

        def __init__(self, args):
            self.app = self.app

    app_module = types.ModuleType("isaaclab.app")
    app_module.AppLauncher = NoCloseLauncher
    isaaclab = types.ModuleType("isaaclab")
    isaaclab.app = app_module
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.app", app_module)
    args.num_envs = 1
    args.rendering_mode = "balanced"
    with pytest.raises(RuntimeError, match="app.close"):
        native_worker_entrypoint._configure_app(args)


def test_run_closes_simulator_after_existing_worker_returns(monkeypatch) -> None:
    events: list[str] = []

    class Launcher:
        app = None

        def __init__(self):
            self.app = self

        def close(self):
            events.append("close")

    launcher = Launcher()
    monkeypatch.setattr(native_worker_entrypoint, "_preflight", lambda args: "release")
    monkeypatch.setattr(native_worker_entrypoint, "_configure_app", lambda args: launcher)
    fake_worker = types.ModuleType("experiments.workshops.spatial_grounding_v1.worker")
    fake_worker.run_partition = lambda *args, **kwargs: events.append("worker") or 0
    monkeypatch.setitem(sys.modules, fake_worker.__name__, fake_worker)
    result = native_worker_entrypoint.run(
        argparse.Namespace(
            model="N3", family="LAT", stage="P", max_valid_episodes=2,
            max_cell_attempts=3, heartbeat_seconds=60,
        )
    )
    assert result == 0
    assert events == ["worker", "close"]


def test_preflight_rejects_false_stage_authorization_before_app(
    monkeypatch, tmp_path: Path
) -> None:
    class Release:
        def partition(self, model, family, stage):
            return [object()]

    contract = types.ModuleType("experiments.workshops.spatial_grounding_v1.contract")
    contract.ContractError = RuntimeError
    contract.load_release = lambda path: Release()
    worker = types.ModuleType("experiments.workshops.spatial_grounding_v1.worker")
    worker._stage_authorized = lambda release, stage: False
    monkeypatch.setitem(sys.modules, contract.__name__, contract)
    monkeypatch.setitem(sys.modules, worker.__name__, worker)
    args = argparse.Namespace(
        release=tmp_path / "release", model="N3", family="LAT", stage="P",
        max_valid_episodes=1, max_cell_attempts=3,
    )
    import pytest
    with pytest.raises(RuntimeError, match="not authorized"):
        native_worker_entrypoint._preflight(args)


def test_owned_server_gpu_override_changes_environment_not_argv(
    monkeypatch, tmp_path: Path
) -> None:
    captured = {}

    class Process:
        pid = 123

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs["env"]
        return Process()

    monkeypatch.setattr(runtime.subprocess, "Popen", fake_popen)
    argv = ["python", "-m", "native_server", "--port", "8123"]
    runtime._launch_owned_server(argv, tmp_path, child_env={"CUDA_VISIBLE_DEVICES": "2"})
    assert captured["argv"] == argv
    assert captured["env"] == {"CUDA_VISIBLE_DEVICES": "2"}


def test_policy_gpu_override_must_be_in_allocated_visible_set(monkeypatch) -> None:
    monkeypatch.setenv("SGW01_POLICY_CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    import pytest
    with pytest.raises(Exception, match="outside the allocated"):
        runtime._policy_child_env()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2")
    assert runtime._policy_child_env()["CUDA_VISIBLE_DEVICES"] == "2"
