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
