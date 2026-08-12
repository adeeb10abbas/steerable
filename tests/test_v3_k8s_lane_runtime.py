from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys
import threading

import pytest


ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "deploy/k8s/v3_lane_bundle/scripts"
SPEC = importlib.util.spec_from_file_location("startup_preflight", SCRIPTS / "startup_preflight.py")
assert SPEC is not None and SPEC.loader is not None
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)
sys.modules.setdefault("startup_preflight", PREFLIGHT)
ENTRY_SPEC = importlib.util.spec_from_file_location("lane_entrypoint", SCRIPTS / "lane_entrypoint.py")
assert ENTRY_SPEC is not None and ENTRY_SPEC.loader is not None
ENTRYPOINT = importlib.util.module_from_spec(ENTRY_SPEC)
ENTRY_SPEC.loader.exec_module(ENTRYPOINT)


def _executable(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_launch_config_and_file_bindings_are_hash_bound(tmp_path: Path) -> None:
    launch = tmp_path / "launch.json"
    launch.write_text('{"role":"simulator"}\n', encoding="utf-8")
    digest = PREFLIGHT.sha256_file(launch)
    loaded, actual = PREFLIGHT.load_launch_config(launch, digest)
    assert loaded["role"] == "simulator"
    assert actual == digest
    with pytest.raises(PREFLIGHT.PreflightError, match="digest changed"):
        PREFLIGHT.load_launch_config(launch, "0" * 64)

    bound = tmp_path / "queue.jsonl"
    bound.write_text("{}\n", encoding="utf-8")
    config = {
        "file_bindings": [
            {"path": str(bound), "bytes": bound.stat().st_size, "sha256": PREFLIGHT.sha256_file(bound)}
        ]
    }
    assert PREFLIGHT.verify_file_bindings(config)[0]["path"] == str(bound.resolve())
    bound.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(PREFLIGHT.PreflightError, match="changed"):
        PREFLIGHT.verify_file_bindings(config)


def test_exact_imports_use_configured_python_bin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    marker = tmp_path / "python-used"
    wrapper = _executable(
        tmp_path / "python-wrapper",
        f"#!/bin/sh\nprintf used > {marker}\nexec {os.path.realpath(os.sys.executable)} \"$@\"\n",
    )
    monkeypatch.setenv("PYTHON_BIN", str(wrapper))
    with pytest.raises(PREFLIGHT.PreflightError, match="command failed"):
        PREFLIGHT.verify_imports({"python_imports": ["json", "curobo.fake"]})
    # The failing exact import still proves the configured interpreter was used.
    assert marker.read_text(encoding="utf-8") == "used"


def test_exact_imports_subprocess_and_curobo_requirement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / "curobo"
    package.mkdir()
    (package / "__init__.py").write_text('__version__ = "test"\n', encoding="utf-8")
    marker = tmp_path / "python-used"
    wrapper = _executable(
        tmp_path / "python-wrapper",
        f"#!/bin/sh\nprintf used > {marker}\nexec {os.path.realpath(os.sys.executable)} \"$@\"\n",
    )
    monkeypatch.setenv("PYTHON_BIN", str(wrapper))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    result = PREFLIGHT.verify_imports({"python_imports": ["json", "curobo"]})
    assert marker.read_text(encoding="utf-8") == "used"
    assert [row["module"] for row in result["results"]] == ["json", "curobo"]
    with pytest.raises(PREFLIGHT.PreflightError, match="CuRobo"):
        PREFLIGHT.verify_imports({"python_imports": ["json"]})
    policy_result = PREFLIGHT.verify_imports(
        {"python_imports": ["json"]}, require_curobo=False
    )
    assert policy_result["curobo_required"] is False
    assert [row["module"] for row in policy_result["results"]] == ["json"]


def test_preflight_checks_are_role_specific(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(PREFLIGHT, "verify_runtime_directories", lambda: {})
    monkeypatch.setattr(PREFLIGHT, "verify_writable_parent", lambda path: {"path": str(path)})
    monkeypatch.setattr(PREFLIGHT, "verify_cuda", lambda config: {"device": "cuda:0"})
    monkeypatch.setattr(
        PREFLIGHT,
        "verify_vulkan_and_render",
        lambda config, evidence: calls.append(("render", evidence)) or {"rendered_frame": True},
    )
    monkeypatch.setattr(
        PREFLIGHT,
        "verify_imports",
        lambda config, *, require_curobo=True: calls.append(("imports", require_curobo))
        or {"curobo_required": require_curobo},
    )
    monkeypatch.setattr(PREFLIGHT, "verify_ffmpeg", lambda evidence: {})
    monkeypatch.setattr(PREFLIGHT, "verify_file_bindings", lambda config: [])
    monkeypatch.setattr(
        PREFLIGHT,
        "verify_checkpoint",
        lambda config: {"checkpoint_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        PREFLIGHT,
        "query_gpu_identity",
        lambda config: {"gpu_uuid": "GPU-test", "gpu_name": "test", "driver_version": "test"},
    )
    monkeypatch.setattr(
        PREFLIGHT,
        "wait_for_policy",
        lambda config: calls.append(("policy_wait", True)) or {"health": {"status": 200, "body": "OK"}},
    )

    output_parent = tmp_path / "output"
    output_parent.mkdir()
    config_path = tmp_path / "launch.json"
    config_path.write_text("{}\n", encoding="utf-8")
    common = {
        "config_path": config_path,
        "config_sha256": PREFLIGHT.sha256_file(config_path),
        "output_parent": output_parent,
        "lane_id": "lane00",
        "attempt_id": "attempt01",
        "pod_identity": {"pod_uid": "uid"},
        "experiment_argv": ["/bin/true"],
        "image_digest": "sha256:" + "b" * 64,
    }

    policy_evidence = tmp_path / "policy-evidence"
    policy_evidence.mkdir()
    policy = PREFLIGHT.run_preflight(
        config={"readiness_contract": "http_healthz_after_checkpoint_load"},
        evidence_dir=policy_evidence,
        role="policy",
        **common,
    )
    assert "vulkan_and_rendered_frame" not in policy["checks"]
    assert policy["checks"]["python_imports"]["curobo_required"] is False
    assert ("render", policy_evidence) not in calls
    assert ("policy_wait", True) not in calls

    calls.clear()
    simulator_evidence = tmp_path / "simulator-evidence"
    simulator_evidence.mkdir()
    simulator = PREFLIGHT.run_preflight(
        config={}, evidence_dir=simulator_evidence, role="simulator", **common
    )
    assert simulator["checks"]["python_imports"]["curobo_required"] is True
    assert ("render", simulator_evidence) in calls
    assert ("policy_wait", True) in calls


def test_policy_port_probe_releases_socket_and_stale_lock_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    with socket.socket() as chooser:
        chooser.bind(("127.0.0.1", 0))
        port = chooser.getsockname()[1]
    handle, lock_path = PREFLIGHT.reserve_policy_port(port)
    assert lock_path.exists()
    with socket.socket() as rebound:
        rebound.bind(("127.0.0.1", port))
    with pytest.raises(PREFLIGHT.PreflightError, match="already exists"):
        PREFLIGHT.reserve_policy_port(port)
    handle.close()


def test_http_healthz_policy_wait_sends_valid_request_and_records_identity() -> None:
    received: list[bytes] = []
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        host, port = server.getsockname()

        def respond() -> None:
            connection, _ = server.accept()
            with connection:
                request = b""
                while b"\r\n\r\n" not in request:
                    request += connection.recv(4096)
                received.append(request)
                connection.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 3\r\nConnection: close\r\n\r\nOK\n"
                )

        thread = threading.Thread(target=respond, daemon=True)
        thread.start()
        result = PREFLIGHT.wait_for_policy(
            {
                "policy_wait": {
                    "mode": "http_healthz",
                    "host": host,
                    "port": port,
                    "service_identity": {"lane": "lane00", "attempt": "attempt01", "config_hash": "abc"},
                    "timeout_seconds": 2,
                    "poll_seconds": 0.01,
                }
            }
        )
        thread.join(timeout=1)
    assert result is not None and result["health"] == {
        "method": "GET",
        "path": "/healthz",
        "status": 200,
        "body": "OK",
    }
    assert result["service_identity"]["attempt"] == "attempt01"
    assert received and received[0].startswith(b"GET /healthz HTTP/1.1\r\n")
    assert b"\r\nHost: " in received[0]


def test_http_healthz_rejects_non_ok_body() -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        host, port = server.getsockname()

        def respond() -> None:
            connection, _ = server.accept()
            with connection:
                while b"\r\n\r\n" not in connection.recv(4096):
                    pass
                connection.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\nConnection: close\r\n\r\nNO!"
                )

        thread = threading.Thread(target=respond, daemon=True)
        thread.start()
        with pytest.raises(PREFLIGHT.PreflightError, match="invalid body"):
            PREFLIGHT.probe_http_healthz(host, port, 2.0)
        thread.join(timeout=1)


def test_absolute_path_contract_and_episode_output_is_not_created(tmp_path: Path) -> None:
    executable = _executable(tmp_path / "runner", "#!/bin/sh\nexit 0\n")
    assert PREFLIGHT.require_absolute_executable(str(executable), "runner") == executable
    with pytest.raises(PREFLIGHT.PreflightError, match="absolute"):
        PREFLIGHT.require_absolute_executable("runner", "runner")

    output_parent = tmp_path / "raw"
    output_parent.mkdir()
    pod_uid = "pod-uid"
    episode_dir = output_parent / pod_uid / "lane-lane00" / "attempt-attempt01" / "episodes"
    assert not episode_dir.exists()


def test_readiness_probe_requires_explicit_checkpoint_loaded_flag() -> None:
    script = SCRIPTS / "check_policy_ready.py"
    missing = subprocess.run(
        [sys.executable, str(script)], cwd=SCRIPTS, text=True, capture_output=True
    )
    assert missing.returncode != 0
    assert "requires explicit --checkpoint-loaded" in missing.stdout + missing.stderr
    explicit = subprocess.run(
        [sys.executable, str(script), "--checkpoint-loaded"],
        cwd=SCRIPTS,
        text=True,
        capture_output=True,
    )
    assert explicit.returncode != 0
    assert "unrecognized arguments" not in explicit.stdout + explicit.stderr


@pytest.mark.parametrize(("role", "expected_signal"), [("policy", True), ("simulator", False)])
def test_prestop_fsyncs_marker_before_policy_sigint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    expected_signal: bool,
) -> None:
    marker = tmp_path / f"{role}-prestop.json"
    monkeypatch.setenv("PRESTOP_MARKER", str(marker))
    monkeypatch.setattr(ENTRYPOINT, "required_environment", lambda: {"LANE_ROLE": role})
    signals: list[tuple[int, int]] = []

    def record_signal(pid: int, sent: int) -> None:
        assert marker.is_file()
        payload = json.loads(marker.read_text(encoding="utf-8"))
        assert payload["signal_after_fsync"] == "SIGINT"
        signals.append((pid, sent))
        if sent == 0:
            raise ProcessLookupError("PID 1 exited after SIGINT")

    monkeypatch.setattr(ENTRYPOINT.os, "kill", record_signal)
    assert ENTRYPOINT.pre_stop() == 0
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["lane_role"] == role
    assert payload["post_signal_wait_seconds"] == (120.0 if expected_signal else None)
    assert signals == ([(1, signal.SIGINT), (1, 0)] if expected_signal else [])


def test_headless_runtime_rejects_display_or_preload(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in PREFLIGHT.RUNTIME_DIRECTORY_KEYS:
        monkeypatch.setenv(key, "/tmp")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("LD_PRELOAD", "")
    with pytest.raises(PREFLIGHT.PreflightError, match="empty DISPLAY"):
        PREFLIGHT.verify_runtime_directories()
    monkeypatch.setenv("DISPLAY", "")
    monkeypatch.setenv("LD_PRELOAD", "/tmp/injected.so")
    with pytest.raises(PREFLIGHT.PreflightError, match="empty LD_PRELOAD"):
        PREFLIGHT.verify_runtime_directories()


def test_prepare_runtime_directories_creates_only_isolated_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_root = tmp_path / "lane-runtime"
    runtime_root.mkdir()
    expected = {
        "HOME": runtime_root / "home",
        "XDG_CACHE_HOME": runtime_root / "xdg/cache",
        "XDG_CONFIG_HOME": runtime_root / "xdg/config",
        "XDG_RUNTIME_DIR": runtime_root / "xdg/runtime",
        "WARP_CACHE_PATH": runtime_root / "warp",
        "MPLCONFIGDIR": runtime_root / "matplotlib",
        "TMPDIR": runtime_root / "tmp",
    }
    for key, path in expected.items():
        monkeypatch.setenv(key, str(path))
    report = PREFLIGHT.prepare_runtime_directories(runtime_root)
    assert set(report) == set(expected)
    assert all(path.is_dir() and not path.is_symlink() for path in expected.values())
    assert not (runtime_root / "episodes").exists()


def test_prepare_runtime_directories_rejects_escape_or_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_root = tmp_path / "lane-runtime"
    runtime_root.mkdir()
    outside = tmp_path / "outside"
    for key in PREFLIGHT.RUNTIME_DIRECTORY_KEYS:
        monkeypatch.setenv(key, str(runtime_root / key.lower()))
    monkeypatch.setenv("HOME", str(outside))
    with pytest.raises(PREFLIGHT.PreflightError, match="escapes"):
        PREFLIGHT.prepare_runtime_directories(runtime_root)

    monkeypatch.setenv("HOME", str(runtime_root / "link/child"))
    (runtime_root / "link").symlink_to(outside)
    with pytest.raises(PREFLIGHT.PreflightError, match="symlink"):
        PREFLIGHT.prepare_runtime_directories(runtime_root)
