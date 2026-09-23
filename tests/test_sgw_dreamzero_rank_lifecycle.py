from __future__ import annotations

import json
import datetime
from pathlib import Path
import socket
import subprocess
import sys
import types
import time

import pytest

from experiments.workshops.spatial_grounding_v1.adapters import AdapterError
from experiments.workshops.spatial_grounding_v1.dreamzero_rank_lifecycle import (
    OwnedD1RankLifecycle,
)
from experiments.workshops.spatial_grounding_v1 import dreamzero_wrapper_entrypoint as entrypoint
from experiments.workshops.spatial_grounding_v1.dreamzero_backend import collective_timeout


SOURCE = "ab790c198fbce33503358efbbd4187ce9a89adf3"
CHECKPOINT = "96ad344138c66e82536422432ad742f015784942"


def _worker_argv(body: str) -> list[str]:
    return [sys.executable, "-c", body]


def test_owned_rank_lifecycle_starts_and_reaps_workers(tmp_path: Path) -> None:
    body = (
        "import json, os, pathlib, time\n"
        "import subprocess\n"
        "p=pathlib.Path(os.environ['SGW01_D1_RANK_READY_DIR']) / "
        "f\"rank-{os.environ['SGW01_D1_RANK']}.json\"\n"
        "st=subprocess.check_output(['ps','-p',str(os.getpid()),'-o','lstart='],text=True).strip()\n"
        "cfg={'num_inference_steps':16,'seed':1140,'cfg_scale':5.0,'action_output_dim':8,'rank': int(os.environ['SGW01_D1_RANK'])}\n"
        "p.write_text(json.dumps({'rank': int(os.environ['SGW01_D1_RANK']), 'pid': os.getpid(), 'start_time': st, "
        "'run_nonce': os.environ['SGW01_D1_RUN_NONCE'], "
        "'source_commit': os.environ['SGW01_D1_SOURCE_COMMIT'], "
        "'checkpoint_revision': os.environ['SGW01_D1_CHECKPOINT_REVISION'], 'native_config': cfg, "
        "'native_config_sha256': __import__('hashlib').sha256((json.dumps(cfg,sort_keys=True,separators=(',',':'))+'\\n').encode()).hexdigest()}))\n"
        "time.sleep(30)\n"
    )
    lifecycle = OwnedD1RankLifecycle(
        worker_argv=_worker_argv(body),
        world_size=3,
        log_dir=tmp_path / "logs",
        ready_dir=tmp_path / "ready",
        source_commit=SOURCE,
        checkpoint_revision=CHECKPOINT,
        startup_timeout=2,
        shutdown_timeout=2,
    )
    lifecycle.start()
    lifecycle.await_ready()
    workers = list(lifecycle.workers)
    pids = [worker.process.pid for worker in workers]
    assert len(pids) == 2
    lifecycle.stop()
    assert all(worker.process.poll() is not None for worker in workers)


def test_owned_rank_lifecycle_cleans_up_on_worker_failure(tmp_path: Path) -> None:
    body = "raise SystemExit(17)\n"
    lifecycle = OwnedD1RankLifecycle(
        worker_argv=_worker_argv(body),
        world_size=2,
        log_dir=tmp_path / "logs",
        ready_dir=tmp_path / "ready",
        source_commit=SOURCE,
        checkpoint_revision=CHECKPOINT,
        startup_timeout=1,
        shutdown_timeout=1,
    )
    lifecycle.start()
    with pytest.raises(AdapterError, match="exited during startup"):
        lifecycle.await_ready()
    lifecycle.stop()
    assert lifecycle.workers == []
    assert (tmp_path / "logs" / "rank-1.stderr.log").is_file()


def test_owned_rank_lifecycle_timeout_reaps_children(tmp_path: Path) -> None:
    body = "import time; time.sleep(30)\n"
    lifecycle = OwnedD1RankLifecycle(
        worker_argv=_worker_argv(body),
        world_size=2,
        log_dir=tmp_path / "logs",
        ready_dir=tmp_path / "ready",
        source_commit=SOURCE,
        checkpoint_revision=CHECKPOINT,
        startup_timeout=0.1,
        shutdown_timeout=1,
    )
    lifecycle.start()
    with pytest.raises(AdapterError, match="timed out"):
        lifecycle.await_ready()
    lifecycle.stop()
    assert lifecycle.workers == []


def test_owned_rank_lifecycle_rejects_non_loopback_rendezvous(tmp_path: Path) -> None:
    with pytest.raises(AdapterError, match="loopback"):
        OwnedD1RankLifecycle(
            worker_argv=["true"],
            world_size=2,
            log_dir=tmp_path / "logs",
            ready_dir=tmp_path / "ready",
            source_commit=SOURCE,
            checkpoint_revision=CHECKPOINT,
            master_addr="0.0.0.0",
        )


@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1", "not-a-number"])
def test_collective_timeout_rejects_invalid_deadlines(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("SGW01_D1_COLLECTIVE_TIMEOUT", value)
    with pytest.raises(AdapterError, match="finite positive"):
        collective_timeout()


def test_native_init_mesh_receives_bounded_timeout() -> None:
    calls: list[object] = []

    class Dist:
        def init_process_group(self, *args: object, **kwargs: object) -> None:
            calls.append((args, kwargs))

    module = types.SimpleNamespace(dist=Dist())
    entrypoint._install_bounded_init_process_group(module, datetime.timedelta(seconds=7))
    module.dist.init_process_group("nccl")
    assert calls == [(("nccl",), {"timeout": datetime.timedelta(seconds=7)})]


def test_native_signal_group_receives_bounded_timeout() -> None:
    calls: list[object] = []

    class Dist:
        def new_group(self, **kwargs: object) -> str:
            calls.append(kwargs)
            return "signal-group"

    module = types.SimpleNamespace(dist=Dist())
    result = entrypoint._new_bounded_signal_group(module, datetime.timedelta(seconds=7))
    assert result == "signal-group"
    assert calls == [{"backend": "gloo", "timeout": datetime.timedelta(seconds=7)}]


def test_rank_zero_startup_guard_bounds_native_construction(tmp_path: Path) -> None:
    lifecycle = OwnedD1RankLifecycle(
        worker_argv=["true"],
        world_size=2,
        log_dir=tmp_path / "logs",
        ready_dir=tmp_path / "ready",
        source_commit=SOURCE,
        checkpoint_revision=CHECKPOINT,
        startup_timeout=0.05,
        shutdown_timeout=1,
    )
    lifecycle.start()
    try:
        with pytest.raises(AdapterError, match="native construction exceeded"):
            with lifecycle.startup_guard():
                time.sleep(0.2)
    finally:
        lifecycle.stop()


def test_main_propagates_verified_identity_to_lifecycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SGW01_D1_WORLD_SIZE", "2")
    monkeypatch.setenv("SGW01_D1_HOST", "127.0.0.1")
    monkeypatch.setenv("SGW01_D1_PORT", "8123")
    monkeypatch.setenv("SGW01_D1_RANK_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("SGW01_D1_RANK_READY_DIR", str(tmp_path / "ready"))
    monkeypatch.setenv("SGW01_TRACE_SIDECAR", str(tmp_path / "trace"))
    monkeypatch.setenv("SGW01_FUTURE_DIR", str(tmp_path / "future"))
    monkeypatch.setenv("SGW01_SERVER_ATTESTATION", str(tmp_path / "attestation"))
    monkeypatch.setenv("SGW01_D1_SOURCE_COMMIT", "caller-string")
    monkeypatch.setenv("SGW01_D1_CHECKPOINT_REVISION", "caller-string")
    verified = {
        "source_root": str(tmp_path),
        "checkpoint_path": str(tmp_path / "checkpoint"),
        "source_commit": SOURCE,
        "checkpoint_revision": CHECKPOINT,
    }
    captured: dict[str, object] = {}
    monkeypatch.setattr(entrypoint, "verify_pinned_dreamzero_prerequisites", lambda: verified)

    class SpyLifecycle:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def configure_rank_zero(self) -> None:
            pass

        def install_signal_cleanup(self) -> None:
            pass

        def __enter__(self) -> "SpyLifecycle":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def startup_guard(self):
            from contextlib import nullcontext
            return nullcontext()

        def await_ready(self) -> None:
            raise AdapterError("stop after lifecycle construction")

    monkeypatch.setattr(entrypoint, "OwnedD1RankLifecycle", SpyLifecycle)
    monkeypatch.setattr(
        entrypoint,
        "build_pinned_dreamzero_backend",
        lambda: (_ for _ in ()).throw(AdapterError("stop before native construction")),
    )
    with pytest.raises(AdapterError, match="stop before native construction"):
        entrypoint.main()
    assert captured["source_commit"] == SOURCE
    assert captured["checkpoint_revision"] == CHECKPOINT


def test_owned_rank_lifecycle_synchronizes_cpu_inference_barrier(tmp_path: Path) -> None:
    coordinator = socket.socket()
    coordinator.bind(("127.0.0.1", 0))
    coordinator.listen(2)
    port = coordinator.getsockname()[1]
    body = (
        "import json, os, pathlib, socket, subprocess\n"
        "s=socket.create_connection((os.environ['MASTER_ADDR'], int(os.environ['MASTER_PORT'])))\n"
        "s.sendall((os.environ['SGW01_D1_RANK']+'\\n').encode())\n"
        "st=subprocess.check_output(['ps','-p',str(os.getpid()),'-o','lstart='],text=True).strip()\n"
        "cfg={'num_inference_steps':16,'seed':1140,'cfg_scale':5.0,'action_output_dim':8,'rank': int(os.environ['SGW01_D1_RANK'])}\n"
        "p=pathlib.Path(os.environ['SGW01_D1_RANK_READY_DIR']) / f\"rank-{os.environ['SGW01_D1_RANK']}.json\"\n"
        "p.write_text(json.dumps({'rank':int(os.environ['SGW01_D1_RANK']),'pid':os.getpid(),'start_time':st,"
        "'run_nonce':os.environ['SGW01_D1_RUN_NONCE'],'source_commit':os.environ['SGW01_D1_SOURCE_COMMIT'],"
        "'checkpoint_revision':os.environ['SGW01_D1_CHECKPOINT_REVISION'],'native_config':cfg,"
        "'native_config_sha256':__import__('hashlib').sha256((json.dumps(cfg,sort_keys=True,separators=(',',':'))+'\\n').encode()).hexdigest()}))\n"
        "s.recv(16); s.sendall(b'done')\n"
    )
    lifecycle = OwnedD1RankLifecycle(
        worker_argv=_worker_argv(body),
        world_size=3,
        log_dir=tmp_path / "logs",
        ready_dir=tmp_path / "ready",
        source_commit=SOURCE,
        checkpoint_revision=CHECKPOINT,
        master_port=port,
        startup_timeout=2,
        shutdown_timeout=2,
    )
    try:
        lifecycle.start()
        connections = [coordinator.accept()[0] for _ in range(2)]
        assert {conn.recv(16).decode().strip() for conn in connections} == {"1", "2"}
        lifecycle.await_ready()
        for conn in connections:
            conn.sendall(b"infer")
        assert {conn.recv(16) for conn in connections} == {b"done"}
        lifecycle.assert_healthy()
        for conn in connections:
            conn.close()
    finally:
        lifecycle.stop()
        coordinator.close()


def test_exported_ar_source_keeps_rank0_server_and_worker_loop_split() -> None:
    export = Path(
        "/Users/SZ5VJY/.copilot/session-state/"
        "c230f3cd-3fe9-4f1f-9ee4-e8b857151a60/files/"
        "sgw-native-d1-ar-server-ak.json"
    )
    if not export.exists():
        pytest.skip("authorized D1 source export is not present")
    source = json.loads(export.read_text())["files"]["socket_test_optimized_AR.py"]["text"]
    assert "asyncio.run(server._worker_loop())" in source
    assert "if rank == 0:" in source
    assert "RoboarenaServer(" in source
    assert "lazy_joint_forward_causal" in source
    assert "signal == 1" in source
    assert "signal == 2" in source


def test_wrapper_rejects_bad_host_before_native_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SGW01_D1_WORLD_SIZE", "2")
    monkeypatch.setenv("SGW01_D1_HOST", "0.0.0.0")
    monkeypatch.setenv("SGW01_D1_PORT", "8123")
    called = []
    monkeypatch.setattr(entrypoint, "verify_pinned_dreamzero_prerequisites", lambda: called.append("verify"))
    monkeypatch.setattr(entrypoint, "build_pinned_dreamzero_backend", lambda: called.append("build"))
    with pytest.raises(RuntimeError, match="loopback"):
        entrypoint.main()
    assert called == []


def test_wrapper_rejects_bad_identity_before_worker_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SGW01_D1_WORLD_SIZE", "2")
    monkeypatch.setenv("SGW01_D1_HOST", "127.0.0.1")
    monkeypatch.setenv("SGW01_D1_PORT", "8123")
    monkeypatch.setattr(
        entrypoint,
        "verify_pinned_dreamzero_prerequisites",
        lambda: (_ for _ in ()).throw(AdapterError("checkpoint hash mismatch")),
    )
    spawned = []

    class SpyLifecycle:
        def __init__(self, **kwargs: object) -> None:
            spawned.append(kwargs)

    monkeypatch.setattr(entrypoint, "OwnedD1RankLifecycle", SpyLifecycle)
    with pytest.raises(AdapterError, match="checkpoint hash mismatch"):
        entrypoint.main()
    assert spawned == []
