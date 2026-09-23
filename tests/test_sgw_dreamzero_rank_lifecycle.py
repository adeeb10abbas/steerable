from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import pytest

from experiments.workshops.spatial_grounding_v1.adapters import AdapterError
from experiments.workshops.spatial_grounding_v1.dreamzero_rank_lifecycle import (
    OwnedD1RankLifecycle,
)


SOURCE = "ab790c198fbce33503358efbbd4187ce9a89adf3"
CHECKPOINT = "96ad344138c66e82536422432ad742f015784942"


def _worker_argv(body: str) -> list[str]:
    return [sys.executable, "-c", body]


def test_owned_rank_lifecycle_starts_and_reaps_workers(tmp_path: Path) -> None:
    body = (
        "import json, os, pathlib, time\n"
        "p=pathlib.Path(os.environ['SGW01_D1_RANK_READY_DIR']) / "
        "f\"rank-{os.environ['SGW01_D1_RANK']}.json\"\n"
        "p.write_text(json.dumps({'rank': int(os.environ['SGW01_D1_RANK']), "
        "'source_commit': os.environ['SGW01_D1_SOURCE_COMMIT'], "
        "'checkpoint_revision': os.environ['SGW01_D1_CHECKPOINT_REVISION']}))\n"
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
    with pytest.raises(AdapterError, match="exited during startup"):
        lifecycle.start()
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
    with pytest.raises(AdapterError, match="timed out"):
        lifecycle.start()
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
