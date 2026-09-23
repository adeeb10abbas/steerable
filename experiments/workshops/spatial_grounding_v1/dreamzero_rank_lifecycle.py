"""Owned finite lifecycle for the non-zero DreamZero ranks.

Rank zero remains the process launched by ``runtime.py`` and is the only
process allowed to bind the SGW loopback HTTP listener.  This helper owns only
the explicitly configured rank-worker children, records their identities, and
cleans them up on every exit path.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Mapping, Sequence

from .adapters import AdapterError


@dataclass(frozen=True)
class RankWorker:
    rank: int
    process: subprocess.Popen[bytes]
    argv: tuple[str, ...]
    start_time: str
    stdout_path: Path
    stderr_path: Path


def _proc_start_time(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split(") ", 1)[-1].split()[19]
    except (OSError, IndexError):
        try:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "lstart="],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise AdapterError("rank worker start identity cannot be read") from exc
        value = result.stdout.strip()
        if not value:
            raise AdapterError("rank worker start identity cannot be read")
        return value


def _kill_owned(worker: RankWorker, sig: int) -> None:
    try:
        if _proc_start_time(worker.process.pid) != worker.start_time:
            return
        os.killpg(os.getpgid(worker.process.pid), sig)
    except (OSError, ProcessLookupError):
        return


class OwnedD1RankLifecycle:
    """Launch and reap only non-zero rank workers for one rank-zero server."""

    def __init__(
        self,
        *,
        worker_argv: Sequence[str],
        world_size: int,
        log_dir: Path,
        ready_dir: Path,
        source_commit: str,
        checkpoint_revision: str,
        master_addr: str = "127.0.0.1",
        master_port: int = 29591,
        startup_timeout: float = 30.0,
        shutdown_timeout: float = 10.0,
    ) -> None:
        if world_size < 2:
            raise AdapterError("D1 distributed lifecycle requires world_size >= 2")
        if not worker_argv or any(not isinstance(item, str) or not item for item in worker_argv):
            raise AdapterError("D1 rank worker argv must be non-empty strings")
        if not source_commit or not checkpoint_revision:
            raise AdapterError("D1 rank lifecycle requires pinned source and checkpoint identities")
        if master_addr not in {"127.0.0.1", "localhost", "::1"}:
            raise AdapterError("D1 rank workers require a loopback rendezvous address")
        self.worker_argv = tuple(worker_argv)
        self.world_size = world_size
        self.log_dir = Path(log_dir)
        self.ready_dir = Path(ready_dir)
        self.source_commit = source_commit
        self.checkpoint_revision = checkpoint_revision
        self.master_addr = master_addr
        self.master_port = int(master_port)
        self.startup_timeout = startup_timeout
        self.shutdown_timeout = shutdown_timeout
        self.workers: list[RankWorker] = []

    def start(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ready_dir.mkdir(parents=True, exist_ok=True)
        try:
            for rank in range(1, self.world_size):
                stdout_path = self.log_dir / f"rank-{rank}.stdout.log"
                stderr_path = self.log_dir / f"rank-{rank}.stderr.log"
                stdout = stdout_path.open("ab")
                stderr = stderr_path.open("ab")
                env = dict(os.environ)
                env.update({
                    "RANK": str(rank),
                    "LOCAL_RANK": str(rank),
                    "WORLD_SIZE": str(self.world_size),
                    "MASTER_ADDR": self.master_addr,
                    "MASTER_PORT": str(self.master_port),
                    "SGW01_D1_RANK": str(rank),
                    "SGW01_D1_RANK_READY_DIR": str(self.ready_dir),
                    "SGW01_D1_SOURCE_COMMIT": self.source_commit,
                    "SGW01_D1_CHECKPOINT_REVISION": self.checkpoint_revision,
                })
                process = subprocess.Popen(
                    list(self.worker_argv),
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    env=env,
                    start_new_session=True,
                )
                stdout.close()
                stderr.close()
                if process.pid is None:
                    raise AdapterError(f"rank {rank} did not expose a PID")
                self.workers.append(RankWorker(
                    rank=rank,
                    process=process,
                    argv=self.worker_argv,
                    start_time=_proc_start_time(process.pid),
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                ))
            self._await_ready()
        except Exception:
            self.stop()
            raise

    def _await_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        expected = set(range(1, self.world_size))
        while time.monotonic() < deadline:
            for worker in self.workers:
                if worker.process.poll() is not None:
                    raise AdapterError(
                        f"rank {worker.rank} exited during startup; see {worker.stderr_path}"
                    )
            ready = set()
            for path in self.ready_dir.glob("rank-*.json"):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if (
                    isinstance(payload, Mapping)
                    and isinstance(payload.get("rank"), int)
                    and payload.get("source_commit") == self.source_commit
                    and payload.get("checkpoint_revision") == self.checkpoint_revision
                ):
                    ready.add(payload["rank"])
            if ready == expected:
                return
            time.sleep(0.05)
        missing = sorted(expected - ready)
        raise AdapterError(f"D1 rank worker startup timed out; missing ready ranks {missing}")

    def stop(self) -> None:
        workers = list(self.workers)
        self.workers.clear()
        for worker in workers:
            if worker.process.poll() is None:
                _kill_owned(worker, signal.SIGTERM)
        deadline = time.monotonic() + self.shutdown_timeout
        for worker in workers:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                worker.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                _kill_owned(worker, signal.SIGKILL)
                try:
                    worker.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass

    def __enter__(self) -> "OwnedD1RankLifecycle":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()
