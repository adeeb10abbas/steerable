"""Owned finite lifecycle for the non-zero DreamZero ranks.

Rank zero remains the process launched by ``runtime.py`` and is the only
process allowed to bind the SGW loopback HTTP listener.  This helper owns only
the explicitly configured rank-worker children, records their identities, and
cleans them up on every exit path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
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


def _json_sha(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def _set_parent_death_signal() -> None:
    if os.name != "posix":
        return
    try:
        import ctypes
        import signal as _signal

        libc = ctypes.CDLL(None)
        libc.prctl(1, int(_signal.SIGTERM), 0, 0, 0)
    except (AttributeError, OSError):
        return


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
    if worker.process.poll() is not None:
        return
    observed = _proc_start_time(worker.process.pid)
    if observed != worker.start_time:
        raise AdapterError(f"rank {worker.rank} process identity changed during cleanup")
    try:
        os.killpg(os.getpgid(worker.process.pid), sig)
    except ProcessLookupError:
        return
    except OSError as exc:
        raise AdapterError(f"rank {worker.rank} cleanup signal failed") from exc


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
        native_config: Mapping[str, object] | None = None,
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
        self.native_config = dict(native_config or {})
        self.run_nonce = f"d1-{os.getpid()}-{time.time_ns()}"
        self.master_addr = master_addr
        self.master_port = int(master_port)
        self.startup_timeout = startup_timeout
        self.shutdown_timeout = shutdown_timeout
        self.workers: list[RankWorker] = []
        self._stopped = False
        self._previous_handlers: dict[int, object] = {}

    def configure_rank_zero(self) -> None:
        os.environ.update({
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": str(self.world_size),
            "MASTER_ADDR": self.master_addr,
            "MASTER_PORT": str(self.master_port),
            "SGW01_D1_RUN_NONCE": self.run_nonce,
        })

    def install_signal_cleanup(self) -> None:
        def handle(signum: int, frame: object) -> None:
            del frame
            self.stop()
            raise SystemExit(128 + signum)

        for signum in (signal.SIGTERM, signal.SIGINT):
            self._previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handle)

    def start(self) -> None:
        self.configure_rank_zero()
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
                    "SGW01_D1_RUN_NONCE": self.run_nonce,
                })
                process = subprocess.Popen(
                    list(self.worker_argv),
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    env=env,
                    start_new_session=True,
                    preexec_fn=_set_parent_death_signal if os.name == "posix" else None,
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
        except Exception:
            self.stop()
            raise

    def await_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        expected = set(range(1, self.world_size))
        workers_by_rank = {worker.rank: worker for worker in self.workers}
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
                    and payload.get("rank") in workers_by_rank
                    and payload.get("run_nonce") == self.run_nonce
                    and payload.get("pid") == workers_by_rank[payload["rank"]].process.pid
                    and payload.get("start_time") == workers_by_rank[payload["rank"]].start_time
                    and payload.get("source_commit") == self.source_commit
                    and payload.get("checkpoint_revision") == self.checkpoint_revision
                    and isinstance(payload.get("native_config"), Mapping)
                    and payload["native_config"].get("num_inference_steps") == 16
                    and payload["native_config"].get("seed") == 1140
                    and payload["native_config"].get("cfg_scale") == 5.0
                    and payload["native_config"].get("action_output_dim") == 8
                    and payload.get("native_config_sha256")
                    == _json_sha(payload["native_config"])
                ):
                    ready.add(payload["rank"])
            if ready == expected:
                return
            time.sleep(0.05)
        missing = sorted(expected - ready)
        raise AdapterError(f"D1 rank worker startup timed out; missing ready ranks {missing}")

    def assert_healthy(self) -> None:
        for worker in self.workers:
            if worker.process.poll() is not None:
                raise AdapterError(
                    f"D1 rank {worker.rank} exited while server was live; see {worker.stderr_path}"
                )

    def stop(self) -> None:
        if self._stopped:
            return
        workers = list(self.workers)
        self.workers.clear()
        errors: list[str] = []
        for worker in workers:
            if worker.process.poll() is None:
                try:
                    _kill_owned(worker, signal.SIGTERM)
                except AdapterError as exc:
                    errors.append(str(exc))
        deadline = time.monotonic() + self.shutdown_timeout
        for worker in workers:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                worker.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    _kill_owned(worker, signal.SIGKILL)
                except AdapterError as exc:
                    errors.append(str(exc))
                try:
                    worker.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    errors.append(f"rank {worker.rank} failed to reap after SIGKILL")
            if worker.process.poll() is None:
                errors.append(f"rank {worker.rank} remained alive during cleanup")
        self._stopped = True
        if errors:
            raise AdapterError("; ".join(errors))

    def __enter__(self) -> "OwnedD1RankLifecycle":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()
