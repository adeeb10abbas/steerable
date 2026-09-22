"""Finite, durable SGW-01 partition worker; production adapters are fail-closed."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time
from typing import Any, Callable, Mapping, Protocol

from .contract import Cell, ContractError, Release, load_release, validate_stage_authorizations, verify_completion_pointer
from .recorder import AttemptRecorder, atomic_json, next_attempt_number, utc_now

EXIT_RELEASE_INVALID, EXIT_ATTEMPTS_EXHAUSTED, EXIT_STORAGE_BUDGET_BLOCKED = 42, 43, 44
STOP_REQUESTED = False
STOP_GRACE_EXPIRED = False


class DeadlineExceeded(RuntimeError):
    pass


class Adapter(Protocol):
    def reset(self, cell: Cell, recorder: AttemptRecorder) -> dict[str, Any]: ...
    def run_episode(self, cell: Cell, recorder: AttemptRecorder, reset: dict[str, Any]) -> dict[str, Any]: ...
    def close(self) -> None: ...


ScoreFn = Callable[[Mapping[str, Any], Cell], Mapping[str, Any]]


def _stop(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    # Do not kill a completed attempt prematurely, but do not permit a hung
    # runtime to survive Kubernetes' documented 180-second termination grace.
    signal.signal(signal.SIGALRM, _grace_expired)
    signal.setitimer(signal.ITIMER_REAL, 180)


def _grace_expired(_signum: int, _frame: Any) -> None:
    global STOP_GRACE_EXPIRED
    STOP_GRACE_EXPIRED = True
    raise DeadlineExceeded("SIGTERM grace period expired")


def _run_with_deadline(seconds: int, operation: str, callback):
    if seconds <= 0:
        raise ContractError(f"{operation} deadline must be positive")
    previous = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, lambda _signum, _frame: (_ for _ in ()).throw(DeadlineExceeded(f"{operation} deadline exceeded")))
        signal.setitimer(signal.ITIMER_REAL, seconds)
        return callback()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@contextmanager
def model_lock(release: Release, model: str):
    path = release.root.parent / "locks" / f"{model}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError(f"global model lock held: {path}") from exc
        stream.write(json.dumps({"pid": os.getpid(), "release_id": release.release_id, "started_at_utc": utc_now()}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def load_adapter(model: str) -> Adapter:
    """Load only the production adapter supplied by the runtime owner."""
    module_name = f"experiments.workshops.spatial_grounding_v1.adapters"
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, "load_production_adapter")
        return factory(model)
    except (ImportError, AttributeError) as exc:
        raise ContractError("production adapter is unavailable; fake adapters are tests-only") from exc


def _completion(release: Release, cell: Cell) -> bool:
    path = release.root.parent / "cells" / f"{cell.cell_id}.complete.json"
    if not path.exists():
        return False
    verify_completion_pointer(release, path)
    return True


def _space_check(release: Release) -> None:
    required = int(release.binding.get("minimum_free_bytes", 100 * 1024**3))
    free = shutil.disk_usage(release.root.parent).free
    if free < required:
        raise OSError(f"persistent storage blocked: {free} < {required} free bytes")


def _status(release: Release, worker_id: str, **values: Any) -> None:
    atomic_json(release.root.parent / "status" / f"{worker_id}.json", {
        "schema_version": "sgw-01-worker-status-v1", "release_id": release.release_id,
        "worker_id": worker_id, "updated_at_utc": utc_now(), **values,
    })


class _Heartbeat:
    def __init__(self, release: Release, worker_id: str, seconds: int):
        if seconds <= 0:
            raise ContractError("heartbeat interval must be positive")
        self.release = release
        self.worker_id = worker_id
        self.seconds = seconds
        self.current_cell: str | None = None
        self.valid = 0
        self.last_error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="sgw-heartbeat", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.seconds + 1)

    def _run(self) -> None:
        while not self._stop.wait(self.seconds):
            _status(self.release, self.worker_id, state="running", current_cell=self.current_cell,
                    valid=self.valid, last_error=self.last_error, heartbeat=True)


def _stage_authorized(release: Release, stage: str) -> None:
    receipt = json.loads((release.root / "release_receipt.json").read_text(encoding="utf-8"))
    authorizations = validate_stage_authorizations(receipt.get("stage_authorizations"), stage)
    if stage not in authorizations:
        raise ContractError(f"stage {stage} lacks a release authorization receipt")


def _load_scorer() -> ScoreFn:
    """Resolve the scorer lazily so production cannot fall back to a fake."""
    try:
        from .scoring import FrozenScoringConfig, GoalSpec, canonical_status, score_episode
    except ImportError as exc:
        raise ContractError("production scorer is unavailable") from exc

    def score(trace: Mapping[str, Any], cell: Cell) -> Mapping[str, Any]:
        goal = GoalSpec(
            family=cell.family, physical_goal_sign=int(cell.row["physical_goal_sign"]),
            form=str(cell.row["form"]),
        )
        scored = score_episode(trace, goal, FrozenScoringConfig())
        return {"status": canonical_status(scored), "score": scored}
    return score


def _canonical_outcome(outcome: Mapping[str, Any], cell: Cell, scorer: ScoreFn | None) -> dict[str, Any]:
    """Scorer, not adapter termination labels, decides every model denominator."""
    value = dict(outcome)
    if value.get("status") == "technical_invalid":
        if not isinstance(value.get("technical_cause"), str):
            raise ContractError("technical invalid execution lacks a technical cause")
        return value
    trace = value.get("episode_mapping")
    if not isinstance(trace, Mapping):
        raise ContractError("nontechnical execution lacks the raw episode mapping required for scoring")
    scored = dict((scorer or _load_scorer())(trace, cell))
    status = scored.get("status")
    if status not in {"valid_success", "valid_model_failure", "censored", "technical_invalid"}:
        raise ContractError("scorer did not emit a canonical SGW status")
    # A normal 450-action endpoint is scored; censoring is only a physical
    # safety truncation, never an adapter convenience label.
    if status == "censored" and value.get("safety_terminated") is not True:
        raise ContractError("only explicit physical safety truncation may be censored")
    value.update(scored)
    value["status"] = status
    return value


def run_partition(release: Release, *, model: str, family: str, stage: str, max_valid: int,
                  max_attempts: int, worker_id: str, adapter: Adapter | None = None,
                  heartbeat_seconds: int = 60, scorer: ScoreFn | None = None) -> int:
    cells = release.partition(model, family, stage)
    if max_valid != len(cells) or max_attempts != 3:
        raise ContractError("partition limits must equal the frozen stage ceiling and three total attempts")
    _stage_authorized(release, stage)
    request_deadline = int(release.binding.get("request_deadline_seconds", 300))
    episode_deadline = int(release.binding.get("episode_deadline_seconds", 900))
    valid = 0
    heartbeat = _Heartbeat(release, worker_id, heartbeat_seconds)
    heartbeat.start()
    try:
        with model_lock(release, model):
            # The global lock must cover model-server/factory construction, not
            # merely requests, so a duplicate Job cannot load a second policy.
            adapter = adapter or load_adapter(model)
            for cell in cells:
                heartbeat.current_cell = cell.cell_id
                heartbeat.valid = valid
                _space_check(release)
                if _completion(release, cell):
                    valid += 1
                    continue
                while not STOP_REQUESTED:
                    number = next_attempt_number(release, cell)
                    if number > max_attempts:
                        _status(release, worker_id, state="blocked", cell_id=cell.cell_id,
                                reason="technical attempts exhausted", valid=valid)
                        return EXIT_ATTEMPTS_EXHAUSTED
                    recorder = AttemptRecorder(release, cell, f"attempt-{number:03d}")
                    recorder.begin()
                    try:
                        reset = _run_with_deadline(request_deadline, "reset", lambda: adapter.reset(cell, recorder))
                        if not isinstance(reset, dict) or not reset.get("full_reset"):
                            raise ContractError("adapter did not attest a full reset")
                        outcome = _run_with_deadline(
                            episode_deadline, "episode", lambda: adapter.run_episode(cell, recorder, reset)
                        )
                        outcome = _canonical_outcome(outcome, cell, scorer)
                        published = recorder.complete(outcome)
                    except DeadlineExceeded as exc:
                        heartbeat.last_error = f"{type(exc).__name__}: {exc}"
                        recorder.event("technical_invalid", error_type=type(exc).__name__, error=str(exc))
                        published = recorder.complete({"status": "technical_invalid", "technical_cause": heartbeat.last_error})
                    except ContractError:
                        raise
                    except OSError as exc:
                        heartbeat.last_error = f"{type(exc).__name__}: {exc}"
                        recorder.event("technical_invalid", error_type=type(exc).__name__, error=str(exc))
                        published = recorder.complete({"status": "technical_invalid", "technical_cause": heartbeat.last_error})
                    except Exception as exc:
                        recorder.event("fatal_unclassified_error", error_type=type(exc).__name__, error=str(exc))
                        raise ContractError(f"unclassified adapter failure; attempt preserved: {type(exc).__name__}") from exc
                    _status(release, worker_id, state="running", current_cell=cell.cell_id, valid=valid,
                            attempts=number, bytes_written=sum(p.stat().st_size for p in recorder.path.rglob("*") if p.is_file()))
                    if published or _completion(release, cell):
                        valid += 1
                        break
                if STOP_REQUESTED:
                    _status(release, worker_id, state="interrupted", current_cell=cell.cell_id, valid=valid)
                    return EXIT_ATTEMPTS_EXHAUSTED
        _status(release, worker_id, state="complete", valid=valid, expected=len(cells))
        return 0
    finally:
        heartbeat.close()
        close = getattr(adapter, "close", None) if adapter is not None else None
        if callable(close):
            close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--model", choices=("N3", "D1"), required=True)
    parser.add_argument("--family", choices=("LAT", "HEIGHT", "DIST"), required=True)
    parser.add_argument("--stage", choices=("P", "D", "C"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-valid-episodes", type=int, required=True)
    parser.add_argument("--max-cell-attempts", type=int, default=3)
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _stop)
    release = load_release(args.release)
    try:
        code = run_partition(release, model=args.model, family=args.family, stage=args.stage,
                             max_valid=args.max_valid_episodes, max_attempts=args.max_cell_attempts,
                             worker_id=f"{args.model}-{args.family}-{args.stage}-{os.getpid()}",
                             heartbeat_seconds=args.heartbeat_seconds)
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        code = EXIT_STORAGE_BUDGET_BLOCKED
    except ContractError as exc:
        print(str(exc), file=sys.stderr)
        code = EXIT_RELEASE_INVALID
    raise SystemExit(code)


if __name__ == "__main__":
    main()
