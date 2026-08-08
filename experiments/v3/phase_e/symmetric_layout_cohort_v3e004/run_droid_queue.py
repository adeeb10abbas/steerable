"""Shard-safe V3-E004 DROID behavioral queue launcher.

One process executes one cell at a time.  Whole LEFT/RIGHT pairs are assigned
to the same deterministic shard, while separate ali-owned pods/GPU UUIDs can
run different shards concurrently.  An advisory GPU lock prevents two Isaac
processes on one mounted lane from being launched by this runner.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Iterator

from .episode_compiler import compile_episode, compile_pair
from .runtime_contract import (
    E004Cell,
    RuntimeContractError,
    canonical_json_sha256,
    load_runtime_bundle,
    sha256_file,
    shard_cells,
    validate_lane_release,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeContractError(message)


def _safe(value: str) -> str:
    return value.replace(":", "__").replace("/", "_")


def _write_new_json(path: Path, value: Any) -> None:
    _require(not path.exists(), f"refusing to overwrite retained evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _next_attempt(cell_root: Path) -> Path:
    number = 1
    while (cell_root / f"attempt{number:03d}").exists():
        number += 1
    return cell_root / f"attempt{number:03d}"


def _existing_valid_episode(cell_root: Path) -> Path | None:
    matches = []
    for path in cell_root.glob("attempt*/raw_episode.jsonl"):
        manifest = path.with_name(path.name + ".manifest.json")
        if not manifest.is_file():
            continue
        try:
            value = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("row_count") == 1 and value.get("jsonl_sha256") == sha256_file(path):
            matches.append(path.resolve())
    _require(len(matches) <= 1, f"multiple valid behavioral episodes exist for {cell_root.name}")
    return matches[0] if matches else None


def _gpu_visible(uuid: str) -> bool:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return uuid in {row.strip() for row in output.splitlines() if row.strip()}


class GpuProcessLock:
    def __init__(self, path: Path, gpu_uuid: str) -> None:
        self.path = Path(path).resolve()
        self.gpu_uuid = gpu_uuid
        self.handle: Any = None

    def __enter__(self) -> "GpuProcessLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeContractError(f"another Isaac process owns GPU lock {self.path}") from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps({"pid": os.getpid(), "gpu_uuid": self.gpu_uuid, "acquired_unix_s": time.time()}) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        return self

    def __exit__(self, *_: Any) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


def _bridge_command(
    *,
    args: argparse.Namespace,
    cell: E004Cell,
    attempt: Path,
    registration_sha256: str,
    queue_sha256: str,
    candidate_sha256: str,
    lane_release_sha256: str,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        args.bridge_module,
        "--study-root",
        str(args.repo_root.resolve()),
        "--registration",
        str(args.registration.resolve()),
        "--registration-sha256",
        registration_sha256,
        "--queue",
        str(args.queue.resolve()),
        "--queue-sha256",
        queue_sha256,
        "--candidate",
        str(args.candidate.resolve()),
        "--candidate-sha256",
        candidate_sha256,
        "--lane-release",
        str(args.lane_release.resolve()),
        "--lane-release-sha256",
        lane_release_sha256,
        "--cell-id",
        cell.cell_id,
        "--lane-pod-uid",
        args.lane_pod_uid,
        "--lane-gpu-uuid",
        args.lane_gpu_uuid,
        "--live-snapshot",
        str(attempt / "live_scene_snapshot.json"),
        "--live-gate",
        str(attempt / "live_scene_gate.json"),
        "--simulator-export",
        str(attempt / "simulator_export.json"),
        "--state-capture-dir",
        str(attempt / "state_capture"),
        "--action-trace-dir",
        str(attempt / "action_traces"),
        "--future-trace-dir",
        str(attempt / "future_traces"),
        "--output-dir",
        str(attempt / "simulator"),
        "--model-endpoint-host",
        args.model_endpoint_host,
        "--model-endpoint-port",
        str(args.model_endpoint_port),
    ]
    for value in args.bridge_arg:
        command.append(value)
    return command


def _attempt_manifest(
    *,
    cell: E004Cell,
    args: argparse.Namespace,
    registration_sha256: str,
    queue_sha256: str,
    candidate_sha256: str,
    lane_release_sha256: str,
    command: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": "vla-wam-shared-v3e004-droid-attempt-manifest-v1",
        "study_id": cell.row["study_id"],
        "amendment_id": cell.row["amendment_id"],
        "registered_cell_id": cell.cell_id,
        "registered_cell_sha256": cell.row_sha256,
        "model_id": cell.model_id,
        "arena": cell.row["arena"],
        "execution_mode": cell.row["execution_mode"],
        "registration_sha256": registration_sha256,
        "queue_sha256": queue_sha256,
        "candidate_sha256": candidate_sha256,
        "lane_release_sha256": lane_release_sha256,
        "lane_pod_uid": args.lane_pod_uid,
        "lane_gpu_uuid": args.lane_gpu_uuid,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "bridge_module": args.bridge_module,
        "invocation_argv": command,
        "created_unix_s": time.time(),
        "model_request_count_at_manifest_write": 0,
        "behavioral_episode_count_at_manifest_write": 0,
    }


def _infra_failure(*, cell: E004Cell, attempt: Path, stage: str, exc: BaseException, attempt_manifest: Path) -> dict[str, Any]:
    value = {
        "schema_version": "vla-wam-shared-v3e004-infrastructure-attempt-v1",
        "record_type": "infrastructure_attempt",
        "behavioral_result_valid": False,
        "denominator_eligible": False,
        "study_id": cell.row["study_id"],
        "amendment_id": cell.row["amendment_id"],
        "registered_cell_id": cell.cell_id,
        "registered_cell_sha256": cell.row_sha256,
        "model_id": cell.model_id,
        "arena": cell.row["arena"],
        "environment_seed": cell.environment_seed,
        "sampling_seed": cell.sampling_seed,
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        "attempt_manifest": {"path": str(attempt_manifest.resolve()), "sha256": sha256_file(attempt_manifest), "bytes": attempt_manifest.stat().st_size},
        "retained_attempt_root": str(attempt.resolve()),
        "missing_measurement_policy": "Infrastructure-invalid attempts are outside behavioral denominators; no behavioral field is imputed.",
    }
    path = attempt / "infrastructure_invalid.json"
    _write_new_json(path, value)
    return {**value, "path": str(path), "sha256": sha256_file(path)}


def _pair_inputs(raw_root: Path, model_id: str, pair_id: str) -> tuple[Path, Path] | None:
    found: dict[str, Path] = {}
    model_root = raw_root / model_id
    for path in model_root.glob("shard-*-of-*/cells/*/attempt*/raw_episode.jsonl"):
        try:
            row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        except (OSError, IndexError, json.JSONDecodeError):
            continue
        if row.get("matched_pair_id") == pair_id:
            relation = row.get("requested_relation")
            _require(relation in {"left", "right"} and relation not in found, f"duplicate valid pair direction for {pair_id}")
            found[relation] = path.resolve()
    return (found["left"], found["right"]) if set(found) == {"left", "right"} else None


def run(args: argparse.Namespace) -> dict[str, Any]:
    registration_sha256 = sha256_file(args.registration)
    queue_sha256 = sha256_file(args.queue)
    candidate_sha256 = sha256_file(args.candidate)
    lane_release_sha256 = sha256_file(args.lane_release)
    for observed, expected, label in (
        (registration_sha256, args.registration_sha256, "registration"),
        (queue_sha256, args.queue_sha256, "queue"),
        (candidate_sha256, args.candidate_sha256, "candidate"),
        (lane_release_sha256, args.lane_release_sha256, "lane release"),
    ):
        _require(observed == expected, f"{label} digest mismatch")
    bundle = load_runtime_bundle(
        registration_path=args.registration,
        registration_sha256=registration_sha256,
        queue_path=args.queue,
        queue_sha256=queue_sha256,
        candidate_path=args.candidate,
        candidate_sha256=candidate_sha256,
    )
    validate_lane_release(
        args.lane_release,
        lane_release_sha256,
        bundle=bundle,
        model_id=args.model_id,
        lane_pod_uid=args.lane_pod_uid,
        lane_gpu_uuid=args.lane_gpu_uuid,
    )
    _require(_gpu_visible(args.lane_gpu_uuid) or args.skip_live_gpu_query_for_test, "lane GPU UUID is not live-visible")
    selected = list(shard_cells(bundle.droid_new_cells(args.model_id), shard_index=args.shard_index, shard_count=args.shard_count))
    if args.limit is not None:
        _require(args.limit >= 0, "limit must be nonnegative")
        # Limits preserve pair integrity.
        pair_limit = args.limit // 2
        selected = selected[: pair_limit * 2]
    shard_root = args.raw_root.resolve() / args.model_id / f"shard-{args.shard_index:03d}-of-{args.shard_count:03d}"
    shard_manifest_path = shard_root / "shard_manifest.json"
    _require(not shard_manifest_path.exists(), f"shard already closed: {shard_manifest_path}")
    results: list[dict[str, Any]] = []
    infrastructure: list[dict[str, Any]] = []
    shard_root.mkdir(parents=True, exist_ok=True)
    for cell in selected:
        cell_root = shard_root / "cells" / _safe(cell.cell_id)
        existing = _existing_valid_episode(cell_root)
        if existing is not None:
            results.append({"cell_id": cell.cell_id, "status": "already_compiled", "raw_episode": str(existing)})
            continue
        attempt = _next_attempt(cell_root)
        attempt.mkdir(parents=True)
        command = _bridge_command(
            args=args,
            cell=cell,
            attempt=attempt,
            registration_sha256=registration_sha256,
            queue_sha256=queue_sha256,
            candidate_sha256=candidate_sha256,
            lane_release_sha256=lane_release_sha256,
        )
        attempt_manifest_path = attempt / "attempt_manifest.json"
        _write_new_json(
            attempt_manifest_path,
            _attempt_manifest(
                cell=cell,
                args=args,
                registration_sha256=registration_sha256,
                queue_sha256=queue_sha256,
                candidate_sha256=candidate_sha256,
                lane_release_sha256=lane_release_sha256,
                command=command,
            ),
        )
        try:
            with GpuProcessLock(args.gpu_lock_file, args.lane_gpu_uuid):
                subprocess.run(command, cwd=args.repo_root.resolve(), check=True)
            export_path = attempt / "simulator_export.json"
            _require(export_path.is_file(), "bridge completed without canonical simulator export")
            result = compile_episode(
                bundle=bundle,
                export_path=export_path,
                export_sha256=sha256_file(export_path),
                output=attempt / "raw_episode.jsonl",
            )
            results.append({"cell_id": cell.cell_id, "status": "compiled_valid_behavioral_cell", **result})
            pair = _pair_inputs(args.raw_root.resolve(), args.model_id, cell.matched_pair_id)
            if pair is not None:
                pair_path = shard_root / "matched_pairs" / (_safe(cell.matched_pair_id) + ".jsonl")
                if not pair_path.exists():
                    compile_pair(left_jsonl=pair[0], right_jsonl=pair[1], output=pair_path)
        except BaseException as exc:
            failure = _infra_failure(cell=cell, attempt=attempt, stage="bridge_or_compile", exc=exc, attempt_manifest=attempt_manifest_path)
            infrastructure.append(failure)
            results.append({"cell_id": cell.cell_id, "status": "infrastructure_invalid_excluded_from_denominator", "failure": failure["path"]})
            if args.fail_fast:
                break
    shard_manifest = {
        "schema_version": "vla-wam-shared-v3e004-droid-shard-manifest-v1",
        "study_id": bundle.registration["study_id"],
        "amendment_id": bundle.registration["amendment_id"],
        "model_id": args.model_id,
        "arena": "droid_robolab",
        "registration_sha256": registration_sha256,
        "queue_sha256": queue_sha256,
        "candidate_sha256": candidate_sha256,
        "lane_release_sha256": lane_release_sha256,
        "lane_pod_uid": args.lane_pod_uid,
        "lane_gpu_uuid": args.lane_gpu_uuid,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "selected_cell_count": len(selected),
        "selected_pair_count": len(selected) // 2,
        "selected_cell_ids": [cell.cell_id for cell in selected],
        "selected_cells_sha256": canonical_json_sha256([cell.row for cell in selected]),
        "results": results,
        "behavioral_valid_count": sum(row["status"] in {"compiled_valid_behavioral_cell", "already_compiled"} for row in results),
        "infrastructure_invalid_count": len(infrastructure),
        "infrastructure_invalid_attempts": infrastructure,
        "closed_unix_s": time.time(),
    }
    _write_new_json(shard_manifest_path, shard_manifest)
    return {**shard_manifest, "shard_manifest": str(shard_manifest_path), "shard_manifest_sha256": sha256_file(shard_manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--registration-sha256", required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--queue-sha256", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--lane-release", type=Path, required=True)
    parser.add_argument("--lane-release-sha256", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--bridge-module", required=True)
    parser.add_argument("--bridge-arg", action="append", default=[])
    parser.add_argument("--model-endpoint-host", required=True)
    parser.add_argument("--model-endpoint-port", type=int, required=True)
    parser.add_argument("--lane-pod-uid", required=True)
    parser.add_argument("--lane-gpu-uuid", required=True)
    parser.add_argument("--gpu-lock-file", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--skip-live-gpu-query-for-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    print(json.dumps(run(args), allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
