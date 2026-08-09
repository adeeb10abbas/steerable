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
from typing import Any

from .episode_compiler import compile_episode, compile_pair
from .request0_replay import (
    AMENDMENT_SCHEMA,
    EVIDENCE_ENVELOPE_SCHEMA,
    load_amendment,
    validate_lane_preflight,
)
from .runtime_contract import (
    E004Cell,
    RuntimeContractError,
    canonical_json_sha256,
    load_runtime_bundle,
    sha256_file,
    shard_cells,
    validate_lane_release,
)


FIXED_SHARD_COUNTS = {
    "cosmos3_nano_policy_droid": 8,
    "pi05_current_stack_droid": 6,
    "cosmos3_edge_policy_droid": 2,
    "dreamzero_droid_action_cfg": 1,
}


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
        if value.get("row_count") != 1 or value.get("jsonl_sha256") != sha256_file(path):
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        except (OSError, IndexError, json.JSONDecodeError):
            continue
        if (
            row.get("request0_replay", {}).get("schema_version") == EVIDENCE_ENVELOPE_SCHEMA
            and isinstance(row.get("request0_pair_identity_sha256"), str)
        ):
            matches.append(path.resolve())
    _require(len(matches) <= 1, f"multiple valid behavioral episodes exist for {cell_root.name}")
    return matches[0] if matches else None


def _validate_existing_r001_artifacts(episode_path: Path) -> None:
    row = json.loads(Path(episode_path).read_text(encoding="utf-8").splitlines()[0])
    artifacts = row.get("request0_replay", {}).get("artifacts", {})
    required = {
        "amendment",
        "cache_manifest",
        "observation_cache",
        "reset_contract",
        "native_reset_contract",
        "attestation",
    }
    _require(isinstance(artifacts, dict) and required <= set(artifacts), "existing R001 artifact inventory is incomplete")
    for name in required:
        binding = artifacts[name]
        path = Path(str(binding.get("path"))) if isinstance(binding, dict) else Path("")
        _require(
            isinstance(binding, dict)
            and path.is_file()
            and binding.get("sha256") == sha256_file(path),
            f"existing R001 artifact is missing or changed: {name}",
        )


def _validate_resumed_left_cache(episode_path: Path, pair_root: Path) -> None:
    row = json.loads(Path(episode_path).read_text(encoding="utf-8").splitlines()[0])
    _require(row.get("requested_relation") == "left", "resumed request-zero source is not LEFT")
    artifacts = row.get("request0_replay", {}).get("artifacts", {})
    expected = {
        "observation_cache": pair_root / "left_request0_observation.npz",
        "cache_manifest": pair_root / "left_request0_observation.manifest.json",
        "reset_contract": pair_root / "left_request0_reset_contract.json",
    }
    for name, path in expected.items():
        binding = artifacts.get(name)
        _require(isinstance(binding, dict), f"resumed LEFT lacks bound {name}")
        _require(Path(str(binding.get("path"))).resolve() == path.resolve(), f"resumed LEFT {name} is outside exact pair root")
        _require(path.is_file() and binding.get("sha256") == sha256_file(path), f"resumed LEFT {name} is missing or changed")


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
    request0_pair_root: Path,
) -> list[str]:
    cache = request0_pair_root / "left_request0_observation.npz"
    cache_manifest = request0_pair_root / "left_request0_observation.manifest.json"
    reset_contract = request0_pair_root / "left_request0_reset_contract.json"
    mode = "capture_left" if cell.relation == "left" else "replay_right"
    native_reset_contract = reset_contract if cell.relation == "left" else attempt / "right_native_request0_reset_contract.json"
    attestation = (
        request0_pair_root / "left_request0_capture_attestation.json"
        if cell.relation == "left"
        else attempt / "right_request0_replay_attestation.json"
    )
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
        "--request0-replay-amendment",
        str(args.request0_replay_amendment.resolve()),
        "--request0-replay-amendment-sha256",
        args.request0_replay_amendment_sha256,
        "--request0-mode",
        mode,
        "--request0-observation-cache",
        str(cache),
        "--request0-observation-manifest",
        str(cache_manifest),
        "--request0-reset-contract",
        str(reset_contract),
        "--request0-native-reset-contract",
        str(native_reset_contract),
        "--request0-replay-attestation",
        str(attestation),
    ]
    if cell.relation == "right":
        for flag, path in (
            ("--request0-observation-cache-sha256", cache),
            ("--request0-observation-manifest-sha256", cache_manifest),
            ("--request0-reset-contract-sha256", reset_contract),
        ):
            _require(path.is_file(), f"matched LEFT request-zero artifact is missing before RIGHT: {path}")
            command.extend((flag, sha256_file(path)))
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
    request0_pair_root: Path,
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
        "request0_replay": {
            "amendment": {
                "path": str(args.request0_replay_amendment.resolve()),
                "sha256": args.request0_replay_amendment_sha256,
                "bytes": args.request0_replay_amendment.stat().st_size,
            },
            "mode": "capture_left" if cell.relation == "left" else "replay_right",
            "pair_root": str(request0_pair_root.resolve()),
        },
    }


def _left_first(cells: list[E004Cell]) -> list[E004Cell]:
    grouped: dict[str, dict[str, E004Cell]] = {}
    order: list[str] = []
    for cell in cells:
        if cell.matched_pair_id not in grouped:
            grouped[cell.matched_pair_id] = {}
            order.append(cell.matched_pair_id)
        grouped[cell.matched_pair_id][cell.relation] = cell
    output: list[E004Cell] = []
    for pair_id in order:
        pair = grouped[pair_id]
        _require(set(pair) == {"left", "right"}, f"selected queue has an incomplete matched pair: {pair_id}")
        output.extend((pair["left"], pair["right"]))
    return output


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
        if (
            row.get("matched_pair_id") == pair_id
            and row.get("request0_replay", {}).get("schema_version") == EVIDENCE_ENVELOPE_SCHEMA
        ):
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
    amendment = load_amendment(args.request0_replay_amendment, args.request0_replay_amendment_sha256)
    _require(amendment.get("schema_version") == AMENDMENT_SCHEMA, "request-zero amendment schema changed")
    for key, wanted in (
        ("registration_sha256", registration_sha256),
        ("queue_sha256", queue_sha256),
        ("candidate_sha256", candidate_sha256),
    ):
        _require(amendment.get(key) == wanted, f"request-zero amendment differs for {key}")
    bundle = load_runtime_bundle(
        registration_path=args.registration,
        registration_sha256=registration_sha256,
        queue_path=args.queue,
        queue_sha256=queue_sha256,
        candidate_path=args.candidate,
        candidate_sha256=candidate_sha256,
    )
    _require(args.model_id in FIXED_SHARD_COUNTS, f"unsupported DROID queue model: {args.model_id}")
    _require(
        args.shard_count == FIXED_SHARD_COUNTS[args.model_id],
        f"{args.model_id} shard_count is frozen at {FIXED_SHARD_COUNTS[args.model_id]}",
    )
    lane_release = validate_lane_release(
        args.lane_release,
        lane_release_sha256,
        bundle=bundle,
        model_id=args.model_id,
        lane_pod_uid=args.lane_pod_uid,
        lane_gpu_uuid=args.lane_gpu_uuid,
    )
    validate_lane_preflight(
        lane_release,
        amendment_sha256=args.request0_replay_amendment_sha256,
        model_id=args.model_id,
        lane_pod_uid=args.lane_pod_uid,
        lane_gpu_uuid=args.lane_gpu_uuid,
    )
    _require(_gpu_visible(args.lane_gpu_uuid) or args.skip_live_gpu_query_for_test, "lane GPU UUID is not live-visible")
    selected = _left_first(
        list(
            shard_cells(
                bundle.droid_new_cells(args.model_id),
                shard_index=args.shard_index,
                shard_count=args.shard_count,
            )
        )
    )
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
    pair_infrastructure: list[dict[str, Any]] = []
    failed_left_pairs: set[str] = set()
    shard_root.mkdir(parents=True, exist_ok=True)
    for cell in selected:
        if cell.relation == "right" and cell.matched_pair_id in failed_left_pairs:
            results.append(
                {
                    "cell_id": cell.cell_id,
                    "status": "not_launched_matched_left_infrastructure_invalid",
                    "matched_pair_id": cell.matched_pair_id,
                }
            )
            continue
        cell_root = shard_root / "cells" / _safe(cell.cell_id)
        request0_pair_root = shard_root / "request0_pairs" / _safe(cell.matched_pair_id)
        existing = _existing_valid_episode(cell_root)
        if existing is not None:
            try:
                _validate_existing_r001_artifacts(existing)
                if cell.relation == "left":
                    _validate_resumed_left_cache(existing, request0_pair_root)
            except BaseException as exc:
                if cell.relation == "left":
                    failed_left_pairs.add(cell.matched_pair_id)
                failure = {
                    "schema_version": "vla-wam-shared-v3e004-matched-pair-infrastructure-attempt-v1",
                    "matched_pair_id": cell.matched_pair_id,
                    "denominator_eligible": False,
                    "stage": "resume_request0_artifact_validation",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "retained_episode": str(existing),
                }
                failure_path = shard_root / "pair_infrastructure" / (
                    _safe(cell.cell_id) + ".json"
                )
                _write_new_json(failure_path, failure)
                pair_infrastructure.append(
                    {**failure, "path": str(failure_path), "sha256": sha256_file(failure_path)}
                )
                results.append(
                    {
                        "cell_id": cell.cell_id,
                        "status": "existing_episode_retained_but_r001_artifact_invalid",
                        "raw_episode": str(existing),
                        "failure": str(failure_path),
                    }
                )
                continue
            results.append({"cell_id": cell.cell_id, "status": "already_compiled", "raw_episode": str(existing)})
            if cell.relation == "right":
                pair = _pair_inputs(args.raw_root.resolve(), args.model_id, cell.matched_pair_id)
                pair_path = shard_root / "matched_pairs" / (_safe(cell.matched_pair_id) + ".jsonl")
                if pair is not None and not pair_path.exists():
                    try:
                        compile_pair(left_jsonl=pair[0], right_jsonl=pair[1], output=pair_path)
                    except BaseException as exc:
                        failure = {
                            "schema_version": "vla-wam-shared-v3e004-matched-pair-infrastructure-attempt-v1",
                            "matched_pair_id": cell.matched_pair_id,
                            "denominator_eligible": False,
                            "stage": "resume_pair_compile",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "left_episode": str(pair[0]),
                            "right_episode": str(pair[1]),
                        }
                        failure_path = shard_root / "pair_infrastructure" / (
                            _safe(cell.matched_pair_id) + "__compile.json"
                        )
                        _write_new_json(failure_path, failure)
                        pair_infrastructure.append(
                            {**failure, "path": str(failure_path), "sha256": sha256_file(failure_path)}
                        )
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
            request0_pair_root=request0_pair_root,
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
                request0_pair_root=request0_pair_root,
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
                    try:
                        compile_pair(left_jsonl=pair[0], right_jsonl=pair[1], output=pair_path)
                    except BaseException as exc:
                        failure = {
                            "schema_version": "vla-wam-shared-v3e004-matched-pair-infrastructure-attempt-v1",
                            "matched_pair_id": cell.matched_pair_id,
                            "denominator_eligible": False,
                            "stage": "pair_compile",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "left_episode": str(pair[0]),
                            "right_episode": str(pair[1]),
                        }
                        failure_path = attempt / "pair_infrastructure_invalid.json"
                        _write_new_json(failure_path, failure)
                        pair_infrastructure.append(
                            {**failure, "path": str(failure_path), "sha256": sha256_file(failure_path)}
                        )
        except BaseException as exc:
            failure = _infra_failure(cell=cell, attempt=attempt, stage="bridge_or_compile", exc=exc, attempt_manifest=attempt_manifest_path)
            infrastructure.append(failure)
            results.append({"cell_id": cell.cell_id, "status": "infrastructure_invalid_excluded_from_denominator", "failure": failure["path"]})
            if cell.relation == "left":
                failed_left_pairs.add(cell.matched_pair_id)
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
        "pair_infrastructure_invalid_count": len(pair_infrastructure),
        "pair_infrastructure_invalid_attempts": pair_infrastructure,
        "request0_replay_amendment": {
            "path": str(args.request0_replay_amendment.resolve()),
            "sha256": args.request0_replay_amendment_sha256,
            "bytes": args.request0_replay_amendment.stat().st_size,
        },
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
    parser.add_argument("--request0-replay-amendment", type=Path, required=True)
    parser.add_argument("--request0-replay-amendment-sha256", required=True)
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
