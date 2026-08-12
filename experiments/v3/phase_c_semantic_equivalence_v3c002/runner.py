"""Fail-closed serial-per-lane V3-C002 block launcher.

One lane owns one simulator, one independent π0.5 server, and every four-cell
seed block assigned to it.  A block is complete only after all four retained
raw episodes validate; a partial attempt is preserved and the whole block is
rerun into a new immutable attempt directory.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

from .contract import (
    ContractError,
    grouped_shard,
    require,
    require_released_gate,
    require_smoke_authorization,
    resolve_binding_path,
    sha256_file,
)
from .runtime import validate_runtime_manifest


class LaneLock:
    def __init__(self, path: Path, *, pod_uid: str, gpu_uuid: str) -> None:
        self.path, self.pod_uid, self.gpu_uuid = path, pod_uid, gpu_uuid
        self.handle: Any = None

    def __enter__(self) -> "LaneLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError(f"lane lock already owned: {self.path}") from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps({"pid": os.getpid(), "pod_uid": self.pod_uid, "gpu_uuid": self.gpu_uuid, "acquired_unix_s": time.time()}) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        return self

    def __exit__(self, *_: Any) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


def _plan(cells: Sequence[Any], *, shard_index: int, shard_count: int, args: Any) -> dict[str, Any]:
    return {
        "schema_version": "vla-wam-shared-v3c002-lane-plan-v2",
        "authorization_mode": args.authorization_mode,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "lane_id": args.lane_id,
        "simulator_pod_uid": args.lane_pod_uid,
        "simulator_gpu_uuid": args.lane_gpu_uuid,
        "policy_server_pod_uid": args.policy_server_pod_uid,
        "policy_server_gpu_uuid": args.policy_server_gpu_uuid,
        "server_port": args.server_port,
        "raw_root": args.raw_root,
        "container_identity": args.container_identity,
        "runtime_identity": args.runtime_identity,
        "server_process_identity": args.server_process_identity,
        "server_lock_identity": args.server_lock_identity,
        "execution_policy": "one serial client queue, one independent policy server, one independent simulator, and all four registered conditions from every seed block on this lane",
        "seed_blocks": sorted({cell.seed for cell in cells}),
        "cell_ids": [cell.cell_id for cell in cells],
    }


def _write_or_validate_plan(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        require(path.read_text(encoding="utf-8") == encoded, "existing lane plan differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")


def _append_infrastructure(ledger: Path, row: Mapping[str, Any]) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), allow_nan=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _raw_row(path: Path, *, cell: Any, mode: str) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    require(len(lines) == 1 and lines[0].strip(), f"raw episode is not one JSONL record: {path}")
    try:
        row = json.loads(lines[0], parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ContractError(f"raw episode is invalid JSON: {path}: {exc}") from exc
    require(isinstance(row, dict), "raw episode is not an object")
    require(row.get("schema_version") == "vla-wam-shared-v3c002-raw-episode-v1", "raw episode schema changed")
    require(row.get("cell_id") == cell.cell_id and row.get("cell_sha256") == cell.row_sha256, "raw episode cell binding changed")
    require(row.get("authorization_mode") == mode, "raw episode authorization mode changed")
    require(row.get("excluded_from_behavioral_denominators") is (mode == "excluded_smoke"), "raw episode denominator status changed")
    return row


def _next_attempt_root(root: Path, seed: int) -> Path:
    seed_root = root / f"seed{seed}"
    index = 1
    while (seed_root / f"attempt{index:03d}").exists():
        index += 1
    return seed_root / f"attempt{index:03d}"


def _completed_marker(root: Path, seed: int) -> Path:
    return root / f"seed{seed}" / "completed_block.json"


def _validate_completed(marker: Path, block: Sequence[Any], mode: str) -> None:
    value = json.loads(marker.read_text(encoding="utf-8"))
    require(value.get("schema_version") == "vla-wam-shared-v3c002-completed-block-v1", "completed block schema changed")
    require(value.get("authorization_mode") == mode, "completed block authorization changed")
    records = value.get("raw_episodes")
    require(isinstance(records, list) and len(records) == 4, "completed block lacks four raw episodes")
    by_cell = {cell.cell_id: cell for cell in block}
    for record in records:
        require(isinstance(record, dict) and record.get("cell_id") in by_cell, "completed block has an unknown cell")
        path = Path(str(record.get("path")))
        require(path.is_file() and record.get("sha256") == sha256_file(path), "completed raw episode changed")
        _raw_row(path, cell=by_cell[str(record["cell_id"])], mode=mode)


def _lane_record(gate: Mapping[str, Any], lane_id: str) -> dict[str, Any]:
    rows = []
    for binding in gate["lane_manifests"]:
        value = json.loads(resolve_binding_path(binding).read_text(encoding="utf-8"))
        if value.get("lane_id") == lane_id:
            rows.append(value)
    require(len(rows) == 1, "release gate does not contain exactly one manifest for this lane")
    return rows[0]


def _dispatch_block(*, block: Sequence[Any], args: Any, registration_sha: str, queue_sha: str) -> None:
    seed = block[0].seed
    root = Path(args.raw_root) / args.authorization_mode
    marker = _completed_marker(root, seed)
    if marker.exists():
        require(args.resume, f"seed {seed} already has a completed block")
        _validate_completed(marker, block, args.authorization_mode)
        return
    attempt = _next_attempt_root(root, seed)
    attempt.mkdir(parents=True, exist_ok=False)
    shared = attempt / "request0"
    cache = shared / "observation_cache.npz"
    manifest = shared / "observation_manifest.json"
    reset_contract = shared / "reset_contract.json"
    outputs: list[dict[str, Any]] = []
    ordered = sorted(block, key=lambda value: int(value.row["execution_order_index"]))
    for index, cell in enumerate(ordered):
        cell_root = attempt / f"cell{index:02d}_{cell.condition}"
        native_reset = reset_contract if index == 0 else cell_root / "native_reset_contract.json"
        replay_attestation = cell_root / "request0_attestation.json"
        command = list(args.adapter_command) + [
            "--registration", str(args.registration.resolve()), "--registration-sha256", registration_sha,
            "--queue", str(args.queue.resolve()), "--queue-sha256", queue_sha,
            "--authorization-gate", str(args.authorization_gate.resolve()), "--authorization-gate-sha256", sha256_file(args.authorization_gate),
            "--authorization-mode", args.authorization_mode,
            "--runtime-manifest", str(args.runtime_manifest.resolve()), "--runtime-manifest-sha256", args.runtime_manifest_sha256,
            "--cell-id", cell.cell_id,
            "--lane-pod-uid", args.lane_pod_uid, "--lane-gpu-uuid", args.lane_gpu_uuid,
            "--policy-server-pod-uid", args.policy_server_pod_uid, "--policy-server-gpu-uuid", args.policy_server_gpu_uuid,
            "--lane-id", args.lane_id, "--raw-root", args.raw_root,
            "--container-identity", args.container_identity, "--runtime-identity", args.runtime_identity,
            "--server-process-identity", args.server_process_identity, "--server-lock-identity", args.server_lock_identity,
            "--model-endpoint-port", str(args.server_port),
            "--live-snapshot", str(cell_root / "live_snapshot.json"),
            "--live-gate", str(cell_root / "live_gate.json"),
            "--simulator-export", str(cell_root / "raw_episode.jsonl"),
            "--raw-event-stream", str(cell_root / "request_events.jsonl"),
            "--state-capture-dir", str(cell_root / "state"),
            "--action-trace-dir", str(cell_root / "actions"),
            "--future-trace-dir", str(cell_root / "futures"),
            "--output-dir", str(cell_root / "robolab_output"),
            "--request0-mode", ("capture_block" if index == 0 else "replay_block"),
            "--request0-observation-cache", str(cache),
            "--request0-observation-manifest", str(manifest),
            "--request0-reset-contract", str(reset_contract),
            "--request0-native-reset-contract", str(native_reset),
            "--request0-replay-attestation", str(replay_attestation),
        ]
        if index:
            command.extend([
                "--request0-observation-cache-sha256", sha256_file(cache),
                "--request0-observation-manifest-sha256", sha256_file(manifest),
                "--request0-reset-contract-sha256", sha256_file(reset_contract),
            ])
        try:
            subprocess.run(command, check=True)
            raw_path = cell_root / "raw_episode.jsonl"
            _raw_row(raw_path, cell=cell, mode=args.authorization_mode)
            outputs.append({"cell_id": cell.cell_id, "path": str(raw_path.resolve()), "bytes": raw_path.stat().st_size, "sha256": sha256_file(raw_path)})
        except BaseException as exc:
            _append_infrastructure(Path(args.infrastructure_ledger), {
                "schema_version": "vla-wam-shared-v3c002-infrastructure-attempt-v1",
                "record_type": "infrastructure_attempt", "infrastructure_status": "infrastructure_invalid_excluded",
                "denominator_eligible": False, "authorization_mode": args.authorization_mode,
                "seed_block_id": cell.block_id, "cell_id": cell.cell_id, "attempt_root": str(attempt.resolve()),
                "completed_cell_ids_before_failure": [record["cell_id"] for record in outputs],
                "entire_partial_block_invalidated": True, "error_type": type(exc).__name__, "error": str(exc),
            })
            raise
    require(len(outputs) == 4, "refusing to mark a partial seed block complete")
    value = {
        "schema_version": "vla-wam-shared-v3c002-completed-block-v1",
        "status": "completed_excluded_smoke_block" if args.authorization_mode == "excluded_smoke" else "completed_behavioral_block",
        "authorization_mode": args.authorization_mode,
        "seed_block_id": block[0].block_id,
        "episode_seed": seed,
        "execution_order": [cell.condition for cell in ordered],
        "attempt_root": str(attempt.resolve()),
        "raw_episodes": outputs,
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--authorization-gate", type=Path, required=True)
    parser.add_argument("--authorization-mode", choices=("excluded_smoke", "behavioral"), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--lane-pod-uid", required=True)
    parser.add_argument("--lane-gpu-uuid", required=True)
    parser.add_argument("--policy-server-pod-uid", required=True)
    parser.add_argument("--policy-server-gpu-uuid", required=True)
    parser.add_argument("--server-port", type=int, required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--container-identity", required=True)
    parser.add_argument("--runtime-identity", required=True)
    parser.add_argument("--lane-id", required=True)
    parser.add_argument("--server-process-identity", required=True)
    parser.add_argument("--server-lock-identity", required=True)
    parser.add_argument("--lane-lock", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--infrastructure-ledger", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--adapter-command", nargs=argparse.REMAINDER, help="Exact C002 simulator adapter after --adapter-command")
    args = parser.parse_args()
    if args.authorization_mode == "behavioral":
        registration, cells, gate = require_released_gate(registration_path=args.registration, queue_path=args.queue, release_gate_path=args.authorization_gate)
        selected = grouped_shard(cells, shard_index=args.shard_index, shard_count=args.shard_count)
    else:
        require(args.shard_index == 0 and args.shard_count == 1, "excluded smoke is exactly one unsharded block")
        registration, selected, gate = require_smoke_authorization(registration_path=args.registration, queue_path=args.queue, authorization_path=args.authorization_gate)
    plan = _plan(selected, shard_index=args.shard_index, shard_count=args.shard_count, args=args)
    _write_or_validate_plan(args.plan_output, plan)
    if not args.execute:
        return
    require(bool(args.adapter_command), "--execute requires the exact C002 simulator adapter command")
    runtime_manifest = validate_runtime_manifest(
        args.runtime_manifest, args.runtime_manifest_sha256,
        registration_path=args.registration, queue_path=args.queue,
        pod_uid=args.lane_pod_uid, gpu_uuid=args.lane_gpu_uuid,
    )
    expected_identity = runtime_manifest["runtime_identity"]
    for key, expected in (
        ("simulator_pod_uid", args.lane_pod_uid), ("simulator_gpu_uuid", args.lane_gpu_uuid),
        ("policy_server_pod_uid", args.policy_server_pod_uid), ("policy_server_gpu_uuid", args.policy_server_gpu_uuid),
        ("server_port", args.server_port), ("raw_root", args.raw_root), ("container_identity", args.container_identity),
        ("runtime_identity", args.runtime_identity), ("lane_id", args.lane_id),
        ("server_process_identity", args.server_process_identity), ("server_lock_identity", args.server_lock_identity),
    ):
        require(expected_identity.get(key) == expected, f"lane/runtime identity differs for {key}")
    if args.authorization_mode == "behavioral":
        lane_record = _lane_record(gate, args.lane_id)
        for key in ("simulator_pod_uid", "simulator_gpu_uuid", "policy_server_pod_uid", "policy_server_gpu_uuid", "server_port", "raw_root", "container_identity", "runtime_identity", "server_process_identity", "server_lock_identity"):
            require(lane_record.get(key) == expected_identity.get(key), f"released lane differs for {key}")
    blocks = []
    for seed in sorted({cell.seed for cell in selected}):
        block = [cell for cell in selected if cell.seed == seed]
        require(len(block) == 4, "refusing to dispatch a partial seed block")
        blocks.append(block)
    with LaneLock(args.lane_lock, pod_uid=args.lane_pod_uid, gpu_uuid=args.lane_gpu_uuid):
        for block in blocks:
            _dispatch_block(block=block, args=args, registration_sha=sha256_file(args.registration), queue_sha=sha256_file(args.queue))


if __name__ == "__main__":
    main()
