"""Serial-per-lane V3-C002 launcher; a seed block is never split."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from .contract import ContractError, grouped_shard, require_released_gate, resolve_binding_path, sha256_file
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
        self.handle.seek(0); self.handle.truncate()
        self.handle.write(json.dumps({"pid": os.getpid(), "pod_uid": self.pod_uid, "gpu_uuid": self.gpu_uuid, "acquired_unix_s": time.time()}) + "\n")
        self.handle.flush(); os.fsync(self.handle.fileno())
        return self

    def __exit__(self, *_: Any) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


def _plan(cells: list[Any], *, shard_index: int, shard_count: int, args: Any) -> dict[str, Any]:
    return {
        "schema_version": "vla-wam-shared-v3c002-lane-plan-v1",
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
        "execution_policy": "one serial client queue, one independent policy server, one independent simulator, and all four conditions from every seed block on this lane",
        "seed_blocks": sorted({cell.seed for cell in cells}),
        "cell_ids": [cell.cell_id for cell in cells],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--release-gate", type=Path, required=True)
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
    parser.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--runtime-manifest-sha256")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--adapter-command", nargs=argparse.REMAINDER, help="External exact-runtime adapter after --execute")
    args = parser.parse_args()
    registration, cells, gate = require_released_gate(registration_path=args.registration, queue_path=args.queue, release_gate_path=args.release_gate)
    selected = grouped_shard(cells, shard_index=args.shard_index, shard_count=args.shard_count)
    plan = _plan(selected, shard_index=args.shard_index, shard_count=args.shard_count, args=args)
    if args.plan_output.exists():
        raise ContractError(f"refusing to overwrite lane plan: {args.plan_output}")
    args.plan_output.parent.mkdir(parents=True, exist_ok=True)
    args.plan_output.write_text(json.dumps(plan, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.execute:
        return
    if not args.adapter_command:
        raise ContractError("--execute requires an independent exact-runtime adapter command")
    if args.runtime_manifest is None or args.runtime_manifest_sha256 is None:
        raise ContractError("--execute requires a hash-bound exact-runtime lane manifest")
    runtime_manifest = validate_runtime_manifest(
        args.runtime_manifest,
        args.runtime_manifest_sha256,
        registration_path=args.registration,
        queue_path=args.queue,
        pod_uid=args.lane_pod_uid,
        gpu_uuid=args.lane_gpu_uuid,
    )
    lane_records = []
    for binding in gate["lane_manifests"]:
        value = json.loads(resolve_binding_path(binding).read_text(encoding="utf-8"))
        if value.get("lane_id") == args.lane_id:
            lane_records.append(value)
    require(len(lane_records) == 1, "release gate does not contain exactly one manifest for this lane")
    lane_record = lane_records[0]
    for key, expected in (
        ("simulator_pod_uid", args.lane_pod_uid), ("simulator_gpu_uuid", args.lane_gpu_uuid),
        ("policy_server_pod_uid", args.policy_server_pod_uid), ("policy_server_gpu_uuid", args.policy_server_gpu_uuid),
        ("server_port", args.server_port), ("raw_root", args.raw_root), ("container_identity", args.container_identity),
        ("runtime_identity", args.runtime_identity), ("server_process_identity", args.server_process_identity),
        ("server_lock_identity", args.server_lock_identity),
    ):
        require(lane_record.get(key) == expected and runtime_manifest["runtime_identity"].get(key) == expected, f"lane/runtime identity differs for {key}")
    with LaneLock(args.lane_lock, pod_uid=args.lane_pod_uid, gpu_uuid=args.lane_gpu_uuid):
        for seed in sorted({cell.seed for cell in selected}):
            block = [cell for cell in selected if cell.seed == seed]
            require(len(block) == 4, "refusing to dispatch a partial seed block")
            for cell in sorted(block, key=lambda value: int(value.row["execution_order_index"])):
                command = list(args.adapter_command) + [
                    "--registered-cell-id", cell.cell_id,
                    "--registered-cell-sha256", cell.row_sha256,
                    "--registration", str(args.registration.resolve()),
                    "--registration-sha256", sha256_file(args.registration),
                    "--queue", str(args.queue.resolve()),
                    "--queue-sha256", sha256_file(args.queue),
                    "--lane-pod-uid", args.lane_pod_uid,
                    "--lane-gpu-uuid", args.lane_gpu_uuid,
                    "--policy-server-pod-uid", args.policy_server_pod_uid,
                    "--policy-server-gpu-uuid", args.policy_server_gpu_uuid,
                    "--server-port", str(args.server_port),
                    "--raw-root", args.raw_root,
                    "--container-identity", args.container_identity,
                    "--runtime-identity", args.runtime_identity,
                    "--lane-id", args.lane_id,
                    "--server-process-identity", args.server_process_identity,
                    "--server-lock-identity", args.server_lock_identity,
                    "--runtime-manifest", str(args.runtime_manifest.resolve()),
                    "--runtime-manifest-sha256", args.runtime_manifest_sha256,
                ]
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
