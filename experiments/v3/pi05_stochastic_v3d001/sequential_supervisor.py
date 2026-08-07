#!/usr/bin/env python3
"""Fail-closed sequential fallback for the three V3-D001 whole-seed lanes.

Lane 0 may already be running when this supervisor starts.  The supervisor
does not infer completion from a process exit: every released raw cell and
matched-pair manifest assigned to a lane must validate before the next lane is
started.  This is the fallback used when the two additional ali-owned RTX
pods remain unschedulable.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from experiments.v3.pi05_stochastic_v3d001.contract import (
    ContractError,
    RELEASE_MANIFEST_SHA256,
    cells_for_lane,
    load_release,
    sha256_file,
    validate_runtime,
)
from experiments.v3.pi05_stochastic_v3d001.queue import cell_plan, compile_completed_pair


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _process_state(pid: int) -> str:
    stat = Path(f"/proc/{pid}/stat")
    if stat.is_file():
        fields = stat.read_text(encoding="utf-8").split()
        return fields[2] if len(fields) >= 3 else "unknown"
    result = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True,
        text=True, check=False,
    )
    value = result.stdout.strip()
    return value[0] if result.returncode == 0 and value else "absent"


def _lane_cells(release: Any, lane_index: int, lane_count: int) -> list[Any]:
    return list(cells_for_lane(release.cells, lane_index, lane_count))


def _raw_paths(raw_root: Path, cells: list[Any], attempt_index: int) -> list[Path]:
    base = raw_root.resolve() / "V3-D001_pi05_nested_stochastic"
    return [
        base / cell.cell_id.replace(":", "__") / f"attempt{attempt_index:02d}" / "raw_episode.jsonl"
        for cell in cells
    ]


def _pair_paths(raw_root: Path, cells: list[Any], attempt_index: int) -> list[Path]:
    blocks = list(dict.fromkeys(cell.block_id for cell in cells))
    base = raw_root.resolve() / "V3-D001_pi05_nested_stochastic" / "matched_pairs" / f"attempt{attempt_index:02d}"
    return [base / (block.replace(":", "__") + ".json") for block in blocks]


def lane_progress(raw_root: Path, cells: list[Any], attempt_index: int) -> dict[str, int | bool]:
    raw = _raw_paths(raw_root, cells, attempt_index)
    pairs = _pair_paths(raw_root, cells, attempt_index)
    valid_raw = sum(path.is_file() and path.with_name(path.name + ".manifest.json").is_file() for path in raw)
    valid_pairs = sum(path.is_file() and path.with_name(path.name + ".manifest.json").is_file() for path in pairs)
    return {
        "expected_cells": len(raw),
        "valid_cells": valid_raw,
        "expected_pairs": len(pairs),
        "valid_pairs": valid_pairs,
        "complete": valid_raw == len(raw) and valid_pairs == len(pairs),
    }


def _validate_lane(
    *, repo_root: Path, release_manifest: Path, release: Any, runtime_identity: Path,
    phase_a_release_gate: Path, raw_root: Path, remote_host: str, remote_port: int,
    device: str, gpu_index: int, lane_pod_uid: str, lane_gpu_uuid: str,
    lane_index: int, lane_count: int, attempt_index: int,
) -> dict[str, int | bool]:
    cells = _lane_cells(release, lane_index, lane_count)
    for cell in cells:
        plan = cell_plan(
            repo_root=repo_root, release_manifest=release_manifest,
            runtime_identity=runtime_identity, phase_a_release_gate=phase_a_release_gate,
            raw_root=raw_root, cell=cell, remote_host=remote_host,
            remote_port=remote_port, device=device, gpu_index=gpu_index,
            lane_pod_uid=lane_pod_uid, lane_gpu_uuid=lane_gpu_uuid,
            attempt_index=attempt_index,
        )
        output = Path(plan["attempt_dir"]) / "raw_episode.jsonl"
        manifest = output.with_name(output.name + ".manifest.json")
        if not output.is_file() or not manifest.is_file():
            raise ContractError(f"lane {lane_index} missing released output: {output}")
        value = json.loads(manifest.read_text(encoding="utf-8"))
        if value.get("row_count") != 1 or value.get("jsonl_sha256") != sha256_file(output):
            raise ContractError(f"lane {lane_index} raw manifest mismatch: {output}")
    for block_id in dict.fromkeys(cell.block_id for cell in cells):
        result = compile_completed_pair(
            repo_root=repo_root, release_manifest=release_manifest, release=release,
            raw_root=raw_root, block_id=block_id, attempt_index=attempt_index,
        )
        if result is None:
            raise ContractError(f"lane {lane_index} missing matched pair: {block_id}")
    return lane_progress(raw_root, cells, attempt_index)


def _queue_command(args: argparse.Namespace, lane_index: int) -> list[str]:
    return [
        sys.executable, "-m", "experiments.v3.pi05_stochastic_v3d001.queue", "run-queue",
        "--repo-root", str(args.repo_root.resolve()),
        "--release-manifest", str(args.release_manifest.resolve()),
        "--runtime-identity", str(args.runtime_identity.resolve()),
        "--phase-a-release-gate", str(args.phase_a_release_gate.resolve()),
        "--raw-root", str(args.raw_root.resolve()),
        "--remote-host", args.remote_host, "--remote-port", str(args.remote_port),
        "--device", args.device, "--gpu-index", str(args.gpu_index),
        "--lane-pod-uid", args.lane_pod_uid,
        "--lane-gpu-uuid", args.lane_gpu_uuid,
        "--lane-index", str(lane_index), "--lane-count", str(args.lane_count),
        "--attempt-index", str(args.attempt_index),
    ]


def _external_lanes(values: list[str]) -> dict[int, int]:
    result: dict[int, int] = {}
    for value in values:
        try:
            lane_text, pid_text = value.split("=", 1)
            lane, pid = int(lane_text), int(pid_text)
        except (TypeError, ValueError) as exc:
            raise ValueError("external-lane-pid must be formatted LANE=PID") from exc
        if lane not in {1, 2} or pid < 1 or lane in result:
            raise ValueError("external lanes must be unique lane 1 or 2 with a positive PID")
        result[lane] = pid
    return result


def _progress_only_lanes(values: list[int], pid_lanes: dict[int, int]) -> set[int]:
    lanes = set(values)
    if any(type(lane) is not int or lane not in {1, 2} for lane in values):
        raise ValueError("progress-only external lanes must be lane 1 or 2")
    if len(lanes) != len(values) or lanes.intersection(pid_lanes):
        raise ValueError("external lanes must be unique across PID and progress-only modes")
    return lanes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--runtime-identity", type=Path, required=True)
    parser.add_argument("--phase-a-release-gate", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-port", type=int, default=8001)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--lane-pod-uid", required=True)
    parser.add_argument("--lane-gpu-uuid", required=True)
    parser.add_argument("--lane-count", type=int, default=3)
    parser.add_argument("--attempt-index", type=int, default=4)
    parser.add_argument("--existing-lane-zero-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--external-lane-pid", action="append", default=[])
    parser.add_argument("--external-lane-progress-only", action="append", type=int, default=[])
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.lane_count != 3 or args.poll_seconds <= 0:
        parser.error("V3-D001 fallback requires lane-count 3 and a positive poll interval")
    try:
        external_lanes = _external_lanes(args.external_lane_pid)
        progress_only_lanes = _progress_only_lanes(
            args.external_lane_progress_only, external_lanes,
        )
    except ValueError as exc:
        parser.error(str(exc))
    all_external_lanes = set(external_lanes).union(progress_only_lanes)
    release = load_release(args.repo_root, args.release_manifest)
    if sha256_file(args.release_manifest.resolve()) != RELEASE_MANIFEST_SHA256:
        raise ContractError("V3-D001 release manifest digest changed")
    validate_runtime(args.repo_root, args.runtime_identity, args.phase_a_release_gate)
    status: dict[str, Any] = {
        "schema_version": "vla-wam-shared-v3d001-sequential-supervisor-v1",
        "started_at_utc": _utc(), "attempt_index": args.attempt_index,
        "lane_count": args.lane_count, "state": "waiting_for_lane_0",
        "existing_lane_zero_pid": args.existing_lane_zero_pid,
    }
    _atomic_json(args.status_output, status)
    lane_zero = _lane_cells(release, 0, args.lane_count)
    while True:
        progress = lane_progress(args.raw_root, lane_zero, args.attempt_index)
        status.update({"updated_at_utc": _utc(), "lane_0": progress})
        _atomic_json(args.status_output, status)
        if progress["complete"]:
            _validate_lane(
                repo_root=args.repo_root, release_manifest=args.release_manifest,
                release=release, runtime_identity=args.runtime_identity,
                phase_a_release_gate=args.phase_a_release_gate, raw_root=args.raw_root,
                remote_host=args.remote_host, remote_port=args.remote_port,
                device=args.device, gpu_index=args.gpu_index,
                lane_pod_uid=args.lane_pod_uid, lane_gpu_uuid=args.lane_gpu_uuid,
                lane_index=0, lane_count=args.lane_count, attempt_index=args.attempt_index,
            )
            break
        state = _process_state(args.existing_lane_zero_pid)
        if state in {"absent", "Z"}:
            status.update({"state": "failed", "failed_at_utc": _utc(),
                           "error": f"lane 0 process state {state} before exact completion"})
            _atomic_json(args.status_output, status)
            raise RuntimeError(status["error"])
        time.sleep(args.poll_seconds)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    for lane_index in (1, 2):
        if lane_index in all_external_lanes:
            status.update({
                "state": f"lane_{lane_index}_externally_managed",
                f"lane_{lane_index}_external_monitor": (
                    {"mode": "local_pid", "pid": external_lanes[lane_index]}
                    if lane_index in external_lanes
                    else {"mode": "shared_pvc_progress_fail_closed"}
                ),
                "updated_at_utc": _utc(),
            })
            _atomic_json(args.status_output, status)
            continue
        status.update({"state": f"running_lane_{lane_index}", "updated_at_utc": _utc()})
        _atomic_json(args.status_output, status)
        log = args.log_dir / f"full_attempt{args.attempt_index:02d}_lane{lane_index}of3.log"
        with log.open("ab", buffering=0) as handle:
            completed = subprocess.run(
                _queue_command(args, lane_index), stdout=handle,
                stderr=subprocess.STDOUT, env=dict(os.environ), check=False,
            )
        if completed.returncode != 0:
            status.update({"state": "failed", "failed_at_utc": _utc(),
                           "failed_lane": lane_index, "returncode": completed.returncode,
                           "log": str(log)})
            _atomic_json(args.status_output, status)
            raise subprocess.CalledProcessError(completed.returncode, _queue_command(args, lane_index))
        status[f"lane_{lane_index}"] = _validate_lane(
            repo_root=args.repo_root, release_manifest=args.release_manifest,
            release=release, runtime_identity=args.runtime_identity,
            phase_a_release_gate=args.phase_a_release_gate, raw_root=args.raw_root,
            remote_host=args.remote_host, remote_port=args.remote_port,
            device=args.device, gpu_index=args.gpu_index,
            lane_pod_uid=args.lane_pod_uid, lane_gpu_uuid=args.lane_gpu_uuid,
            lane_index=lane_index, lane_count=args.lane_count,
            attempt_index=args.attempt_index,
        )
        _atomic_json(args.status_output, status)
    for lane_index in sorted(all_external_lanes):
        pid = external_lanes.get(lane_index)
        cells = _lane_cells(release, lane_index, args.lane_count)
        status.update({"state": f"waiting_for_external_lane_{lane_index}",
                       "updated_at_utc": _utc()})
        _atomic_json(args.status_output, status)
        while True:
            progress = lane_progress(args.raw_root, cells, args.attempt_index)
            status[f"lane_{lane_index}"] = progress
            status["updated_at_utc"] = _utc()
            _atomic_json(args.status_output, status)
            if progress["complete"]:
                break
            state = _process_state(pid) if pid is not None else None
            if state in {"absent", "Z"}:
                status.update({"state": "failed", "failed_at_utc": _utc(),
                               "failed_lane": lane_index,
                               "error": f"external lane {lane_index} process state {state} before exact completion"})
                _atomic_json(args.status_output, status)
                raise RuntimeError(status["error"])
            time.sleep(args.poll_seconds)
        status[f"lane_{lane_index}"] = _validate_lane(
            repo_root=args.repo_root, release_manifest=args.release_manifest,
            release=release, runtime_identity=args.runtime_identity,
            phase_a_release_gate=args.phase_a_release_gate, raw_root=args.raw_root,
            remote_host=args.remote_host, remote_port=args.remote_port,
            device=args.device, gpu_index=args.gpu_index,
            lane_pod_uid=args.lane_pod_uid, lane_gpu_uuid=args.lane_gpu_uuid,
            lane_index=lane_index, lane_count=args.lane_count,
            attempt_index=args.attempt_index,
        )
        _atomic_json(args.status_output, status)
    status.update({"state": "complete", "completed_at_utc": _utc()})
    _atomic_json(args.status_output, status)


if __name__ == "__main__":
    main()
