#!/usr/bin/env python3
"""Plan/run V3-B002 in whole-seed lanes without overwriting evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from experiments.v3.pi05_phase_b.contract import (
    AuthorizedCell, ContractError, cells_for_lane, load_release_bundle,
    sha256_file,
)
from experiments.v3.pi05_phase_b.runtime import (
    validate_release_gate, validate_runtime_identity,
)
from tools.vla_wam_v3_episode_schema import parse_jsonl_record


FIXTURE_SHA256 = "e1799b815da41f9a08a4000a360c4958003269fed27e2abe75b273519e4d1c88"
FROZEN_VK_ICD = "/etc/vulkan/icd.d/nvidia_icd.json"


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


def _attempt(raw_root: Path, cell: AuthorizedCell) -> Path:
    return Path(raw_root).resolve()/"V3-B002_pi05_position_mirror"/cell.cell_id.replace(":", "__")/"attempt01"


def cell_plan(*, repo_root: Path, release_manifest: Path, release_manifest_sha256: str,
              runtime_manifest: Path, release_gate: Path, fixture_candidate: Path,
              raw_root: Path, cell: AuthorizedCell, remote_host: str, remote_port: int,
              device: str, gpu_index: int, lane_pod_uid: str, lane_gpu_uuid: str) -> dict[str, Any]:
    attempt = _attempt(raw_root, cell)
    root = Path(repo_root).resolve()
    common = [
        "--study-root", str(root), "--release-manifest", str(release_manifest.resolve()),
        "--release-manifest-sha256", release_manifest_sha256,
        "--runtime-manifest", str(runtime_manifest.resolve()),
        "--release-gate", str(release_gate.resolve()), "--cell-id", cell.cell_id,
        "--lane-pod-uid", lane_pod_uid, "--lane-gpu-uuid", lane_gpu_uuid,
        "--fixture-candidate", str(fixture_candidate.resolve()),
        "--fixture-candidate-sha256", FIXTURE_SHA256,
    ]
    bridge_worker = [
        sys.executable, str(root/"experiments/v3/pi05_phase_b/robolab_bridge.py"), *common,
        "--state-capture-dir", str(attempt/"state_capture"),
        "--action-trace-dir", str(attempt/"action_traces"),
        "--reset-attestation", str(attempt/"reset_attestation.json"),
        "--simulator-export", str(attempt/"simulator_export.json"),
        "--remote-host", remote_host, "--remote-port", str(remote_port),
        "--open-loop-horizon", "15", "--instruction-controller", "static",
        "--output-dir", str(attempt/"simulator"),
        "--output-folder-name", str(attempt/"simulator"),
        "--num-envs", "1", "--num-runs", "1", "--headless",
        "--renderer", "realtime", "--rendering-type", "balanced",
        "--device", device, "--kit_args=--/rtx/verifyDriverVersion/enabled=false",
        "--video-mode", "viewport", "--instruction-type", "default", "--disable-subtask",
    ]
    bridge = [
        sys.executable, str(root/"tools/native_process_group_thermal_guard.py"),
        "--launch", "--gpu-index", str(gpu_index),
        "--output", str(attempt/"thermal_events.jsonl"),
        "--ledger-output", str(attempt/f"runtime_interventions_{cell.row['model_id']}.json"),
        "--invalid-attempts-output", str(attempt/f"invalid_attempts_{cell.row['model_id']}.json"),
        "--model-id", cell.row["model_id"],
        "--pair-id", f"{cell.row['matched_block_id']}:{cell.arm}",
        "--environment-seed", str(cell.seed), "--sampling-seed", str(cell.seed),
        "--requested-relation", cell.relation, "--", *bridge_worker,
    ]
    compiler = [
        sys.executable, str(root/"experiments/v3/pi05_phase_b/compile_cell.py"),
        "--repo-root", str(root), "--release-manifest", str(release_manifest.resolve()),
        "--release-manifest-sha256", release_manifest_sha256,
        "--runtime-manifest", str(runtime_manifest.resolve()), "--cell-id", cell.cell_id,
        "--export", str(attempt/"simulator_export.json"),
        "--output-jsonl", str(attempt/"raw_episode.jsonl"),
    ]
    return {
        "cell_id": cell.cell_id, "seed": cell.seed, "arm": cell.arm,
        "relation": cell.relation,
        "execution_order_index_within_seed": cell.row["execution_order_index_within_seed"],
        "attempt_dir": str(attempt), "bridge_command": bridge,
        "guarded_worker_command": bridge_worker, "compiler_command": compiler,
        "environment": {
            "OMNI_KIT_ACCEPT_EULA": "YES", "NVIDIA_DRIVER_CAPABILITIES": "all",
            "VK_ICD_FILENAMES": FROZEN_VK_ICD,
            "XDG_CACHE_HOME": str(attempt/"cache/xdg"), "WARP_CACHE_PATH": str(attempt/"cache/warp"),
            "MPLCONFIGDIR": str(attempt/"cache/matplotlib"),
            # Isaac/Kit temporary files must be pod-local.  Prior runs showed
            # that putting TMPDIR on the shared PVC can leave NFS cleanup
            # failures even when the behavioral episode itself is valid.
            "TMPDIR": str(
                Path("/tmp")/"vla_wam_v3b002"/lane_pod_uid/
                cell.cell_id.replace(":", "__")
            ),
        },
    }


def _completed(plan: dict[str, Any]) -> dict[str, Any] | None:
    output = Path(plan["attempt_dir"])/"raw_episode.jsonl"
    manifest = output.with_name(output.name+".manifest.json")
    if not output.exists() and not manifest.exists():
        return None
    if not output.is_file() or not manifest.is_file():
        raise ContractError(f"partial retained attempt: {output.parent}")
    lines = output.read_text(encoding="utf-8").splitlines()
    row = parse_jsonl_record(lines[0]) if len(lines) == 1 else None
    if row is None or row.get("registered_cell_id") != plan["cell_id"]:
        raise ContractError("retained cell JSONL identity changed")
    value = _object(manifest)
    if value.get("row_count") != 1 or value.get("jsonl_sha256") != sha256_file(output):
        raise ContractError("retained cell post-close manifest changed")
    return {"cell_id": plan["cell_id"], "status": "already_compiled", "raw_jsonl": str(output)}


def _append_infra(attempt: Path, status: str, error: BaseException | None = None) -> None:
    record = {
        "schema_version": "vla-wam-shared-v3b-pi05-infrastructure-event-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": status, "denominator_eligible": status == "compiled_valid_behavioral_cell",
    }
    if error is not None:
        record["error"] = f"{type(error).__name__}: {error}"
    # Operational events are append-only but are not the common v3
    # infrastructure-attempt schema; keep the filename distinct so the result
    # compiler cannot accidentally treat them as denominator-adjacent rows.
    with (attempt/"attempt_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"))+"\n")


def run_cell(plan: dict[str, Any]) -> dict[str, Any]:
    attempt = Path(plan["attempt_dir"])
    if attempt.exists():
        done = _completed(plan)
        if done is not None:
            return done
        raise FileExistsError(f"partial attempt preserved outside denominator: {attempt}")
    if not Path(FROZEN_VK_ICD).is_file():
        raise ContractError(f"Vulkan ICD missing: {FROZEN_VK_ICD}")
    attempt.mkdir(parents=True, exist_ok=False)
    for key in ("XDG_CACHE_HOME", "WARP_CACHE_PATH", "MPLCONFIGDIR", "TMPDIR"):
        Path(plan["environment"][key]).mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment.pop("DISPLAY", None)
    environment.update(plan["environment"])
    _append_infra(attempt, "bridge_started")
    try:
        subprocess.run(plan["bridge_command"], check=True, env=environment)
        subprocess.run(plan["compiler_command"], check=True, env=environment)
    except BaseException as exc:
        _append_infra(attempt, "infrastructure_failed_excluded_from_denominator", exc)
        raise
    result = _completed(plan)
    if result is None:
        raise RuntimeError("compiler returned without a valid JSONL cell")
    _append_infra(attempt, "compiled_valid_behavioral_cell")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="mode", required=True)
    for mode in ("plan", "run-cell", "run-queue"):
        command = commands.add_parser(mode)
        command.add_argument("--repo-root", type=Path, required=True)
        command.add_argument("--release-manifest", type=Path, required=True)
        command.add_argument("--release-manifest-sha256", required=True)
        command.add_argument("--runtime-manifest", type=Path, required=True)
        command.add_argument("--release-gate", type=Path, required=True)
        command.add_argument("--fixture-candidate", type=Path, required=True)
        command.add_argument("--raw-root", type=Path, required=True)
        command.add_argument("--remote-host", required=True)
        command.add_argument("--remote-port", type=int, default=8001)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--gpu-index", type=int, default=0)
        command.add_argument("--lane-pod-uid", required=True)
        command.add_argument("--lane-gpu-uuid", required=True)
        command.add_argument("--lane-index", type=int, default=0)
        command.add_argument("--lane-count", type=int, default=1)
        if mode == "run-cell":
            command.add_argument("--cell-id", required=True)
        if mode == "run-queue":
            command.add_argument("--limit-seeds", type=int)
    args = parser.parse_args()
    release = load_release_bundle(args.repo_root, args.release_manifest,
                                  expected_manifest_sha256=args.release_manifest_sha256)
    runtime = validate_runtime_identity(_object(args.runtime_manifest), repo_root=args.repo_root, release=release)
    validate_release_gate(_object(args.release_gate), release=release, runtime=runtime)
    lane_matches = [lane for lane in runtime["live_topology"]["simulator_lanes"]
                    if lane["pod_uid"] == args.lane_pod_uid and lane["gpu_uuid"] == args.lane_gpu_uuid]
    if len(lane_matches) != 1:
        raise ContractError("queue lane is not the exact runtime-bound pod UID/GPU UUID")
    if not args.fixture_candidate.is_file() or sha256_file(args.fixture_candidate) != FIXTURE_SHA256:
        raise ContractError("fixture candidate is not the exact B001 candidate")
    cells = list(cells_for_lane(release.cells, lane_index=args.lane_index, lane_count=args.lane_count))
    cells.sort(key=lambda cell: (cell.seed, cell.row["execution_order_index_within_seed"]))
    if args.mode == "run-queue" and args.limit_seeds is not None:
        if args.limit_seeds < 1:
            raise ValueError("--limit-seeds must be positive")
        keep = sorted({cell.seed for cell in cells})[:args.limit_seeds]
        cells = [cell for cell in cells if cell.seed in keep]
    plans = [cell_plan(repo_root=args.repo_root, release_manifest=args.release_manifest,
                       release_manifest_sha256=args.release_manifest_sha256,
                       runtime_manifest=args.runtime_manifest, release_gate=args.release_gate,
                       fixture_candidate=args.fixture_candidate, raw_root=args.raw_root,
                       cell=cell, remote_host=args.remote_host, remote_port=args.remote_port,
                       device=args.device, gpu_index=args.gpu_index,
                       lane_pod_uid=args.lane_pod_uid, lane_gpu_uuid=args.lane_gpu_uuid) for cell in cells]
    if args.mode == "plan":
        print(json.dumps({"lane_index": args.lane_index, "lane_count": args.lane_count,
                          "seed_count": len({p['seed'] for p in plans}), "cell_count": len(plans),
                          "cells": plans}, indent=2, sort_keys=True))
        return
    if args.mode == "run-cell":
        match = [index for index, plan in enumerate(plans) if plan["cell_id"] == args.cell_id]
        if len(match) != 1:
            raise ContractError("cell is not owned by this whole-seed lane")
        index = match[0]
        for predecessor in plans[:index]:
            if predecessor["seed"] == plans[index]["seed"] and _completed(predecessor) is None:
                raise ContractError(f"within-seed order requires {predecessor['cell_id']} next")
        plans = [plans[index]]
    results = [run_cell(plan) for plan in plans]
    print(json.dumps({"completed": len(results), "results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
