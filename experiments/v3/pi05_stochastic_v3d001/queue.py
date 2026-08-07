#!/usr/bin/env python3
"""Plan and execute the exact V3-D001 π0.5 queue in whole-seed lanes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from experiments.v3.pi05_stochastic_v3d001.contract import (
    AuthorizedCell, ContractError, RELEASE_MANIFEST_SHA256, cells_for_lane,
    load_release, sha256_file, validate_runtime,
)
from tools.vla_wam_v3_episode_schema import parse_jsonl_record


FROZEN_VK_ICD = "/etc/vulkan/icd.d/nvidia_icd.json"


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


def _attempt(raw_root: Path, cell: AuthorizedCell) -> Path:
    return Path(raw_root).resolve()/"V3-D001_pi05_nested_stochastic"/cell.cell_id.replace(":", "__")/"attempt01"


def cell_plan(*, repo_root: Path, release_manifest: Path, runtime_identity: Path,
              phase_a_release_gate: Path, raw_root: Path, cell: AuthorizedCell,
              remote_host: str, remote_port: int, device: str, gpu_index: int,
              lane_pod_uid: str, lane_gpu_uuid: str) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    attempt = _attempt(raw_root, cell)
    common = [
        "--study-root", str(root), "--release-manifest", str(release_manifest.resolve()),
        "--runtime-identity", str(runtime_identity.resolve()),
        "--phase-a-release-gate", str(phase_a_release_gate.resolve()),
        "--cell-id", cell.cell_id, "--lane-pod-uid", lane_pod_uid,
        "--lane-gpu-uuid", lane_gpu_uuid,
    ]
    worker = [
        sys.executable, "-m", "experiments.v3.pi05_stochastic_v3d001.robolab_bridge", *common,
        "--state-capture-dir", str(attempt/"state_capture"),
        "--action-trace-dir", str(attempt/"action_traces"),
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
        "--model-id", cell.row["model_id"], "--pair-id", cell.block_id,
        "--environment-seed", str(cell.environment_seed),
        "--sampling-seed", str(cell.sampling_seed_base),
        "--requested-relation", cell.relation, "--", *worker,
    ]
    compiler = [
        sys.executable, "-m", "experiments.v3.pi05_stochastic_v3d001.compile_cell",
        "--repo-root", str(root), "--release-manifest", str(release_manifest.resolve()),
        "--runtime-identity", str(runtime_identity.resolve()),
        "--phase-a-release-gate", str(phase_a_release_gate.resolve()),
        "--cell-id", cell.cell_id, "--export", str(attempt/"simulator_export.json"),
        "--output-jsonl", str(attempt/"raw_episode.jsonl"),
    ]
    return {
        "cell_id": cell.cell_id, "environment_seed": cell.environment_seed,
        "sampling_index": cell.sampling_index, "relation": cell.relation,
        "matched_stochastic_block_id": cell.block_id,
        "execution_order_index_within_matched_stochastic_block": cell.row["execution_order_index_within_matched_stochastic_block"],
        "attempt_dir": str(attempt), "bridge_command": bridge,
        "guarded_worker_command": worker, "compiler_command": compiler,
        "environment": {
            "OMNI_KIT_ACCEPT_EULA": "YES", "NVIDIA_DRIVER_CAPABILITIES": "all",
            "VK_ICD_FILENAMES": FROZEN_VK_ICD,
            "XDG_CACHE_HOME": str(attempt/"cache/xdg"),
            "WARP_CACHE_PATH": str(attempt/"cache/warp"),
            "MPLCONFIGDIR": str(attempt/"cache/matplotlib"),
            "TMPDIR": str(Path("/tmp")/"vla_wam_v3d001"/lane_pod_uid/cell.cell_id.replace(":", "__")),
        },
    }


def _completed(plan: dict[str, Any]) -> dict[str, Any] | None:
    output = Path(plan["attempt_dir"])/"raw_episode.jsonl"
    manifest = output.with_name(output.name+".manifest.json")
    if not output.exists() and not manifest.exists():
        return None
    if not output.is_file() or not manifest.is_file():
        raise ContractError(f"partial retained V3-D001 attempt: {output.parent}")
    lines = output.read_text(encoding="utf-8").splitlines()
    row = parse_jsonl_record(lines[0]) if len(lines) == 1 else None
    if row is None or row.get("registered_cell_id") != plan["cell_id"]:
        raise ContractError("retained V3-D001 JSONL identity changed")
    value = _object(manifest)
    if value.get("row_count") != 1 or value.get("jsonl_sha256") != sha256_file(output):
        raise ContractError("retained V3-D001 post-close manifest changed")
    return {"cell_id": plan["cell_id"], "status": "already_compiled", "raw_jsonl": str(output)}


def _append_event(attempt: Path, status: str, error: BaseException | None = None) -> None:
    row: dict[str, Any] = {
        "schema_version": "vla-wam-shared-v3d001-infrastructure-event-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": status, "denominator_eligible": status == "compiled_valid_behavioral_cell",
    }
    if error is not None:
        row["error"] = f"{type(error).__name__}: {error}"
    with (attempt/"attempt_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"))+"\n")


def run_cell(plan: dict[str, Any]) -> dict[str, Any]:
    attempt = Path(plan["attempt_dir"])
    if attempt.exists():
        done = _completed(plan)
        if done is not None:
            return done
        raise FileExistsError(f"partial V3-D001 attempt preserved outside denominator: {attempt}")
    if not Path(FROZEN_VK_ICD).is_file():
        raise ContractError(f"Vulkan ICD missing: {FROZEN_VK_ICD}")
    attempt.mkdir(parents=True, exist_ok=False)
    for key in ("XDG_CACHE_HOME", "WARP_CACHE_PATH", "MPLCONFIGDIR", "TMPDIR"):
        Path(plan["environment"][key]).mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment.pop("DISPLAY", None)
    environment.update(plan["environment"])
    _append_event(attempt, "bridge_started")
    try:
        subprocess.run(plan["bridge_command"], check=True, env=environment)
        if (attempt/"bridge_failure.json").exists() or not (attempt/"simulator_export.json").is_file():
            raise ContractError("bridge produced no valid export; retained failure is outside denominator")
        subprocess.run(plan["compiler_command"], check=True, env=environment)
    except BaseException as exc:
        _append_event(attempt, "infrastructure_failed_excluded_from_denominator", exc)
        raise
    result = _completed(plan)
    if result is None:
        raise RuntimeError("V3-D001 compiler returned without valid JSONL")
    _append_event(attempt, "compiled_valid_behavioral_cell")
    return result


def _pair_paths(release: Any, raw_root: Path, block_id: str) -> tuple[Path, Path, Path]:
    cells = [cell for cell in release.cells if cell.block_id == block_id]
    if len(cells) != 2:
        raise ContractError("released matched block must contain exactly two cells")
    by_relation = {cell.relation: cell for cell in cells}
    left = _attempt(raw_root, by_relation["left"])/"raw_episode.jsonl"
    right = _attempt(raw_root, by_relation["right"])/"raw_episode.jsonl"
    output = Path(raw_root).resolve()/"V3-D001_pi05_nested_stochastic"/"matched_pairs"/(block_id.replace(":", "__")+".json")
    return left, right, output


def compile_completed_pair(*, repo_root: Path, release_manifest: Path, release: Any,
                           raw_root: Path, block_id: str) -> dict[str, Any] | None:
    left, right, output = _pair_paths(release, raw_root, block_id)
    manifest = output.with_name(output.name+".manifest.json")
    if output.is_file() and manifest.is_file():
        value = _object(manifest)
        if value.get("matched_stochastic_block_id") != block_id or value.get("json_sha256") != sha256_file(output):
            raise ContractError("retained matched-pair evidence changed")
        return {"pair": str(output), "status": "already_compiled"}
    if output.exists() or manifest.exists():
        raise ContractError("partial matched-pair output is preserved outside analysis")
    if not left.is_file() or not right.is_file():
        return None
    command = [
        sys.executable, "-m", "experiments.v3.pi05_stochastic_v3d001.compile_pair",
        "--repo-root", str(Path(repo_root).resolve()),
        "--release-manifest", str(Path(release_manifest).resolve()),
        "--block-id", block_id, "--left-jsonl", str(left),
        "--right-jsonl", str(right), "--output", str(output),
    ]
    subprocess.run(command, check=True)
    return {"pair": str(output), "status": "compiled"}


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="mode", required=True)
    for mode in ("plan", "run-cell", "run-queue"):
        command = commands.add_parser(mode)
        command.add_argument("--repo-root", type=Path, required=True)
        command.add_argument("--release-manifest", type=Path, required=True)
        command.add_argument("--runtime-identity", type=Path, required=True)
        command.add_argument("--phase-a-release-gate", type=Path, required=True)
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
            command.add_argument("--limit-blocks", type=int)
    args = parser.parse_args()
    release = load_release(args.repo_root, args.release_manifest)
    if sha256_file(args.release_manifest.resolve()) != RELEASE_MANIFEST_SHA256:
        raise ContractError("V3-D001 release manifest digest changed")
    validate_runtime(args.repo_root, args.runtime_identity, args.phase_a_release_gate)
    cells = list(cells_for_lane(release.cells, args.lane_index, args.lane_count))
    if args.mode == "run-queue" and args.limit_blocks is not None:
        if args.limit_blocks < 1:
            raise ValueError("--limit-blocks must be positive")
        keep: list[str] = []
        for cell in cells:
            if cell.block_id not in keep:
                keep.append(cell.block_id)
        keep_set = set(keep[:args.limit_blocks])
        cells = [cell for cell in cells if cell.block_id in keep_set]
    plans = [cell_plan(repo_root=args.repo_root, release_manifest=args.release_manifest,
                       runtime_identity=args.runtime_identity,
                       phase_a_release_gate=args.phase_a_release_gate,
                       raw_root=args.raw_root, cell=cell, remote_host=args.remote_host,
                       remote_port=args.remote_port, device=args.device,
                       gpu_index=args.gpu_index, lane_pod_uid=args.lane_pod_uid,
                       lane_gpu_uuid=args.lane_gpu_uuid) for cell in cells]
    if args.mode == "plan":
        print(json.dumps({"release_manifest_sha256": RELEASE_MANIFEST_SHA256,
                          "lane_index": args.lane_index, "lane_count": args.lane_count,
                          "environment_seed_count": len({p['environment_seed'] for p in plans}),
                          "matched_block_count": len({p['matched_stochastic_block_id'] for p in plans}),
                          "cell_count": len(plans), "cells": plans}, indent=2, sort_keys=True))
        return
    if args.mode == "run-cell":
        matches = [i for i, plan in enumerate(plans) if plan["cell_id"] == args.cell_id]
        if len(matches) != 1:
            raise ContractError("cell is not owned by this whole-seed lane")
        index = matches[0]
        for predecessor in plans[:index]:
            if predecessor["matched_stochastic_block_id"] == plans[index]["matched_stochastic_block_id"] and _completed(predecessor) is None:
                raise ContractError(f"matched-block order requires {predecessor['cell_id']} next")
        plans = [plans[index]]
    results = []
    pairs = []
    for plan in plans:
        results.append(run_cell(plan))
        pair = compile_completed_pair(repo_root=args.repo_root,
                                      release_manifest=args.release_manifest,
                                      release=release, raw_root=args.raw_root,
                                      block_id=plan["matched_stochastic_block_id"])
        if pair is not None:
            pairs.append(pair)
    print(json.dumps({"completed": len(results), "pair_outputs": pairs,
                      "results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
