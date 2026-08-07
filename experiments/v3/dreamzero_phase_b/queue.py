#!/usr/bin/env python3
"""Run V3-B003 in whole-seed RTX lanes without overwriting evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.v3.dreamzero_phase_b.contract import (
    EXPECTED_SHA256,
    FIXTURE_CANDIDATE_SHA256,
    ContractError,
    load_cells,
    sha256_file,
)


FROZEN_VK_ICD = "/etc/vulkan/icd.d/nvidia_icd.json"
FROZEN_LD_LIBRARY_PATH = (
    "/data/users/ali/vla_wam/envs/robolab-native-libs-ubuntu2204/"
    "usr/lib/x86_64-linux-gnu:"
    "/data/users/ali/glvnd/lib:"
    "/data/users/ali/vla_wam/envs/fastwam-native-libs/lib:"
    "/usr/lib/x86_64-linux-gnu"
)


def _attempt(raw_root: Path, cell_id: str) -> Path:
    return Path(raw_root).resolve() / cell_id.replace(":", "__") / "attempt01"


def _completed(attempt: Path, cell_id: str) -> bool:
    output = attempt / "raw_episode.jsonl"
    manifest = output.with_name(output.name + ".manifest.json")
    if not output.exists() and not manifest.exists():
        return False
    if not output.is_file() or not manifest.is_file():
        raise ContractError(f"partial retained V3-B003 output: {attempt}")
    value = json.loads(manifest.read_text())
    return (
        value.get("registered_cell_id") == cell_id
        and value.get("row_count") == 1
        and value.get("jsonl_sha256") == sha256_file(output)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("plan", "run-queue"))
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--release-manifest-sha256", required=True)
    parser.add_argument("--runtime-identity", type=Path, required=True)
    parser.add_argument("--release-gate", type=Path, required=True)
    parser.add_argument("--fixture-candidate", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-port", type=int, required=True)
    parser.add_argument("--lane-index", type=int, required=True)
    parser.add_argument("--lane-count", type=int, required=True)
    parser.add_argument("--lane-pod-uid", required=True)
    parser.add_argument("--lane-gpu-uuid", required=True)
    parser.add_argument("--limit-seeds", type=int)
    parser.add_argument("--max-new-cells", type=int)
    args = parser.parse_args()
    if args.max_new_cells is not None and args.max_new_cells <= 0:
        parser.error("--max-new-cells must be positive")
    if (
        args.release_manifest_sha256 != EXPECTED_SHA256["manifest"]
        or sha256_file(args.release_manifest) != EXPECTED_SHA256["manifest"]
        or sha256_file(args.fixture_candidate) != FIXTURE_CANDIDATE_SHA256
    ):
        raise ContractError("V3-B003 queue inputs changed")
    all_cells = load_cells(args.repo_root)
    lane_seeds = [
        seed for index, seed in enumerate(range(9400, 9427))
        if index % args.lane_count == args.lane_index
    ]
    if args.limit_seeds is not None:
        lane_seeds = lane_seeds[:args.limit_seeds]
    cells = [cell for cell in all_cells if cell.seed in lane_seeds]
    cells.sort(key=lambda item: (item.seed, item.row["execution_order_index_within_seed"]))
    plan = [{
        "cell_id": cell.cell_id,
        "seed": cell.seed,
        "arm": cell.arm,
        "relation": cell.relation,
        "order": cell.row["execution_order_index_within_seed"],
    } for cell in cells]
    if args.mode == "plan":
        print(json.dumps({"lane_index": args.lane_index, "seeds": lane_seeds, "cells": plan}, indent=2))
        return
    new_cells = 0
    for cell in cells:
        attempt = _attempt(args.raw_root, cell.cell_id)
        if _completed(attempt, cell.cell_id):
            continue
        if args.max_new_cells is not None and new_cells >= args.max_new_cells:
            break
        if attempt.exists():
            raise FileExistsError(f"partial V3-B003 attempt preserved: {attempt}")
        attempt.mkdir(parents=True, exist_ok=False)
        caches = {
            "XDG_CACHE_HOME": attempt / "cache/xdg",
            "WARP_CACHE_PATH": attempt / "cache/warp",
            "MPLCONFIGDIR": attempt / "cache/matplotlib",
            "TMPDIR": (
                Path("/tmp")
                / "v3b003"
                / args.lane_pod_uid
                / hashlib.sha256(str(attempt).encode("utf-8")).hexdigest()[:16]
            ),
        }
        for path in caches.values():
            path.mkdir(parents=True, exist_ok=False)
        common = [
            "--study-root", str(args.repo_root.resolve()),
            "--release-manifest", str(args.release_manifest.resolve()),
            "--release-manifest-sha256", args.release_manifest_sha256,
            "--cell-id", cell.cell_id,
            "--runtime-identity", str(args.runtime_identity.resolve()),
            "--release-gate", str(args.release_gate.resolve()),
            "--fixture-candidate", str(args.fixture_candidate.resolve()),
            "--fixture-candidate-sha256", FIXTURE_CANDIDATE_SHA256,
        ]
        worker = [
            sys.executable, "-m", "experiments.v3.dreamzero_phase_b.robolab_bridge",
            *common,
            "--state-capture-dir", str(attempt / "state_capture"),
            "--action-trace-dir", str(attempt / "action_traces"),
            "--reset-attestation", str(attempt / "reset_attestation.json"),
            "--simulator-export", str(attempt / "simulator_export.json"),
            "--remote-host", args.remote_host, "--remote-port", str(args.remote_port),
            "--lane-pod-uid", args.lane_pod_uid, "--lane-gpu-uuid", args.lane_gpu_uuid,
            "--open-loop-horizon", "8", "--instruction-controller", "static",
            "--output-dir", str(attempt / "simulator"),
            "--output-folder-name", str(attempt / "simulator"),
            "--num-envs", "1", "--num-runs", "1", "--headless",
            "--renderer", "realtime", "--rendering-type", "balanced",
            "--device", "cuda:0", "--video-mode", "viewport",
            "--instruction-type", "default", "--disable-subtask",
            "--kit_args=--/rtx/verifyDriverVersion/enabled=false",
        ]
        guarded = [
            sys.executable, str(args.repo_root / "tools/native_process_group_thermal_guard.py"),
            "--launch", "--gpu-index", "0",
            "--output", str(attempt / "thermal_events.jsonl"),
            "--ledger-output", str(
                attempt / "runtime_interventions_dreamzero_droid_action_cfg.json"
            ),
            "--invalid-attempts-output", str(
                attempt / "invalid_attempts_dreamzero_droid_action_cfg.json"
            ),
            "--model-id", "dreamzero_droid_action_cfg",
            "--pair-id", f"{cell.row['matched_block_id']}:{cell.arm}",
            "--environment-seed", str(cell.seed), "--sampling-seed", str(cell.seed),
            "--requested-relation", cell.relation, "--", *worker,
        ]
        environment = dict(os.environ)
        environment.pop("DISPLAY", None)
        environment.update({key: str(value) for key, value in caches.items()})
        environment.update({
            "LD_LIBRARY_PATH": FROZEN_LD_LIBRARY_PATH,
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "NVIDIA_DRIVER_CAPABILITIES": "all",
            "VK_ICD_FILENAMES": FROZEN_VK_ICD,
        })
        try:
            subprocess.run(guarded, check=True, env=environment)
            if (attempt / "bridge_failure.json").exists() or not (attempt / "simulator_export.json").is_file():
                raise RuntimeError("V3-B003 bridge ended without a valid export")
            subprocess.run([
                sys.executable, "-m", "experiments.v3.dreamzero_phase_b.compile_cell",
                "--repo-root", str(args.repo_root.resolve()),
                "--release-manifest", str(args.release_manifest.resolve()),
                "--release-manifest-sha256", args.release_manifest_sha256,
                "--cell-id", cell.cell_id,
                "--runtime-identity", str(args.runtime_identity.resolve()),
                "--export", str(attempt / "simulator_export.json"),
                "--output-jsonl", str(attempt / "raw_episode.jsonl"),
            ], check=True, env=environment)
            new_cells += 1
        except BaseException as error:
            (attempt / "infrastructure_failure.json").write_text(json.dumps({
                "schema_version": "vla-wam-shared-v3b-dreamzero-infrastructure-failure-v1",
                "registered_cell_id": cell.cell_id,
                "behavioral_result_valid": False,
                "denominator_policy": "excluded_from_behavioral_denominator",
                "error": f"{type(error).__name__}: {error}",
            }, indent=2, sort_keys=True) + "\n")
            raise
    print(json.dumps({
        "planned_cells": len(cells),
        "new_cells_completed": new_cells,
        "lane_index": args.lane_index,
    }, indent=2))


if __name__ == "__main__":
    main()
