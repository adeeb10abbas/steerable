#!/usr/bin/env python3
"""Plan or run the released V3-B008/B009 Nano queue without overwriting evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from experiments.v3.cosmos_nano_tier_b.runtime_contract import (
    AuthorizedCell,
    CONFIG,
    ContractError,
    ReleaseBundle,
    load_json,
    load_release,
    load_runtime,
    sha256_file,
    validate_behavioral_release_gate,
)
from tools.vla_wam_v3_episode_schema import parse_jsonl_record
from experiments.v3.cosmos_nano_tier_b.build_launch_manifest import LIVE_FILES


RuntimeContractError = ContractError


FROZEN_VK_ICD = "/etc/vulkan/icd.d/nvidia_icd.json"
FROZEN_LD_LIBRARY_PATH = (
    "/data/users/ali/vla_wam/envs/robolab-native-libs-ubuntu2204/"
    "usr/lib/x86_64-linux-gnu:"
    "/data/users/ali/glvnd/lib:"
    "/data/users/ali/vla_wam/envs/fastwam-native-libs/lib:"
    "/usr/lib/x86_64-linux-gnu"
)


def validate_launch_manifest(
    path: Path, *, study_root: Path, release: ReleaseBundle, runtime: dict[str, Any], release_gate: Path
) -> dict[str, Any]:
    launch = load_json(path, "behavior launch manifest")
    expected = {
        "schema_version": "vla-wam-shared-v3b008-v3b009-nano-behavior-launch-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": release.amendment_id,
        "model_id": "cosmos3_nano_policy_droid",
        "status": "behavior_launch_hash_bound_zero_cells_launched",
        "authorized_behavioral_cells": release.config["cells"],
        "launched_behavioral_cells": 0,
        "release_manifest_sha256": release.manifest_sha256,
        "cells_sha256": release.cells_sha256,
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "behavioral_release_gate_sha256": sha256_file(release_gate),
        "candidate_sha256": release.cells[0].row["candidate_sha256"],
    }
    for key, wanted in expected.items():
        if launch.get(key) != wanted:
            raise RuntimeContractError(f"behavior launch manifest mismatch for {key}")
    rows = launch.get("live_sources")
    if not isinstance(rows, list) or not rows:
        raise RuntimeContractError("behavior launch manifest lacks source inventory")
    root = study_root.resolve()
    for row in rows:
        relative = row.get("path")
        if not isinstance(relative, str) or relative.startswith("/") or ".." in Path(relative).parts:
            raise RuntimeContractError("behavior launch source path is invalid")
        source = root / relative
        if sha256_file(source) != row.get("sha256") or source.stat().st_size != row.get("bytes"):
            raise RuntimeContractError(f"behavior launch source changed: {relative}")
    required = set(LIVE_FILES)
    if not required.issubset({row["path"] for row in rows}):
        raise RuntimeContractError("behavior launch source inventory is incomplete")
    return launch


def ordered_cells(release: ReleaseBundle) -> list[AuthorizedCell]:
    cells = sorted(
        release.cells,
        key=lambda cell: (cell.seed, cell.row["execution_order_index_within_seed"]),
    )
    if len(cells) != release.config["cells"]:
        raise RuntimeContractError(
            f"live queue must contain exactly {release.config['cells']} cells"
        )
    return cells


def _attempt_dir(raw_root: Path, cell: AuthorizedCell) -> Path:
    return (
        Path(raw_root).resolve()
        / cell.row["amendment_id"].lower().replace("-", "")
        / cell.cell_id.replace(":", "__")
        / "attempt01"
    )


def bridge_command(
    *,
    study_root: Path,
    amendment_id: str,
    release_manifest: Path,
    release_manifest_sha256: str,
    runtime_manifest: Path,
    release_gate: Path,
    safe_fixture: Path,
    safe_fixture_sha256: str,
    raw_root: Path,
    cell: AuthorizedCell,
    remote_host: str,
    remote_port: int,
    lane_pod_uid: str,
    lane_gpu_uuid: str,
    **_: Any,
) -> list[str]:
    attempt = _attempt_dir(raw_root, cell)
    stem = cell.cell_id.replace(":", "__")
    return [
        sys.executable,
        "-m",
        "experiments.v3.cosmos_nano_tier_b.robolab_bridge",
        "--study-root", str(Path(study_root).resolve()),
        "--amendment-id", amendment_id,
        "--release-manifest", str(Path(release_manifest).resolve()),
        "--release-manifest-sha256", release_manifest_sha256,
        "--runtime-manifest", str(Path(runtime_manifest).resolve()),
        "--release-gate", str(Path(release_gate).resolve()),
        "--cell-id", cell.cell_id,
        "--safe-fixture", str(Path(safe_fixture).resolve()),
        "--safe-fixture-sha256", safe_fixture_sha256,
        "--state-capture-dir", str(attempt / "state_capture"),
        "--action-trace-dir", str(attempt / "action_traces"),
        "--future-trace-dir", str(attempt / "decoded_futures"),
        "--reset-attestation", str(attempt / "reset_attestation.json"),
        "--simulator-export", str(attempt / "simulator_export.json"),
        "--remote-host", remote_host,
        "--remote-port", str(remote_port),
        "--lane-pod-uid", lane_pod_uid,
        "--lane-gpu-uuid", lane_gpu_uuid,
        "--open-loop-horizon", "32",
        "--instruction-controller", "static",
        "--output-dir", str(attempt / "simulator"),
        # RoboLab joins this value beneath its package output directory, but
        # os.path.join preserves an absolute final component.  Supplying the
        # immutable attempt-local directory keeps videos/HDF5/configs on the
        # PVC evidence path and prevents cross-attempt output reuse.
        "--output-folder-name", str(attempt / "simulator"),
        "--num-envs", "1",
        "--num-runs", "1",
        "--headless",
        "--renderer", "realtime",
        "--rendering-type", "balanced",
        "--device", "cuda:0",
        "--kit_args=--/rtx/verifyDriverVersion/enabled=false",
        "--video-mode", "viewport",
        "--instruction-type", "default",
        "--disable-subtask",
    ]


def compiler_command(
    *,
    study_root: Path,
    amendment_id: str,
    release_manifest: Path,
    release_manifest_sha256: str,
    runtime_manifest: Path,
    release_gate: Path,
    raw_root: Path,
    cell: AuthorizedCell,
) -> list[str]:
    attempt = _attempt_dir(raw_root, cell)
    return [
        sys.executable,
        "-m",
        "experiments.v3.cosmos_nano_tier_b.compile_cell",
        "--study-root", str(Path(study_root).resolve()),
        "--amendment-id", amendment_id,
        "--release-manifest", str(Path(release_manifest).resolve()),
        "--release-manifest-sha256", release_manifest_sha256,
        "--runtime-manifest", str(Path(runtime_manifest).resolve()),
        "--release-gate", str(Path(release_gate).resolve()),
        "--reset-attestation", str(attempt / "reset_attestation.json"),
        "--cell-id", cell.cell_id,
        "--export", str(attempt / "simulator_export.json"),
        "--output-jsonl", str(attempt / "raw_episode.jsonl"),
    ]


def cell_plan(**kwargs: Any) -> dict[str, Any]:
    cell = kwargs["cell"]
    attempt = _attempt_dir(kwargs["raw_root"], cell)
    cache_root = attempt / "runtime_cache"
    environment = {
        "CUDA_VISIBLE_DEVICES": "0",
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        "VK_ICD_FILENAMES": FROZEN_VK_ICD,
        "LD_LIBRARY_PATH": FROZEN_LD_LIBRARY_PATH,
        "XDG_CACHE_HOME": str(cache_root / "xdg"),
        "WARP_CACHE_PATH": str(cache_root / "warp"),
        "MPLCONFIGDIR": str(cache_root / "matplotlib"),
        "TMPDIR": str(cache_root / "tmp"),
    }
    return {
        "cell_id": cell.cell_id,
        "seed": cell.seed,
        "arm": cell.arm,
        "target_object": cell.row.get("target_object", "rubiks_cube"),
        "reference_object": cell.row.get("reference_object", "bowl"),
        "relation": cell.relation,
        "prompt": cell.row["prompt"],
        "execution_order_index_within_seed": cell.row["execution_order_index_within_seed"],
        "matched_block_id": f"{cell.row['amendment_id'].lower().replace('-', '')}:nano:seed{cell.seed}",
        "study_root": kwargs["study_root_string"],
        "lane_pod_uid": kwargs["lane_pod_uid"],
        "lane_gpu_uuid": kwargs["lane_gpu_uuid"],
        "attempt_dir": str(attempt),
        "environment": environment,
        "thermal_guard": "native_process_group",
        "bridge_command": bridge_command(**kwargs),
        "compiler_command": compiler_command(
            study_root=kwargs["study_root"],
            amendment_id=kwargs["amendment_id"],
            release_manifest=kwargs["release_manifest"],
            release_manifest_sha256=kwargs["release_manifest_sha256"],
            runtime_manifest=kwargs["runtime_manifest"],
            release_gate=kwargs["release_gate"],
            raw_root=kwargs["raw_root"],
            cell=cell,
        ),
    }


def _completed_result(plan: dict[str, Any]) -> dict[str, Any] | None:
    attempt = Path(plan["attempt_dir"])
    output = attempt / "raw_episode.jsonl"
    manifest_path = output.with_name(output.name + ".manifest.json")
    if not output.exists() and not manifest_path.exists():
        return None
    if not output.is_file() or not manifest_path.is_file():
        raise RuntimeContractError(
            f"retained attempt is partial and remains outside the denominator: {attempt}"
        )
    lines = output.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise RuntimeContractError(f"compiled cell must contain exactly one JSONL row: {output}")
    row = parse_jsonl_record(lines[0])
    if (
        row.get("record_type") != "behavioral_episode"
        or row.get("behavioral_result_valid") is not True
        or row.get("registered_cell_id") != plan["cell_id"]
    ):
        raise RuntimeContractError("retained JSONL is not the exact valid behavioral cell")
    manifest = load_json(manifest_path, "compiled cell batch manifest")
    expected_manifest = {
        "schema_version": "vla-wam-shared-v3-jsonl-batch-manifest-v1",
        "jsonl_sha256": sha256_file(output),
        "jsonl_bytes": output.stat().st_size,
        "row_count": 1,
    }
    for key, value in expected_manifest.items():
        if manifest.get(key) != value:
            raise RuntimeContractError(f"compiled cell manifest mismatch for {key}")
    return {
        "cell_id": plan["cell_id"],
        "status": "already_compiled_valid_behavioral_cell",
        "raw_jsonl": str(output),
        "raw_jsonl_sha256": sha256_file(output),
        "batch_manifest": str(manifest_path),
        "batch_manifest_sha256": sha256_file(manifest_path),
    }


def _append_attempt_event(attempt: Path, status: str, **details: Any) -> None:
    record = {
        "schema_version": "vla-wam-shared-v3b-nano-attempt-event-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": status,
        **details,
    }
    with (attempt / "attempt_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n")


def run_cell(plan: dict[str, Any]) -> dict[str, Any]:
    attempt = Path(plan["attempt_dir"])
    if attempt.exists():
        completed = _completed_result(plan)
        if completed is not None:
            return completed
        raise FileExistsError(f"retained partial attempt is preserved outside the denominator: {attempt}")
    if not Path(FROZEN_VK_ICD).is_file():
        raise RuntimeContractError(f"known-good NVIDIA Vulkan ICD is missing: {FROZEN_VK_ICD}")
    library_directories = [Path(value) for value in FROZEN_LD_LIBRARY_PATH.split(":")]
    missing_libraries = [str(path) for path in library_directories if not path.is_dir()]
    if missing_libraries:
        raise RuntimeContractError(f"known-good native library paths are missing: {missing_libraries}")
    attempt.mkdir(parents=True, exist_ok=False)
    for key in ("XDG_CACHE_HOME", "WARP_CACHE_PATH", "MPLCONFIGDIR", "TMPDIR"):
        Path(plan["environment"][key]).mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment.pop("DISPLAY", None)
    # The user explicitly accepted the Omniverse EULA for this authorized run.
    environment.update(plan["environment"])
    _append_attempt_event(
        attempt,
        "bridge_started",
        cell_id=plan["cell_id"],
        denominator_eligible=False,
    )
    try:
        guarded = [
            sys.executable,
            str(Path(plan["study_root"]) / "tools/native_process_group_thermal_guard.py"),
            "--launch",
            "--gpu-index", "0",
            "--output", str(attempt / "thermal_events.jsonl"),
            "--ledger-output", str(attempt / "runtime_interventions_cosmos3_nano_policy_droid.json"),
            "--invalid-attempts-output", str(attempt / "invalid_attempts_cosmos3_nano_policy_droid.json"),
            "--model-id", "cosmos3_nano_policy_droid",
            "--pair-id", plan["matched_block_id"],
            "--environment-seed", str(plan["seed"]),
            "--sampling-seed", str(plan["seed"]),
            "--requested-relation", plan["relation"],
            "--",
            *plan["bridge_command"],
        ]
        subprocess.run(guarded, check=True, env=environment)
    except BaseException as exc:
        _append_attempt_event(
            attempt,
            "infrastructure_failed_before_compilation",
            cell_id=plan["cell_id"],
            denominator_eligible=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    bridge_failure = attempt / "bridge_failure.json"
    simulator_export = attempt / "simulator_export.json"
    if bridge_failure.is_file() or not simulator_export.is_file():
        reason = (
            f"bridge recorded {bridge_failure}"
            if bridge_failure.is_file()
            else f"bridge did not write {simulator_export}"
        )
        _append_attempt_event(
            attempt,
            "infrastructure_failed_before_compilation",
            cell_id=plan["cell_id"],
            denominator_eligible=False,
            error=reason,
        )
        raise RuntimeContractError(reason)
    try:
        subprocess.run(plan["compiler_command"], check=True, env=environment)
    except BaseException as exc:
        _append_attempt_event(
            attempt,
            "infrastructure_failed_during_compilation",
            cell_id=plan["cell_id"],
            denominator_eligible=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    output = attempt / "raw_episode.jsonl"
    manifest = output.with_name(output.name + ".manifest.json")
    if not output.is_file() or not manifest.is_file():
        raise RuntimeError("cell completed without compiler JSONL integrity evidence")
    result = {
        "cell_id": plan["cell_id"],
        "status": "compiled_valid_behavioral_cell",
        "raw_jsonl": str(output),
        "raw_jsonl_sha256": sha256_file(output),
        "batch_manifest": str(manifest),
        "batch_manifest_sha256": sha256_file(manifest),
    }
    _append_attempt_event(
        attempt,
        "compiled_valid_behavioral_cell",
        cell_id=plan["cell_id"],
        denominator_eligible=True,
        raw_jsonl_sha256=result["raw_jsonl_sha256"],
    )
    return result


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--amendment-id", choices=tuple(CONFIG), required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--release-manifest-sha256", required=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="mode", required=True)
    for mode in ("plan", "run-cell", "run-queue"):
        command = commands.add_parser(mode)
        _common(command)
        command.add_argument("--runtime-manifest", type=Path, required=True)
        command.add_argument("--release-gate", type=Path, required=True)
        command.add_argument("--launch-manifest", type=Path, required=True)
        command.add_argument("--safe-fixture", type=Path, required=True)
        command.add_argument("--raw-root", type=Path, required=True)
        command.add_argument("--remote-host", required=True)
        command.add_argument("--remote-port", type=int)
        command.add_argument("--lane-index", type=int, required=True)
        command.add_argument("--lane-count", type=int, required=True)
        command.add_argument("--lane-pod-uid", required=True)
        command.add_argument("--lane-gpu-uuid", required=True)
        if mode == "run-cell":
            command.add_argument("--cell-id", required=True)
        if mode == "run-queue":
            command.add_argument("--limit-seeds", type=int)
    args = parser.parse_args()
    release = load_release(args.study_root, args.amendment_id, args.release_manifest)
    if release.manifest_sha256 != args.release_manifest_sha256:
        raise RuntimeContractError("release manifest CLI hash mismatch")
    runtime = load_runtime(args.runtime_manifest, study_root=args.study_root, release=release)
    validate_behavioral_release_gate(args.release_gate, release=release, runtime=runtime)
    validate_launch_manifest(
        args.launch_manifest,
        study_root=args.study_root,
        release=release,
        runtime=runtime,
        release_gate=args.release_gate,
    )
    candidate_sha256 = release.cells[0].row["candidate_sha256"]
    if not args.safe_fixture.is_file() or sha256_file(args.safe_fixture) != candidate_sha256:
        raise RuntimeContractError("model-blind candidate path does not match released SHA-256")
    remote_port = CONFIG[args.amendment_id]["port"] if args.remote_port is None else args.remote_port
    if remote_port != CONFIG[args.amendment_id]["port"]:
        raise RuntimeContractError("remote port differs from isolated amendment contract")
    if args.lane_count < 1 or not 0 <= args.lane_index < args.lane_count:
        raise ValueError("lane index must be within 0..lane-count-1")
    all_cells = ordered_cells(release)
    lane_seeds = [
        seed for index, seed in enumerate(sorted({cell.seed for cell in all_cells}))
        if index % args.lane_count == args.lane_index
    ]
    if args.mode == "run-queue" and args.limit_seeds is not None:
        if args.limit_seeds < 1:
            raise ValueError("--limit-seeds must be positive")
        lane_seeds = lane_seeds[: args.limit_seeds]
    cells = [cell for cell in all_cells if cell.seed in lane_seeds]
    plans = [
        cell_plan(
            study_root=args.study_root,
            amendment_id=args.amendment_id,
            release_manifest=args.release_manifest,
            release_manifest_sha256=args.release_manifest_sha256,
            runtime_manifest=args.runtime_manifest,
            release_gate=args.release_gate,
            safe_fixture=args.safe_fixture,
            safe_fixture_sha256=candidate_sha256,
            raw_root=args.raw_root,
            cell=cell,
            remote_host=args.remote_host,
            remote_port=remote_port,
            study_root_string=str(args.study_root.resolve()),
            lane_pod_uid=args.lane_pod_uid,
            lane_gpu_uuid=args.lane_gpu_uuid,
        )
        for cell in cells
    ]
    if args.mode == "plan":
        print(json.dumps({"lane_index": args.lane_index, "lane_count": args.lane_count, "seeds": lane_seeds, "cell_count": len(plans), "cells": plans}, indent=2, sort_keys=True))
        return
    if args.mode == "run-cell":
        target_indices = [index for index, plan in enumerate(plans) if plan["cell_id"] == args.cell_id]
        if len(target_indices) != 1:
            raise RuntimeContractError("smoke cell is not in the exact released queue")
        target_index = target_indices[0]
        target_seed = plans[target_index]["seed"]
        for predecessor in [item for item in plans[:target_index] if item["seed"] == target_seed]:
            if _completed_result(predecessor) is None:
                raise RuntimeContractError(
                    "run-cell preserves the frozen within-seed order; the exact next cell is "
                    f"{predecessor['cell_id']}"
                )
        plans = [plans[target_index]]
    results = [run_cell(plan) for plan in plans]
    print(json.dumps({"completed": len(results), "results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
