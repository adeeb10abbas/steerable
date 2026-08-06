#!/usr/bin/env python3
"""Plan or run the released V3-B001 Nano queue without overwriting evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from experiments.v3.cosmos_nano_phase_b.live_support import (
    bind_live_stack_runtime,
    verify_live_runtime_identity,
)
from experiments.v3.cosmos_nano_phase_b.runtime_adapter import (
    AuthorizedCell,
    ReleaseBundle,
    RuntimeContractError,
    load_json,
    load_release_bundle,
    sha256_file,
)
from tools.vla_wam_v3_episode_schema import parse_jsonl_record


FROZEN_VK_ICD = "/etc/vulkan/icd.d/nvidia_icd.json"
FROZEN_LD_LIBRARY_PATH = (
    "/data/users/ali/vla_wam/envs/robolab-native-libs-ubuntu2204/"
    "usr/lib/x86_64-linux-gnu:"
    "/data/users/ali/glvnd/lib:"
    "/data/users/ali/vla_wam/envs/fastwam-native-libs/lib:"
    "/usr/lib/x86_64-linux-gnu"
)


def _candidate_hash(release: ReleaseBundle) -> str:
    value = release.amendment.get("calibration_evidence", {}).get("candidate", {}).get("sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeContractError("released calibration candidate hash is missing")
    return value


def ordered_cells(release: ReleaseBundle) -> list[AuthorizedCell]:
    cells = sorted(
        release.cells,
        key=lambda cell: (cell.seed, cell.row["execution_order_index_within_seed"]),
    )
    if len(cells) != 108:
        raise RuntimeContractError("live queue must contain exactly 108 cells")
    return cells


def _attempt_dir(raw_root: Path, cell: AuthorizedCell) -> Path:
    return (
        Path(raw_root).resolve()
        / "V3-B001_cosmos3_nano_position_mirror"
        / cell.cell_id.replace(":", "__")
        / "attempt01"
    )


def bridge_command(
    *,
    study_root: Path,
    release_manifest: Path,
    release_manifest_sha256: str,
    runtime_manifest: Path,
    fixture_candidate: Path,
    fixture_candidate_sha256: str,
    raw_root: Path,
    cell: AuthorizedCell,
    remote_host: str,
    remote_port: int,
) -> list[str]:
    attempt = _attempt_dir(raw_root, cell)
    stem = cell.cell_id.replace(":", "__")
    return [
        sys.executable,
        str(Path(study_root).resolve() / "experiments/v3/cosmos_nano_phase_b/robolab_bridge.py"),
        "--study-root", str(Path(study_root).resolve()),
        "--release-manifest", str(Path(release_manifest).resolve()),
        "--release-manifest-sha256", release_manifest_sha256,
        "--runtime-manifest", str(Path(runtime_manifest).resolve()),
        "--cell-id", cell.cell_id,
        "--fixture-candidate", str(Path(fixture_candidate).resolve()),
        "--fixture-candidate-sha256", fixture_candidate_sha256,
        "--state-capture-dir", str(attempt / "state_capture"),
        "--action-trace-dir", str(attempt / "action_traces"),
        "--future-trace-dir", str(attempt / "decoded_futures"),
        "--reset-attestation", str(attempt / "reset_attestation.json"),
        "--simulator-export", str(attempt / "simulator_export.json"),
        "--remote-host", remote_host,
        "--remote-port", str(remote_port),
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
    release_manifest: Path,
    release_manifest_sha256: str,
    runtime_manifest: Path,
    raw_root: Path,
    cell: AuthorizedCell,
) -> list[str]:
    attempt = _attempt_dir(raw_root, cell)
    return [
        sys.executable,
        str(Path(study_root).resolve() / "experiments/v3/cosmos_nano_phase_b/compile_cell.py"),
        "--study-root", str(Path(study_root).resolve()),
        "--release-manifest", str(Path(release_manifest).resolve()),
        "--release-manifest-sha256", release_manifest_sha256,
        "--runtime-manifest", str(Path(runtime_manifest).resolve()),
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
        "relation": cell.relation,
        "prompt": cell.row["prompt"],
        "execution_order_index_within_seed": cell.row["execution_order_index_within_seed"],
        "attempt_dir": str(attempt),
        "environment": environment,
        "thermal_guard": "not_used",
        "bridge_command": bridge_command(**kwargs),
        "compiler_command": compiler_command(
            study_root=kwargs["study_root"],
            release_manifest=kwargs["release_manifest"],
            release_manifest_sha256=kwargs["release_manifest_sha256"],
            runtime_manifest=kwargs["runtime_manifest"],
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
        subprocess.run(plan["bridge_command"], check=True, env=environment)
    except BaseException as exc:
        _append_attempt_event(
            attempt,
            "infrastructure_failed_before_compilation",
            cell_id=plan["cell_id"],
            denominator_eligible=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
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
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--release-manifest-sha256", required=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="mode", required=True)
    bind = commands.add_parser("bind-runtime")
    _common(bind)
    bind.add_argument("--base-runtime-manifest", type=Path, required=True)
    bind.add_argument("--output", type=Path, required=True)
    for mode in ("plan", "run-cell", "run-queue"):
        command = commands.add_parser(mode)
        _common(command)
        command.add_argument("--runtime-manifest", type=Path, required=True)
        command.add_argument("--fixture-candidate", type=Path, required=True)
        command.add_argument("--raw-root", type=Path, required=True)
        command.add_argument("--remote-host", required=True)
        command.add_argument("--remote-port", type=int, default=18011)
        if mode == "run-cell":
            command.add_argument("--cell-id", required=True)
        if mode == "run-queue":
            command.add_argument("--limit", type=int)
    args = parser.parse_args()
    release = load_release_bundle(
        args.release_manifest,
        expected_manifest_sha256=args.release_manifest_sha256,
    )
    if args.mode == "bind-runtime":
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite runtime identity: {args.output}")
        payload = bind_live_stack_runtime(
            study_root=args.study_root,
            release=release,
            base_runtime_manifest=args.base_runtime_manifest,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        verify_live_runtime_identity(
            args.output,
            study_root=args.study_root,
            release=release,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    verify_live_runtime_identity(
        args.runtime_manifest,
        study_root=args.study_root,
        release=release,
    )
    candidate_hash = _candidate_hash(release)
    if not args.fixture_candidate.is_file() or sha256_file(args.fixture_candidate) != candidate_hash:
        raise RuntimeContractError("fixture candidate path does not match released SHA-256")
    cells = ordered_cells(release)
    if args.mode == "run-queue" and args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        cells = cells[: args.limit]
    plans = [
        cell_plan(
            study_root=args.study_root,
            release_manifest=args.release_manifest,
            release_manifest_sha256=args.release_manifest_sha256,
            runtime_manifest=args.runtime_manifest,
            fixture_candidate=args.fixture_candidate,
            fixture_candidate_sha256=candidate_hash,
            raw_root=args.raw_root,
            cell=cell,
            remote_host=args.remote_host,
            remote_port=args.remote_port,
        )
        for cell in cells
    ]
    if args.mode == "plan":
        print(json.dumps({"cell_count": len(plans), "cells": plans}, indent=2, sort_keys=True))
        return
    if args.mode == "run-cell":
        target_indices = [index for index, plan in enumerate(plans) if plan["cell_id"] == args.cell_id]
        if len(target_indices) != 1:
            raise RuntimeContractError("smoke cell is not in the exact released queue")
        target_index = target_indices[0]
        for predecessor in plans[:target_index]:
            if _completed_result(predecessor) is None:
                raise RuntimeContractError(
                    "run-cell preserves released global order; the exact next cell is "
                    f"{predecessor['cell_id']}"
                )
        plans = [plans[target_index]]
    results = [run_cell(plan) for plan in plans]
    print(json.dumps({"completed": len(results), "results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
