#!/usr/bin/env python3
"""Compile one complete DreamZero V3-B003 raw cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from experiments.v3.dreamzero_droid.adapter import build_behavioral_record
from experiments.v3.dreamzero_phase_b.contract import (
    EXPECTED_SHA256,
    load_cell,
    sha256_file,
)


LEGACY_FAILURE_STAGES = {
    "success",
    "no_object_interaction",
    "object_moved_no_verified_pickup",
    "picked_never_entered_requested_region",
    "entered_requested_region_not_released",
}
V3_FAILURE_TAXONOMY = {
    "correct",
    "pick_failed",
    "transport_failed",
    "wrong_side",
    "release_failed",
}


def _first_sustained(mask: list[bool], count: int = 3) -> int | None:
    run = 0
    for index, value in enumerate(mask):
        run = run + 1 if value else 0
        if run == count:
            return index - count + 1
    return None


def _xyz(sample: dict[str, Any], key: str) -> tuple[float, float, float]:
    value = sample.get(key)
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"retained capture sample has invalid {key}")
    return tuple(float(item) for item in value)


def _normalize_frozen_failure_stage(capture: dict[str, Any]) -> dict[str, Any]:
    """Repair only the known compile-time taxonomy/stage namespace mix-up.

    The retained capture remains immutable.  The returned in-memory copy derives
    the exact v2 DROID stage from its raw state samples before schema validation.
    """

    stage = capture.get("frozen_failure_stage")
    if stage in LEGACY_FAILURE_STAGES:
        return capture
    if stage not in V3_FAILURE_TAXONOMY:
        raise ValueError(f"unknown frozen_failure_stage {stage!r}")
    samples = capture.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("cannot derive legacy stage without retained state samples")
    if capture.get("requested_success") is True:
        derived = "success"
    else:
        object_xyz = [_xyz(sample, "object_xyz") for sample in samples]
        reference_xyz = [_xyz(sample, "reference_xyz") for sample in samples]
        initial = object_xyz[0]
        moved = [math.dist(initial, point) >= 0.01 for point in object_xyz]
        pickup = [point[2] - initial[2] >= 0.03 for point in object_xyz]
        relation = capture.get("requested_relation")
        if relation not in {"left", "right"}:
            raise ValueError("retained capture has invalid requested_relation")
        direction = 1.0 if relation == "left" else -1.0
        entered = False
        for obj, ref in zip(object_xyz, reference_xyz):
            forward = obj[0] - ref[0]
            lateral = obj[1] - ref[1]
            distance = math.hypot(forward, lateral)
            entered |= (
                distance > 1e-8
                and direction * lateral / distance >= math.cos(math.radians(45.0))
            )
        if _first_sustained(moved) is None:
            derived = "no_object_interaction"
        elif _first_sustained(pickup) is None:
            derived = "object_moved_no_verified_pickup"
        elif entered:
            derived = "entered_requested_region_not_released"
        else:
            derived = "picked_never_entered_requested_region"
    normalized = dict(capture)
    normalized["frozen_failure_stage"] = derived
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--release-manifest-sha256", required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--runtime-identity", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.release_manifest_sha256 != EXPECTED_SHA256["manifest"]
        or sha256_file(args.release_manifest) != EXPECTED_SHA256["manifest"]
    ):
        raise ValueError("DreamZero V3-B003 manifest binding changed")
    cell = load_cell(args.repo_root, args.cell_id)
    export = json.loads(args.export.read_text())
    if export.get("registered_cell_id") != cell.cell_id:
        raise ValueError("DreamZero V3-B003 simulator export names another cell")
    capture_path = Path(export["capture_path"])
    trace_path = Path(export["trace_manifest_path"])
    video_path = Path(export["viewport_video_path"])
    capture = _normalize_frozen_failure_stage(json.loads(capture_path.read_text()))
    phase_a_shape = {
        **cell.row,
        "pair_id": cell.row["matched_block_id"],
        "sampling_seed": cell.row["registered_sampling_seed_label"],
        "reset_identity": (
            f"v3b003:droid_robolab:{cell.arm}:environment_seed_{cell.seed}"
        ),
    }
    record = build_behavioral_record(
        args.repo_root,
        phase_a_shape,
        capture,
        args.runtime_identity,
        video_path,
        trace_path,
        args.output_jsonl,
    )
    record.update({
        "amendment_id": "V3-B003",
        "phase": "B_confound_ablation",
        "arm": cell.arm,
        "matched_block_id": cell.row["matched_block_id"],
        "failure_taxonomy": capture["failure_taxonomy"],
        "signed_final_lateral_offset_m": capture["signed_final_lateral_offset_m"],
        "requested_side_depth_m": capture["requested_side_depth_m"],
        "cone_entry_step": capture["cone_entry_step"],
        "cone_entry_sustained": capture["cone_entry_sustained"],
        "episode_length_steps": capture["episode_length_steps"],
        "time_to_first_contact": capture["first_contact_step"],
        "grasp_step": capture["grasp_step"],
        "object_path_length_m": capture["object_path_length_m"],
        "cumulative_lateral_path_m": capture["cumulative_lateral_path_m"],
        "peak_lateral_excursion_m": capture["peak_lateral_excursion_m"],
    })
    record["artifacts"]["reset_attestation"] = {
        "path": export["reset_attestation_path"],
        "sha256": sha256_file(Path(export["reset_attestation_path"])),
        "bytes": Path(export["reset_attestation_path"]).stat().st_size,
    }
    if args.output_jsonl.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_jsonl}")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    args.output_jsonl.write_text(payload)
    manifest = {
        "schema_version": "vla-wam-shared-v3b-dreamzero-cell-jsonl-manifest-v1",
        "registered_cell_id": cell.cell_id,
        "row_count": 1,
        "jsonl_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "jsonl_bytes": len(payload.encode()),
    }
    manifest_path = args.output_jsonl.with_name(args.output_jsonl.name + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"jsonl": str(args.output_jsonl), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
