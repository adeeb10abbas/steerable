#!/usr/bin/env python3
"""Hash-bind the exact Nano V3-B005 queue after its zero-request physical gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


AMENDMENT_SHA256 = "ff23475b53791c42715938d51a303e0ab82de88b1b8a7a30758c008c9919a47b"
FIXTURE_SHA256 = "87ff070be25b61538ead16ddbe06d2e9c155698ec2ea8acecbc30bd20b0197a5"
EXPECTED_LEVELS = [
    0.03658219039440155, 0.06658219039440155, 0.09658219039440155,
    0.12658219039440155, 0.15658219039440155, 0.18658219039440156,
    0.21658219039440155,
]
PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(row: dict) -> str:
    return json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--safe-fixture", type=Path, required=True)
    parser.add_argument("--physical-gate", type=Path, required=True)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.amendment) != AMENDMENT_SHA256 or sha256(args.safe_fixture) != FIXTURE_SHA256:
        raise ValueError("V3-B005 preregistration binding changed")
    amendment = json.loads(args.amendment.read_text())
    gate = json.loads(args.physical_gate.read_text())
    if (
        gate.get("schema_version")
        != "vla-wam-shared-v3b-nano-lateral-model-blind-calibration-v1"
        or gate.get("amendment_id") != "V3-B005"
        or gate.get("passed") is not True
        or gate.get("model_request_count") != 0
        or gate.get("behavioral_episode_count") != 0
        or gate.get("dense_scan", {}).get("row_count") != 42
        or gate.get("dense_scan", {}).get("passing_candidate_y_m") != EXPECTED_LEVELS
        or gate.get("selection", {}).get("ordered_seven_levels_y_m") != EXPECTED_LEVELS
        or gate.get("selection", {}).get("banana_y_override_m") != -0.2755556747317314
    ):
        raise ValueError("V3-B005 physical gate did not pass the exact registered design")
    conditions = [(level, relation) for level in range(7) for relation in ("left", "right")]
    rows = []
    for seed in range(9500, 9515):
        if seed < 9514:
            shift = seed - 9500
            order = conditions[shift:] + conditions[:shift]
            randomization = "14-seed cyclic Latin rotation"
        else:
            order = sorted(conditions, key=lambda item: hashlib.sha256(
                f"vla_wam_language_steerability_v3:V3-B005:nano:seed9514:{item[0]}:{item[1]}:order".encode()
            ).hexdigest())
            randomization = "prospectively specified SHA-256 order"
        for position, (level_index, relation) in enumerate(order, 1):
            prompt = PROMPTS[relation]
            cell_id = f"v3b005:nano:seed{seed}:level{level_index}:{relation}"
            rows.append({
                "schema_version": "vla-wam-shared-v3b-nano-lateral-cell-v1",
                "study_id": "vla_wam_language_steerability_v3",
                "amendment_id": "V3-B005",
                "phase": "B_confound_ablation",
                "arena": "droid_robolab",
                "model_id": "cosmos3_nano_policy_droid",
                "cell_id": cell_id,
                "matched_block_id": f"v3b005:nano:seed{seed}:level{level_index}",
                "seed_block_id": f"v3b005:nano:seed{seed}",
                "environment_seed": seed,
                "sampling_seed": seed,
                "level_index": level_index,
                "reference_object_initial_lateral_position_y_m": EXPECTED_LEVELS[level_index],
                "relation": relation,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "execution_order_index_within_seed": position,
                "randomization": randomization,
                "safe_distractor_fixture_sha256": FIXTURE_SHA256,
                "physical_gate_sha256": sha256(args.physical_gate),
                "success_predicate_id": "v2_frozen_droid_robolab_release_inside_45deg_requested_relation",
                "runtime_identity_requirement": amendment["runtime_identity_requirement"],
                "execution_status": "registered_after_physical_gate_runtime_and_fixed_observation_release_required",
                "valid_failure_policy": "retain every valid behavioral failure",
                "technical_invalidity_policy": "separate stream; repair only the identical registered cell",
            })
    if len(rows) != 210 or len({row["cell_id"] for row in rows}) != 210:
        raise RuntimeError("V3-B005 queue construction did not produce 210 unique cells")
    if args.cells.exists() or args.manifest.exists():
        raise FileExistsError("refusing to overwrite V3-B005 queue evidence")
    args.cells.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(canonical(row) + "\n" for row in rows)
    args.cells.write_text(payload)
    position_counts = {}
    for level, relation in conditions:
        key = f"level{level}:{relation}"
        position_counts[key] = {
            str(position): sum(
                row["level_index"] == level
                and row["relation"] == relation
                and row["execution_order_index_within_seed"] == position
                for row in rows
            )
            for position in range(1, 15)
        }
    manifest = {
        "schema_version": "vla-wam-shared-v3b-nano-lateral-manifest-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-B005",
        "status": "hash_bound_after_passed_physical_gate_runtime_release_required",
        "counts": {"matched_seeds": 15, "levels": 7, "relations": 2, "registered_cells": 210},
        "amendment_sha256": AMENDMENT_SHA256,
        "safe_fixture_sha256": FIXTURE_SHA256,
        "physical_gate": {
            "path": str(args.physical_gate.name),
            "sha256": sha256(args.physical_gate),
            "model_request_count": 0,
            "behavioral_episode_count": 0,
        },
        "cells": {"path": str(args.cells.name), "sha256": hashlib.sha256(payload.encode()).hexdigest(), "row_count": 210},
        "execution_order_position_counts": position_counts,
        "behavioral_release": False,
        "next_gate": "Fresh Nano runtime identity and fixed-observation exact-repeat/prompt-sensitivity gate; no behavioral cell is released yet.",
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "cells_sha256": sha256(args.cells),
        "manifest_sha256": sha256(args.manifest),
        "physical_gate_sha256": sha256(args.physical_gate),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
