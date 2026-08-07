#!/usr/bin/env python3
"""Build the immutable, model-blind reference-controller E002 registration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/vla_wam_shared_v3/phase_e/reference_controller_symmetry_v3e002"
SEEDS = list(range(9400, 9427))
ARMS = ("control", "position_mirrored")
RELATIONS = ("left", "right")
PROMPTS = {"left": "Put the Rubik's cube to the left of the bowl.", "right": "Put the Rubik's cube to the right of the bowl."}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bind(path_text: str) -> dict[str, object]:
    path = ROOT / path_text
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": path_text, "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    parent_paths = [
        "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/nano_mirror_v3b001_manifest.json",
        "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/post_result_nano_mirror_v3b001_amendment.json",
        "artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/pi05_mirror_v3b002_manifest.json",
        "artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/post_result_pi05_mirror_v3b002_amendment.json",
        "artifacts/vla_wam_shared_v3/phase_b/dreamzero_mirror_v3b003/dreamzero_mirror_v3b003_manifest.json",
        "artifacts/vla_wam_shared_v3/phase_b/dreamzero_mirror_v3b003/post_result_dreamzero_mirror_v3b003_amendment.json",
    ]
    queue = []
    for seed in SEEDS:
        for arm in ARMS:
            for relation in RELATIONS:
                queue.append({
                    "schema_version": "vla-wam-shared-v3e002-cell-v1",
                    "study_id": "vla_wam_language_steerability_v3",
                    "amendment_id": "V3-E002",
                    "cell_id": f"v3e002:reference_controller:seed{seed}:{arm}:{relation}",
                    "environment_seed": seed,
                    "arm": arm,
                    "relation": relation,
                    "prompt": PROMPTS[relation],
                    "fixture_source": "phase_b_positions_only_reflection_fixture",
                    "success_predicate_id": "v2_frozen_droid_robolab_release_inside_45deg_requested_relation",
                    "required_outputs": ["viewport_video", "raw_episode_jsonl", "final_state", "controller_metrics"],
                    "valid_failure_policy": "retain_all_behavioral_failures",
                })
    registration = {
        "schema_version": "vla-wam-shared-v3e002-registration-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-E002",
        "status": "registered_before_inference",
        "question": "Are LEFT and RIGHT placements under the control and reflected fixtures mechanically feasible under one verified symmetric reference controller?",
        "model_blind": True,
        "learned_model_request_count": 0,
        "environment_seed_range": [9400, 9426],
        "seeds": SEEDS,
        "arms": list(ARMS),
        "relations": list(RELATIONS),
        "expected_behavioral_episodes": 108,
        "candidate_target_depths_m": [0.075, 0.1, 0.15, 0.2],
        "selection_rule": "Select the largest candidate passing all four arm/relation conditions using static IK, collision, joint-limit, frame, predicate, and symmetric-waypoint gates before behavior.",
        "waypoint_recipe": ["pre_grasp", "grasp", "close_dwell", "vertical_lift", "symmetric_pre_place", "selected_depth", "release", "open_dwell"],
        "metrics": ["planner_ik_success", "task_success", "pickup", "requested_region_entry", "detached_release", "requested_side_depth", "endpoint_error", "task_space_path_length", "joint_space_path_length", "max_joint_velocity", "min_joint_limit_margin", "min_manipulability", "min_collision_clearance", "execution_duration", "failure_taxonomy"],
        "parent_bindings": [bind(path) for path in parent_paths],
        "queue": queue,
    }
    (OUT / "registration.json").write_text(json.dumps(registration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (OUT / "queue.jsonl").open("w", encoding="utf-8") as f:
        for row in queue:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"path": str(OUT / "registration.json"), "queue_path": str(OUT / "queue.jsonl"), "sha256": sha256(OUT / "registration.json"), "status": registration["status"]}, indent=2))


if __name__ == "__main__":
    main()
