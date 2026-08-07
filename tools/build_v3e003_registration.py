#!/usr/bin/env python3
"""Create the immutable, pre-inference V3-E003 registration and scene gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.v3.phase_e.bilateral_symmetry_null_control_v3e003.contract import (  # noqa: E402
    AMENDMENT_ID, ARENA, B001_AMENDMENT, B001_AMENDMENT_SHA256, B001_CELLS,
    B001_CELLS_SHA256, B001_MANIFEST, B001_MANIFEST_SHA256, CHECKPOINT_MANIFEST_SHA256,
    EQUIVALENCE_MARGIN, MODEL_ID, OPENPI_COMMIT, PHASE, PROMPTS, RELATIONS,
    ROBOLAB_COMMIT, SEEDS, SUCCESS_PREDICATE_ID, SYMMETRIC_POSITIONS,
    expected_queue, sha256_file,
)

BASE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/bilateral_symmetry_null_control_v3e003"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    queue = expected_queue(ROOT)
    registration = {
        "schema_version": "vla-wam-shared-v3e003-registration-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": AMENDMENT_ID,
        "phase": PHASE,
        "status": "registered_before_inference",
        "registered_at_utc": "2026-08-07T00:00:00Z",
        "model_id": MODEL_ID,
        "arena": ARENA,
        "model_request_count_before_registration": 0,
        "behavioral_episode_count_before_registration": 0,
        "design": {
            "matched_seed_count": len(SEEDS),
            "matched_seeds": list(SEEDS),
            "cells": 54,
            "cells_per_seed": 2,
            "relations": list(RELATIONS),
            "prompts": PROMPTS,
            "controller_and_predicate_frozen": True,
            "success_predicate_id": SUCCESS_PREDICATE_ID,
            "sign_convention": "+Y = robot-left",
            "held_fixed": ["robot base", "cameras", "all non-movable scene geometry", "controller", "cone definition", "prompt bytes"],
            "seed_source": "exact V3-B001 seed list and per-seed source randomization keys",
        },
        "scene_construction": {
            "name": "symmetric object layout relative to the robot midline",
            "not_symmetric_robot_or_embodiment": True,
            "caveat": "Robot joint configuration and wrist-camera mounting are not bilaterally symmetric; this is not a symmetric-embodiment control. The prior base-rotation control was invalid and is not reused.",
            "exactly_one_bowl": True,
            "movable_inventory": ["rubiks_cube", "bowl", "banana_left", "banana_right"],
            "reference_object": "bowl",
            "target_object": "rubiks_cube",
            "positions_robot_base_m": SYMMETRIC_POSITIONS,
            "construction": "B001 banana payload duplicated as a mirrored clutter pair in the new task scene; target and single reference are centered on y=0. Non-movable table, robot, cameras, and ground remain unchanged.",
            "tolerances_m": {"midline": 0.001, "pair_x": 0.001, "pair_y_sum": 0.001},
            "symmetry_gate_required_before_inference": True,
        },
        "registered_predictions": {
            "H1_binary_success_null": "binary LEFT/RIGHT success gap approximately zero; nondetection is not equivalence",
            "H2_requested_side_depth_null": "requested-side depth contrast approximately zero; nondetection is not equivalence",
            "H3_endpoint_redirection_positive_control": "paired endpoint shift remains strongly positive and comparable to control/reflected layouts",
            "reference_control_layout": {"left": "4/27", "right": "25/27", "source": "user-registered prior reference; existing committed summaries are retained separately and are not modified"},
        },
        "equivalence_margins": EQUIVALENCE_MARGIN,
        "analysis_plan": {
            "success": ["Wilson 95% interval", "exact McNemar with discordant counts"],
            "depth": ["mean and median RIGHT-minus-LEFT requested-side depth", "20,000 matched-seed bootstrap 95% CI", "exact two-sided sign test and sign counts", "TOST or CI against 0.05 m margin"],
            "endpoint": ["paired endpoint shift mean and median", "bootstrap 95% CI"],
            "failure_taxonomy": ["pick_failed", "transport_failed", "wrong_side", "release_failed", "correct"],
            "missingness": "Infrastructure-invalid attempts are separate; NR remains null and is never imputed.",
        },
        "source_bindings": {
            str(B001_CELLS): {"sha256": B001_CELLS_SHA256, "bytes": (ROOT / B001_CELLS).stat().st_size},
            str(B001_MANIFEST): {"sha256": B001_MANIFEST_SHA256, "bytes": (ROOT / B001_MANIFEST).stat().st_size},
            str(B001_AMENDMENT): {"sha256": B001_AMENDMENT_SHA256, "bytes": (ROOT / B001_AMENDMENT).stat().st_size},
            "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
            "openpi_commit": OPENPI_COMMIT,
            "robolab_commit": ROBOLAB_COMMIT,
        },
        "queue": queue,
        "release_boundary": "No model request or behavioral episode is authorized until the symmetry gate, exact runtime identity, reset gate, and raw-output write proof are hash-bound after this registration commit.",
    }
    write_json(BASE / "registration.json", registration)
    # The candidate is compact, reproducible input to the live task module.
    candidate = {
        "schema_version": "vla-wam-shared-v3e003-symmetric-object-layout-candidate-v1",
        "status": "model_blind_candidate_not_released_for_inference",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "exact_prompts": PROMPTS,
        "source_v3b001_cells_sha256": B001_CELLS_SHA256,
        "scene_asset": "rubiks_cube_banana_bowl.usda",
        "nonposition_state_sha256": "fab99e58180af0a57631b8a3e629b1e4b38b5f551090f66221e2c2865aa9af8f",
        "layout": "symmetric_object_layout",
        "positions_robot_base_m": SYMMETRIC_POSITIONS,
        "symmetry_residual": {"max_m": 0.0, "bowl_abs_y_m": 0.0, "cube_abs_y_m": 0.0, "banana_pair_y_sum_m": 0.0, "banana_pair_abs_x_diff_m": 0.0},
        "exactly_one_bowl": True,
        "robot_and_non_movable_geometry_unchanged": True,
    }
    write_json(BASE / "symmetry_gate/candidate.json", candidate)
    print(json.dumps({"registration": str(BASE / "registration.json"), "queue_cells": len(queue), "candidate": str(BASE / "symmetry_gate/candidate.json")}, indent=2))


if __name__ == "__main__":
    main()
