#!/usr/bin/env python3
"""Freeze the V3-E004 registration, layout candidate, and execution queue.

This is a model-blind builder.  It binds closed Phase-B controls without
rerunning them, prespecifies every new cell, and records the power boundary
before any E004 policy request or behavioral episode.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.candidate_builder import (  # noqa: E402
    build_from_spec,
)
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.layout_contract import (  # noqa: E402
    PoseSE2,
    canonical_json_bytes,
)


OUT = ROOT / "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004"
REGISTERED_AT = "2026-08-08T22:17:16Z"
CORE_SEEDS = tuple(range(9400, 9427))
RELATIONS = ("left", "right")
PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
ROBOTWIN_PROMPTS = {
    "left": "Put the small woodenblock to the left of the red playingcards box.",
    "right": "Put the small woodenblock to the right of the red playingcards box.",
}
SUCCESS_PREDICATE = "v2_frozen_droid_robolab_release_inside_45deg_requested_relation"

SOURCE_BINDINGS = {
    "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/nano_mirror_v3b001_cells.jsonl": "018b8b6ae76ac46f2f89eef83c4b16d7a4ff3d1ff15d91527b96fb56b5432c5a",
    "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/results/nano_v3b001_summary.json": "f43636a03caade5f3dc65de6736808c8257c78eacb07ba4cb963bfc6a0e36578",
    "artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/pi05_mirror_v3b002_cells.jsonl": "0db680a3aee04c991bcc78904cb572b7d962971e04fbc879e828354da30dafee",
    "artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/results/pi05_v3b002_report.json": "32ffa99f720906abe8679b0791be3f12d3c91dfce0274d85457a6d2ba59d2b71",
    "artifacts/vla_wam_shared_v3/phase_b/dreamzero_mirror_v3b003/dreamzero_mirror_v3b003_cells.jsonl": "a6d0f0a5d4c7cdfa5d3de95d44d7b11f42750a76a603ff8c2e44848e34b8f70d",
    "artifacts/vla_wam_shared_v3/phase_b/dreamzero_mirror_v3b003/results/dreamzero_v3b003_summary.json": "f45ab7d11281b994592948cdc11ea5d0fe24837d75d21a8a7dd42ff63b9c5817",
    "artifacts/vla_wam_shared_v3/results/cosmos3_edge_policy_droid_phase_a_summary.json": "fc93b6518185641068ce51a4d81882dd1828ebeb9227b44c4ffc21a07b526c87",
    "artifacts/vla_wam_shared_v3/prospective_tier_b/results/v3b007/fastwam_v3b007_summary.json": "1b9f91a0fb24ca5853ae0fe20f48dc3b784003d2169b6eae0b563d320d867d48",
    "artifacts/vla_wam_shared_v3/phase_e/bilateral_symmetry_null_control_v3e003/results.json": "06f61682bc296972c521e46d87972eae8cbab2840c97de9d9f1e9306dc2689a0",
}

SCENE_METADATA_SHA256 = "83ecf76a1fde9091b5db9012b76790aca36c2fe6b2c36a8885f4f98d7c4b7e1c"
ASSET_CUBE = f"rubiks_cube_banana_bowl.usda::rubiks_cube@{SCENE_METADATA_SHA256}"
ASSET_BOWL = f"rubiks_cube_banana_bowl.usda::bowl@{SCENE_METADATA_SHA256}"
ASSET_BANANA = f"rubiks_cube_banana_bowl.usda::banana@{SCENE_METADATA_SHA256}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
        for row in rows
    )


def pose_json(pose: PoseSE2) -> dict[str, Any]:
    return pose.to_json()


CONTROL = {
    "banana": PoseSE2(0.538878858089447, -0.07555567473173141, 0.0684281587600708, 0.00032307957706215006, ASSET_BANANA),
    "bowl": PoseSE2(0.44258353114128113, 0.12658219039440155, 0.07732785493135452, -0.0002099830569167544, ASSET_BOWL),
    "rubiks_cube": PoseSE2(0.303364634513855, 0.12396888434886932, 0.08113233000040054, 0.00016694049804018868, ASSET_CUBE),
}
SYMMETRIC = {
    "banana": PoseSE2(0.538878858089447, -0.22, 0.0684281587600708, 0.00032307957706215006, ASSET_BANANA),
    "banana_right": PoseSE2(0.538878858089447, 0.22, 0.0684281587600708, -0.00032307957706215006, ASSET_BANANA),
    "bowl": PoseSE2(0.44258353114128113, 0.0, 0.07732785493135452, 0.0, ASSET_BOWL),
    "rubiks_cube": PoseSE2(0.303364634513855, 0.0, 0.08113233000040054, 0.0, ASSET_CUBE),
}
COMPANION = {
    # The companion does not exist at s=0, so its interpolation anchor is a
    # registered counterfactual.  Anchor it at the collision-free symmetric
    # endpoint rather than on top of the sole B001 banana.  The earlier anchor
    # was rejected by the zero-request live gate before inference.
    "banana_right": SYMMETRIC["banana_right"]
}


def control_pose_digest() -> str:
    payload = json.dumps(
        {name: pose_json(pose) for name, pose in sorted(CONTROL.items())},
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


def layout_spec() -> dict[str, Any]:
    return {
        "schema_version": "vla-wam-shared-v3e004-layout-builder-input-v1",
        "registered_before_inference": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "control_poses": {name: pose_json(pose) for name, pose in sorted(CONTROL.items())},
        "symmetric_poses": {name: pose_json(pose) for name, pose in sorted(SYMMETRIC.items())},
        "companion_counterfactual_s0_poses": {name: pose_json(pose) for name, pose in sorted(COMPANION.items())},
        "orientation_invariant_objects": ["bowl"],
        "mirror_pairs": [["banana", "banana_right"]],
        "midline_objects": ["rubiks_cube", "bowl"],
        "target_object": "rubiks_cube",
        "reference_object": "bowl",
        "expected_cameras": ["head_camera", "over_shoulder_left_camera", "over_shoulder_right_camera", "wrist_cam"],
        "robot_base_xy_m": [0.0, 0.0],
        "asymmetry_weights": {"position_inverse_m": 10.0, "orientation_inverse_rad": 1.0},
        "realisation_position_tolerance_m": 0.005,
        "realisation_orientation_tolerance_rad": math.radians(2.0),
        "s0_frozen_control_attestation": {
            "inventory_policy": "exact_b001_inventory_and_poses",
            "inventory_transition": True,
            "source_fixture_id": "V3-B001/control",
            "source_fixture_sha256": "c5f3c667eda6f512b9e33beb5f7abc91700404feafa8b22279103b809dd238cd",
            "source_queue_sha256": SOURCE_BINDINGS["artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/nano_mirror_v3b001_cells.jsonl"],
            "source_candidate_sha256": "e1799b815da41f9a08a4000a360c4958003269fed27e2abe75b273519e4d1c88",
            "source_inventory": sorted(CONTROL),
            "control_poses_sha256": control_pose_digest(),
            "dose_response_primary_levels": [0.25, 0.5, 0.75, 1.0],
            "s0_analysis_role": "anchored_reference_not_in_primary_H3_slope",
            "design_limitation": "H1 s0-to-s1 includes the registered same-asset companion activation; the inventory-matched primary H3 slope excludes s=0.",
        },
        "source_bindings": {
            "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/nano_mirror_v3b001_cells.jsonl": SOURCE_BINDINGS["artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/nano_mirror_v3b001_cells.jsonl"],
            "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/model_blind_calibration_report.json": "112716acada89050561c9488d93a333a300b1675c2305329c8e5aceeb4e6da71",
        },
    }


MODELS: dict[str, dict[str, Any]] = {
    "pi05_current_stack_droid": {
        "short": "pi05",
        "arena": "droid_robolab",
        "levels": {0.0: 341, 0.25: 27, 0.5: 27, 0.75: 27, 1.0: 341},
        "preserved_s0_source": "artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/pi05_mirror_v3b002_cells.jsonl",
        "runtime": {"checkpoint": "pi05_droid_jointpos_polaris", "openpi_commit": "c23745b5ad24e98f66967ea795a07b2588ed6c79", "robolab_commit": "0aef241fb088ca21bb4ebd24448940ed56620d17", "checkpoint_manifest_sha256": "f5a56d9565f9381ccdeeaa165b0495dab6d17a81836cc7b01c5fbc6ab89e74ca", "action_horizon": 15, "action_dim": 8, "action_cap": 450},
    },
    "cosmos3_nano_policy_droid": {
        "short": "nano",
        "arena": "droid_robolab",
        "levels": {0.0: 521, 0.25: 27, 0.5: 27, 0.75: 27, 1.0: 521},
        "preserved_s0_source": "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/nano_mirror_v3b001_cells.jsonl",
        "runtime": {"checkpoint": "nvidia/Cosmos3-Nano-Policy-DROID", "checkpoint_revision": "6706d7680581c255ff61e0f3bb49d90eac55c79e", "server_repository_commit": "411d25b2e35bc441126f48c44a4b93e1c0564274", "robolab_commit": "0aef241fb088ca21bb4ebd24448940ed56620d17", "phase_a_runtime_identity_sha256": "d4bc4ab7d03fd1d1041f0bcc384d34321f3bd7b16c0c4cf517b62b8a1a2160e2", "action_horizon": 32, "action_cap": 450},
    },
    "dreamzero_droid_action_cfg": {
        "short": "dreamzero",
        "arena": "droid_robolab",
        "levels": {0.0: 27, 1.0: 27},
        "preserved_s0_source": "artifacts/vla_wam_shared_v3/phase_b/dreamzero_mirror_v3b003/dreamzero_mirror_v3b003_cells.jsonl",
        "runtime": {"checkpoint": "GEAR-Dreams/DreamZero-DROID", "checkpoint_revision": "96ad344138c66e82536422432ad742f015784942", "checkpoint_sha256": "b260d781c59b84a9a325184a31dcabd88f7874c69a7b5e514b78e6fabbc3af68", "identity_binding": "V2-A015:dreamzero_action_cfg_s2", "action_cfg_style_scale": 2, "video_cfg_scale": 5, "isolated_two_rank_server_required": True, "action_horizon": 24, "executed_prefix": 8, "action_cap": 450},
    },
    "cosmos3_edge_policy_droid": {
        "short": "edge",
        "arena": "droid_robolab",
        "levels": {0.0: 27, 1.0: 27},
        "runtime": {"checkpoint": "nvidia/Cosmos3-Edge-Policy-DROID", "checkpoint_revision": "3ea407af3e156c0af3b4bb6edd85842cc9a58777", "checkpoint_sha256": "b58d38088b3baad884a44ff9587ba10584a573f15e2cf7b08b836336cb53e48e", "phase_a_runtime_identity_sha256": "e92f68c02345042190a415a67e3eafbb12b35fded6d59d77074c74cb28ef1940", "action_horizon": 32, "action_cap": 450},
    },
    "fastwam_robotwin": {
        "short": "fastwam",
        "arena": "robotwin",
        "levels": {0.0: 27, 1.0: 27},
        "stretch": True,
        "runtime": {"checkpoint_family": "FastWAM", "source_release": "V3-B007", "source_release_queue_sha256": "2ffe2f99e4d6c4b3d80c24fab7276b21bb83de86d92b8a3438ce38a7ba9e1ae3", "separate_robotwin_fixture_gate_required": True},
    },
}


POWER = {
    "formula": "MDE80(n)=(z_0.95+z_0.80)*sigma_plan/sqrt(n); strict gate MDE80 <= 0.5*(0.20*abs(control effect))",
    "alpha_one_sided": 0.05,
    "power": 0.80,
    "z_sum": 2.48647,
    "variance_policy": "maximum seed-level SD among committed prior layouts for checkpoint and estimand",
    "rows": [
        {"model_id": "pi05_current_stack_droid", "estimand": "binary_R_minus_L", "control_effect": 0.7777777778, "sigma_plan": 0.5773502692, "margin": 0.1555555556, "mde_n27": 0.2763, "strict_n": 341, "target_n": 341, "status": "strictly_powered_at_endpoints"},
        {"model_id": "pi05_current_stack_droid", "estimand": "depth_R_minus_L_m", "control_effect": 0.2074701658, "sigma_plan": 0.1226599828, "margin": 0.0414940332, "mde_n27": 0.05870, "strict_n": 217, "target_n": 341, "status": "strictly_powered_at_endpoints"},
        {"model_id": "cosmos3_nano_policy_droid", "estimand": "binary_R_minus_L", "control_effect": 0.0, "sigma_plan": 0.3620, "margin": 0.0, "mde_n27": 0.1732, "strict_n": None, "target_n": 521, "status": "margin_zero_equivalence_not_defined_test_emergence_only"},
        {"model_id": "cosmos3_nano_policy_droid", "estimand": "depth_R_minus_L_m", "control_effect": 0.1476625642, "sigma_plan": 0.1354508319, "margin": 0.0295325128, "mde_n27": 0.06482, "strict_n": 521, "target_n": 521, "status": "strictly_powered_at_endpoints"},
        {"model_id": "dreamzero_droid_action_cfg", "estimand": "binary_R_minus_L", "control_effect": 0.1111111111, "sigma_plan": 0.6405, "margin": 0.0222222222, "mde_n27": 0.3065, "strict_n": 20546, "target_n": 27, "status": "underpowered_no_equivalence_claim"},
        {"model_id": "dreamzero_droid_action_cfg", "estimand": "depth_R_minus_L_m", "control_effect": 0.02295332, "sigma_plan": 0.1274, "margin": 0.004590664, "mde_n27": 0.06097, "strict_n": 19051, "target_n": 27, "status": "underpowered_no_equivalence_claim"},
        {"model_id": "cosmos3_edge_policy_droid", "estimand": "binary_R_minus_L", "control_effect": 0.2592592593, "sigma_plan": 0.5944, "margin": 0.0518518519, "mde_n27": 0.2844, "strict_n": 3250, "target_n": 27, "status": "underpowered_no_equivalence_claim"},
        {"model_id": "cosmos3_edge_policy_droid", "estimand": "depth_R_minus_L_m", "control_effect": 0.23731777, "sigma_plan": 0.1848, "margin": 0.047463554, "mde_n27": 0.08844, "strict_n": 375, "target_n": 27, "status": "underpowered_no_equivalence_claim"},
        {"model_id": "fastwam_robotwin", "estimand": "binary_R_minus_L", "control_effect": 0.1481481481, "sigma_plan": 0.4804, "margin": 0.0296296296, "mde_n27": 0.2299, "strict_n": 6501, "target_n": 27, "status": "underpowered_no_equivalence_claim_stretch"},
        {"model_id": "fastwam_robotwin", "estimand": "depth_R_minus_L_m", "control_effect": 0.3530, "sigma_plan": 0.2152, "margin": 0.07060, "mde_n27": 0.1030, "strict_n": 230, "target_n": 27, "status": "underpowered_no_equivalence_claim_stretch"},
    ],
    "policy_deviation": "DreamZero, Edge, and FastWAM require thousands of matched pairs under the prespecified half-margin MDE gate. Their 27-pair cores test positive interactions but cannot support H2 equivalence. No nonsignificant result from those rows will be called equivalent.",
}


def load_closed_controls(path: str) -> dict[tuple[int, str], dict[str, Any]]:
    rows = [json.loads(line) for line in (ROOT / path).read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = {
        (int(row["environment_seed"]), str(row["relation"])): row
        for row in rows
        if row.get("arm") == "control" and int(row["environment_seed"]) in CORE_SEEDS
    }
    if len(selected) != 54:
        raise RuntimeError(f"closed control source {path} is not 27 matched pairs")
    return selected


def queue_rows(candidate: dict[str, Any], candidate_sha256: str) -> list[dict[str, Any]]:
    historical_comparators: dict[str, dict[tuple[int, str], dict[str, Any]]] = {}
    for model_id, config in MODELS.items():
        if config.get("preserved_s0_source"):
            historical_comparators[model_id] = load_closed_controls(config["preserved_s0_source"])

    rows: list[dict[str, Any]] = []
    for model_id, config in MODELS.items():
        prompts = ROBOTWIN_PROMPTS if config["arena"] == "robotwin" else PROMPTS
        for level, count in config["levels"].items():
            seeds = range(9400, 9400 + count)
            for seed in seeds:
                for relation in RELATIONS:
                    has_historical_comparator = (
                        level == 0.0
                        and seed in CORE_SEEDS
                        and model_id in historical_comparators
                    )
                    level_code = f"{int(round(level * 100)):03d}"
                    row: dict[str, Any] = {
                        "schema_version": "vla-wam-shared-v3e004-cell-v1",
                        "study_id": "vla_wam_language_steerability_v3",
                        "amendment_id": "V3-E004",
                        "model_id": model_id,
                        "arena": config["arena"],
                        "cell_id": f"v3e004:{config['short']}:seed{seed}:s{level_code}:{relation}",
                        "matched_pair_id": f"v3e004:{config['short']}:seed{seed}:s{level_code}",
                        "environment_seed": seed,
                        "sampling_seed": seed,
                        "symmetry_level_s": level,
                        "relation": relation,
                        "prompt": prompts[relation],
                        "prompt_sha256": sha256_bytes(prompts[relation].encode("utf-8")),
                        "static_episode_prompt": True,
                        "execution_mode": "new_behavioral_episode",
                        "runtime_identity_requirement": config["runtime"],
                        "success_predicate_id": SUCCESS_PREDICATE if config["arena"] == "droid_robolab" else "frozen_v3b007_robotwin_relation_aware_success",
                        "layout_candidate_sha256": candidate_sha256 if config["arena"] == "droid_robolab" else None,
                        "registered_expected_asymmetry_A": candidate["levels"][f"{level:.2f}"]["asymmetry_metric_A"] if config["arena"] == "droid_robolab" else None,
                        "required_raw_outputs": ["simulator_video", "executed_action_trace", "raw_episode_jsonl", "final_state", "realised_object_poses", "arm_reset_pose"],
                        "required_episode_fields": ["success", "failure_category", "signed_final_lateral_offset", "requested_side_depth", "cone_entry_step", "cone_entry_sustained", "endpoint_shift", "action_distinct", "episode_length", "time_to_first_contact", "grasp_step", "cumulative_lateral_path", "peak_lateral_excursion", "symmetry_level_s", "asymmetry_metric_A", "position_residual", "orientation_residual", "midline_residual", "occlusion_check", "realised_object_poses", "arm_reset_pose"],
                        "behavioral_failure_policy": "retain_in_denominator",
                        "infrastructure_failure_policy": "separate_stream_excluded_from_behavioral_denominator",
                        "missing_measurement_policy": "NR remains null and is never converted to zero",
                        "release_status": "not_released_pending_registration_commit_and_all_model_blind_runtime_gates",
                    }
                    if has_historical_comparator:
                        source = historical_comparators[model_id][(seed, relation)]
                        row["historical_control_comparator_cell_id"] = source["cell_id"]
                        row["historical_control_comparator_fixture_sha256"] = source["fixture_sha256"]
                        row["historical_control_comparator_queue_sha256"] = SOURCE_BINDINGS[config["preserved_s0_source"]]
                        row["historical_control_comparator_not_an_e004_cell"] = True
                    rows.append(row)

    # Prespecify deterministic execution order within every model/seed for new
    # cells only.  Preserved controls already have immutable historical order.
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault((row["model_id"], row["environment_seed"]), []).append(row)
    for key, bucket in buckets.items():
        bucket.sort(key=lambda row: sha256_bytes(f"v3e004-order-v1:{row['cell_id']}".encode("utf-8")))
        for index, row in enumerate(bucket, 1):
            row["execution_order_index_within_model_seed"] = index
            row["execution_order_block"] = f"{key[0]}:seed{key[1]}"
    rows.sort(key=lambda row: (row["model_id"], row["environment_seed"], row["symmetry_level_s"], row["relation"]))
    return rows


def main() -> None:
    for relative, expected in SOURCE_BINDINGS.items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"immutable source binding changed: {relative}")

    spec = layout_spec()
    spec_payload = canonical_json_bytes(spec)
    candidate = build_from_spec(spec, repo_root=ROOT)
    candidate_payload = canonical_json_bytes(candidate)
    candidate_sha256 = sha256_bytes(candidate_payload)
    rows = queue_rows(candidate, candidate_sha256)
    queue_payload = jsonl_bytes(rows)
    queue_sha256 = sha256_bytes(queue_payload)

    total = len(rows)
    comparator_count = sum("historical_control_comparator_cell_id" in row for row in rows)
    new_count = sum(row["execution_mode"] == "new_behavioral_episode" for row in rows)
    if (total, comparator_count, new_count) != (4096, 162, 4096):
        raise RuntimeError(f"unexpected queue counts: {(total, comparator_count, new_count)}")

    registration = {
        "schema_version": "vla-wam-shared-v3e004-registration-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-E004",
        "supersedes": "V3-E003",
        "prior_evidence_policy": "V3-E003 retained as a superseded first pass; no prior file or predicate is modified",
        "status": "prospectively_registered_zero_e004_model_requests_or_behavioral_episodes",
        "registered_at_utc": REGISTERED_AT,
        "pre_inference_correction": {
            "recorded_at_utc": "2026-08-08T22:28:21Z",
            "model_requests_before_correction": 0,
            "behavioral_episodes_before_correction": 0,
            "reason": "Closed B001/B002/B003 controls lack E004 realised poses, A, residuals, camera occlusion, and arm-reset fields. They remain historical comparators but cannot satisfy the new raw logging contract.",
            "effect": "All 4,096 registered E004 cells are new behavioral measurements; no missing historical field is imputed or encoded as zero.",
        },
        "pre_inference_layout_gate_correction": {
            "recorded_at_utc": "2026-08-08T23:04:33Z",
            "model_requests_before_correction": 0,
            "behavioral_episodes_before_correction": 0,
            "reason": "Zero-request Isaac gates showed that the provisional s>0 companion anchor placed the two identical bananas in contact at intermediate levels (peak residual speeds 0.184799552 m/s at s=.25 and 0.100256555 m/s at s=.50), while the exact B001 control settled 4.253151 mm from its commanded pose. The circular bowl also changed yaw by 0.114287321 rad under settling although its yaw is physically and visually invariant.",
            "effect": "The absent companion's counterfactual interpolation anchor is frozen at its collision-free s=1 pose; the live position-realisation tolerance is prospectively set to 5 mm; bowl yaw is explicitly orientation-invariant. Full-symmetry residual gates remain position <1 mm, mirrored-clutter orientation <0.5 degrees, midline <1 mm, and no target occlusion in every camera.",
            "failed_gate_evidence": [
                {"path": "/data/users/ali/vla_wam/raw/v3e004/gates/nano_seed9400_s000_left_smoke_attempt02.log", "sha256": "d6565bb7d8367c7277f8629cc4fd68af90edb2c619c721e8cf55d216e36901df"},
                {"path": "/data/users/ali/vla_wam/raw/v3e004/gates/nano_seed9400_s025_left_smoke_attempt03.log", "sha256": "ba6730fbc792b037d122fa2b6a580b94246d0e592f8f27ff1af844a30369b1c3"},
                {"path": "/data/users/ali/vla_wam/raw/v3e004/gates/nano_seed9400_s050_left_smoke_attempt03.log", "sha256": "dc1e970c695b4e7ebf5c3ee261655feb5a53d787282a10c29427c4b560b858a8"},
                {"path": "/data/users/ali/vla_wam/raw/v3e004/gates/nano_seed9400_s075_left_smoke_attempt03.log", "sha256": "3825da54cd8ebf93105bbf1bdcf9e3e5b40bcf959a228b08cd597f9dcc011be0"},
                {"path": "/data/users/ali/vla_wam/raw/v3e004/gates/nano_seed9400_s100_left_smoke_attempt01/model_blind_gate_report.json", "sha256": "13f09a907505dffd5c4a8525728a7d2c60cec079687bc2ee1464ee103bce1083"}
            ],
        },
        "model_request_count_before_registration": 0,
        "behavioral_episode_count_before_registration": 0,
        "exact_prompts": {"droid": PROMPTS, "robotwin": ROBOTWIN_PROMPTS},
        "sign_convention": "+Y = robot-left",
        "success_predicates_frozen": True,
        "design": {
            "core_seeds": list(CORE_SEEDS),
            "core_seed_source": "exact V3-B001 seed list",
            "extension_seed_boundary": "Seeds above 9426 are matched within E004 but are not historical B001 matches; closed geometry ablations are not rerun.",
            "model_levels_and_pairs": {model_id: {f"{level:.2f}": count for level, count in config["levels"].items()} for model_id, config in MODELS.items()},
            "total_evidence_cells": total,
            "historical_control_comparator_links_not_e004_cells": comparator_count,
            "new_behavioral_cells": new_count,
            "droid_and_robotwin_never_pooled": True,
        },
        "layout": {
            "name": "graded symmetric object layout relative to robot midline",
            "not_symmetric_robot_or_embodiment": True,
            "arm_reset_pose_required": True,
            "candidate_path": str((OUT / "layout/candidate.json").relative_to(ROOT)),
            "candidate_sha256": candidate_sha256,
            "builder_input_path": str((OUT / "layout/builder_input.json").relative_to(ROOT)),
            "builder_input_sha256": sha256_bytes(spec_payload),
            "asymmetry_metric": candidate["asymmetry_metric"],
            "inventory_transition": candidate["companion_activation_policy"],
            "s1_tolerances": candidate["tolerances"],
            "target_visibility_rule": "cube is in front of bowl and bowl must not lie between any registered camera and cube; target segmentation must be visible in every view",
        },
        "registered_predictions": {
            "H1_interaction": "Within checkpoint, the directional binary/depth gap shrinks from s=0 to s=1; Nano binary and DreamZero binary instead test prospectively named gap emergence because control is at ceiling/near floor.",
            "H2_equivalence": "At s=1, compare binary and depth gaps against 0.20 times each checkpoint's committed control effect. Equivalence is claimed only for rows whose preregistered strict MDE gate is met.",
            "H3_dose_response": "For pi05 and Nano, gap-versus-realised-A slope is positive when A decreases toward symmetry; primary slope uses inventory-matched s=.25,.50,.75,1, with s=0 shown as anchored reference.",
            "H4_positive_control": "Endpoint redirection remains positive at every level. Failure closes that checkpoint/level against an equalisation interpretation.",
            "H5_failure_signature": "Among failures, wrong-side share increases as symmetry rises while pick/transport share decreases.",
        },
        "estimands": {
            "binary_gap": "Y_i,l = success_RIGHT_i,l - success_LEFT_i,l",
            "requested_depth_gap": "B_i,l = (-signed_endpoint_RIGHT_i,l) - signed_endpoint_LEFT_i,l",
            "endpoint_redirection": "D_i,l = signed_endpoint_LEFT_i,l - signed_endpoint_RIGHT_i,l",
            "interaction": "mean(seed-level estimand at s=1 minus same-seed estimand at s=0)",
        },
        "power_registration": POWER,
        "analysis_plan": {
            "per_cell": ["LEFT/RIGHT counts and Wilson 95% intervals", "exact McNemar with discordant counts", "paired requested-depth mean/median with 20,000 seed bootstrap CI and exact sign test", "paired endpoint shift with bootstrap CI and sign test"],
            "interaction": "Seed-matched s1-minus-s0 bootstrap CI plus exact 2^27 within-seed layout-label permutation on the historical core; augmented endpoints are used for TOST/CI only.",
            "equivalence": "TOST with registered margin, paired 90% CI, bootstrap 95% CI, achieved MDE, and explicit comparison to control effect. Underpowered rows cannot support equivalence wording.",
            "dose_response": "Per-seed slope on realised A for pi05/Nano inventory-matched positive levels, seed bootstrap CI, and slope signs.",
            "failures": "Direction/level taxonomy plus seed-clustered wrong-side-share trend with within-seed level-label permutation; zero-failure cells are unavailable, not zero.",
            "missingness": "Infrastructure-invalid attempts are separate; behavioral failures remain in denominators; NR is never imputed.",
        },
        "source_bindings": {path: {"sha256": digest, "bytes": (ROOT / path).stat().st_size} for path, digest in SOURCE_BINDINGS.items()},
        "queue": {"path": str((OUT / "queue.jsonl").relative_to(ROOT)), "sha256": queue_sha256, "bytes": len(queue_payload), "rows": total},
        "release_boundary": "No E004 model request or new behavioral episode is authorized until this registration/candidate/queue commit is pushed and the model-blind static, live camera/reset, raw-write, renderer, and exact runtime gates pass for that lane.",
    }

    write_bytes(OUT / "layout/builder_input.json", spec_payload)
    write_bytes(OUT / "layout/candidate.json", candidate_payload)
    write_bytes(OUT / "queue.jsonl", queue_payload)
    write_bytes(OUT / "registration.json", canonical_json_bytes(registration))
    print(json.dumps({"registration": str(OUT / "registration.json"), "candidate_sha256": candidate_sha256, "queue_sha256": queue_sha256, "evidence_cells": total, "new_behavioral_cells": new_count}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
