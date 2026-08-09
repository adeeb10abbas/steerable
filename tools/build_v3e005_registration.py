#!/usr/bin/env python3
"""Build the prospective V3-E005 registration and exact 108-cell queue.

This command is model-blind.  It reads only committed historical evidence and
must run before any E005 model request or behavioral episode.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/vla_wam_shared_v3/phase_e/cross_arena_geometry_v3e005"
REGISTERED_AT = "2026-08-09T14:14:44Z"
SEEDS = tuple(range(9400, 9427))
LEVELS = (0.0, 1.0)
RELATIONS = ("left", "right")
SCENES = tuple(range(3, 10))
Z_SUM = 2.48647486

SOURCE_BINDINGS = {
    "artifacts/vla_wam_shared_v3/results/lingbot_va_robotwin_phase_a_summary.json": "ebffedaeaf42fb8916449136f617aaa9dab149645caad45aa461dec7aef878e3",
    "artifacts/vla_wam_shared_v3/results/lingbot_va_robotwin_phase_a_evidence_hash_manifest.json": "34ec6bf63ef825706f4ccc8bbf2b1d57854ebaa485879b54875b697ddea96cc4",
    "artifacts/vla_wam_shared_v3/results/efficient_wam_rt_robotwin_phase_a_summary.json": "5796bdbedde6124b9a849fa4ab9785171c0caddd5fd18d28e67706e30ed554e5",
    "artifacts/vla_wam_shared_v3/results/fastwam_robotwin_phase_a_summary.json": "da76cf534907714c1b9e74e096ed39578b308642504e145f40e834ff4ba65b4e",
    "artifacts/vla_wam_shared_v3/robotwin_direct_registry.json": "2a840a6eaa418980f8237f5f8ab522028d4b4453fc2133dd0c0180ac9d6be8b5",
    "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004/results/results.json": "be8cd0c9a2458a7eefec0ed7760d1e6f6f19e4162d8c427f12f8a113d3290493",
    "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004/results/pairs.jsonl": "fe2d7d8879314122a48aa269374118a45d13f204c9c83691735920e917d68bcc",
}

RUNTIME = {
    "model_id": "lingbot_va_robotwin",
    "checkpoint_id": "lerobot/lingbot_va_robotwin",
    "checkpoint_revision": "lerobot/lingbot_va_robotwin@d1e1f93a84eaf9bca9880856fda800cc98cc8eaa;robbyant/lingbot-va-posttrain-robotwin@8c9dea8abbc5c91cc9e18bc3264b8915083bbe70",
    "checkpoint_manifest_sha256": "91d32f57b7edbb9b624ef5e64e0440177c529a3bf099fc1aae5c51d1ac847c18",
    "runtime_payload_sha256": "f2af400c5d6fac539564c7d2a0f3ff76479f98120896e333bc50cf3615e41e89",
    "environment_lock_sha256": "a274ecfe0d4cede35350ee94d53710ead30137ae2e75e1dd21bfee7c06aaa768",
    "adapter_contract_sha256": "59189aeec5f1454fbe0467558ba4eefa47f67641d8ed0283a1c760304b60a53d",
    "external_repository_commit": "d42efbc04e502057dab4b18bb14770cc48e85131",
    "simulator_repository_commit": "0aeea2d669c0f8516f4d5785f0aa33ba812c14b4",
    "renderer": "headless SAPIEN Vulkan on NVIDIA RTX PRO 6000 Blackwell",
}


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json(row) for row in rows)


def historical_prompts() -> dict[int, dict[str, str]]:
    source = json.loads((ROOT / "artifacts/vla_wam_shared_v3/results/lingbot_va_robotwin_phase_a_summary.json").read_text())
    prompts: dict[int, dict[str, str]] = {}
    for row in source["v3_primary_results"]["episodes"]:
        pair = int(row["pair"])
        if pair in SCENES:
            prompts.setdefault(pair, {})[str(row["relation"])] = str(row["prompt"])
    if set(prompts) != set(SCENES) or any(set(v) != set(RELATIONS) for v in prompts.values()):
        raise RuntimeError("committed LingBot source does not contain one exact LEFT/RIGHT prompt per scene")
    return prompts


def scene_for_seed(seed: int) -> int:
    return SCENES[(seed - SEEDS[0]) % len(SCENES)]


def anchor_task(scene: int) -> str:
    return "place_a2b_right" if scene % 2 else "place_a2b_left"


def build_queue(prompts: dict[int, dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        scene = scene_for_seed(seed)
        for level in LEVELS:
            level_code = f"{int(level * 100):03d}"
            for relation in RELATIONS:
                prompt = prompts[scene][relation]
                rows.append(
                    {
                        "schema_version": "vla-wam-shared-v3e005-cell-v1",
                        "study_id": "vla_wam_language_steerability_v3",
                        "amendment_id": "V3-E005",
                        "cell_id": f"v3e005:lingbot:seed{seed}:scene{scene:02d}:s{level_code}:{relation}",
                        "matched_seed_id": f"v3e005:lingbot:seed{seed}",
                        "matched_layout_pair_id": f"v3e005:lingbot:seed{seed}:s{level_code}",
                        "model_id": "lingbot_va_robotwin",
                        "arena": "robotwin",
                        "scene_id": f"robotwin_pair_{scene:02d}",
                        "scene_cluster_id": f"robotwin_pair_{scene:02d}",
                        "anchor_task": anchor_task(scene),
                        "environment_seed": seed,
                        "sampling_seed": seed,
                        "symmetry_level_s": level,
                        "layout": "control" if level == 0.0 else "symmetric_object_layout",
                        "relation": relation,
                        "prompt": prompt,
                        "prompt_sha256": sha256_bytes(prompt.encode()),
                        "static_episode_prompt": True,
                        "success_predicate_id": "frozen_v3_robotwin_relation_aware_success",
                        "outcome_coordinate_contract": "frozen_robotwin_native_lateral_axis_and_region",
                        "layout_coordinate_contract": "calibrated_robot_frame_y_midline",
                        "runtime_identity_requirement": RUNTIME,
                        "execution_mode": "new_behavioral_episode",
                    }
                )
    return rows


def build_registration(queue_sha256: str, prompts: dict[int, dict[str, str]]) -> dict[str, Any]:
    scene_counts = {f"robotwin_pair_{scene:02d}": sum(scene_for_seed(seed) == scene for seed in SEEDS) for scene in SCENES}
    return {
        "schema_version": "vla-wam-shared-v3e005-registration-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-E005",
        "title": "Cross-arena geometry replication (RoboTwin, LingBot-VA)",
        "status": "registered_before_any_e005_model_request_or_behavioral_episode",
        "registered_at_utc": REGISTERED_AT,
        "model_request_count_before_registration": 0,
        "behavioral_episode_count_before_registration": 0,
        "separate_amendment_not_e004_extension": True,
        "final_cohort_stopping_rule": "On completion, regardless of outcome, the experimental phase closes and the manuscript is written against the result.",
        "selection": {
            "rule": "checkpoint chosen on committed endpoint-alignment positive-control proxy, not E005 geometry outcome",
            "aligned_pairs_of_63": {"lingbot_va_robotwin": 47, "efficient_wam_rt_robotwin": 42, "fastwam_robotwin": 39},
            "no_checkpoint_substitution_if_h4_fails": True,
        },
        "design": {
            "arena": "robotwin",
            "never_pool_with_droid": True,
            "model_id": "lingbot_va_robotwin",
            "seeds": list(SEEDS),
            "seed_source": "prospectively fixed consecutive list 9400-9426; no E005 behavior existed at registration",
            "scene_assignment_rule": "scene_pair = 3 + ((seed - 9400) mod 7)",
            "scene_seed_counts": scene_counts,
            "levels": list(LEVELS),
            "directions": list(RELATIONS),
            "matched_seeds": 27,
            "behavioral_episodes": 108,
            "formula": "27 seeds * 2 layouts * 2 directions",
            "whole_seed_atomic_block": "Each seed's s0 LEFT/RIGHT and s1 LEFT/RIGHT cells remain on one lane.",
            "optional_reflected_stage": "not_registered_and_not_authorized; may be separately registered only after H4 passes at both levels",
        },
        "prompts_by_scene": {f"robotwin_pair_{k:02d}": v for k, v in sorted(prompts.items())},
        "frozen_task_contract": {
            "success_predicate": "RoboTwin's frozen relation-aware success predicate",
            "controller": "LingBot-VA frozen RoboTwin controller",
            "relation_definition": "RoboTwin native region/cone and sign convention",
            "prohibited": ["DROID_45_degree_cone", "DROID_success_predicate", "cross_arena_pooling"],
        },
        "layout": {
            "name": "symmetric object layout relative to the robot midline",
            "never_call_unqualified_symmetric": True,
            "not_symmetric_robot_embodiment_joint_configuration_or_camera_mount": True,
            "scene_specific_not_global": True,
            "control": "exact frozen RoboTwin scene reset for the assigned pair",
            "symmetric": {
                "reference": "one reference instance centered at calibrated robot-frame y=0",
                "target": "target centered at calibrated robot-frame y=0 and visible from every registered camera",
                "clutter": "all other movable objects occur in same-asset/same-scale/same-material mirrored pairs",
                "se2_reflection": "y_b=-y_a; x_b=x_a; wrapped_yaw_b=-wrapped_yaw_a",
            },
            "coordinate_boundary": "Layout symmetry is checked in calibrated robot-frame y. Behavioral endpoints retain RoboTwin's frozen native outcome coordinate; no DROID axis is imported.",
            "strict_tolerances": {
                "position_residual_m": 0.001,
                "orientation_residual_rad": math.radians(0.5),
                "orientation_residual_deg": 0.5,
                "midline_residual_m": 0.001,
                "occlusion_check": False,
            },
            "required_logs": ["realised_object_poses", "arm_reset_pose", "asset_identity", "scale", "material", "all_camera_occlusion_checks"],
            "fail_closed": True,
        },
        "runtime_identity_requirement": RUNTIME,
        "predictions": {
            "H4": {
                "role": "positive_control_hard_gate_evaluated_first",
                "estimand": "paired endpoint redirection = endpoint_LEFT - endpoint_RIGHT in frozen RoboTwin outcome coordinates",
                "pass_at_each_level": "mean > 0.05 m and scene-clustered 95% bootstrap CI excludes zero",
                "threshold_m": 0.05,
                "if_fail": "Do not interpret H1-H3; stop; supplement beside FastWAM; do not run reflected stage or another checkpoint.",
            },
            "H1": {
                "estimands": ["binary success gap RIGHT-minus-LEFT", "requested-depth gap RIGHT-minus-LEFT"],
                "interaction": "seed-matched (s1 gap - s0 gap), reported separately for binary and depth",
                "prediction": "directional gap is smaller at s1 than s0",
                "inference": "20,000-resample scene-clustered seed bootstrap CI and exact within-seed layout-label permutation",
            },
            "H2": {
                "level": 1.0,
                "binary": {"historical_control_effect": 0.0, "sigma_plan": 0.762000762001143, "margin": 0.0, "mde80_n27": 0.3646343647921233, "status": "zero_margin_tost_undefined_underpowered_no_equivalence_claim"},
                "requested_depth_m": {"historical_control_effect": 0.02582497987157059, "sigma_plan": 0.9945449095448322, "margin": 0.0051649959743141185, "mde80_n27": 0.47591192743266003, "status": "mde_exceeds_half_margin_underpowered_no_equivalence_claim"},
                "margin_rule": "0.20 * abs(LingBot-VA committed Phase-A control effect)",
                "mde_rule": "(z_0.95 + z_0.80) * observed paired seed SD / sqrt(27), z sum 2.48647486",
                "nondetection_is_not_equivalence": True,
            },
            "H3": {"prediction": "wrong-side share of failures rises s0-to-s1 while pick and transport share falls", "missing_share_policy": "no failures means unavailable, never zero"},
        },
        "analysis": {
            "h4_must_be_compiled_and_recorded_before_h1_h3": True,
            "bootstrap_resamples": 20000,
            "scene_clustered_intervals_required": True,
            "nested_scene_warning": "27 seed replicates are nested within seven scenes and are not independent environments.",
            "per_level": ["Wilson 95% intervals", "exact McNemar with discordants", "paired requested-depth mean/median bootstrap CI", "exact sign test with sign counts", "paired endpoint redirection bootstrap CI"],
            "interaction": ["seed-matched difference-in-differences", "exact within-seed layout-label permutation"],
            "failure_taxonomy": ["correct", "pick_failed", "transport_failed", "wrong_side", "release_failed"],
            "missingness": "Infrastructure-invalid attempts are separate; behavioral failures remain in denominators; missing values are null/NR, never zero.",
        },
        "required_raw_fields": [
            "success", "failure_category", "signed_final_lateral_offset", "requested_side_depth", "cone_entry_step", "cone_entry_sustained", "endpoint_shift", "action_distinct", "episode_length", "time_to_first_contact", "grasp_step", "cumulative_lateral_path", "peak_lateral_excursion", "symmetry_level_s", "asymmetry_metric_A", "position_residual", "orientation_residual", "midline_residual", "occlusion_check", "realised_object_poses", "arm_reset_pose", "scene_id"
        ],
        "source_bindings": SOURCE_BINDINGS,
        "queue": {"path": "artifacts/vla_wam_shared_v3/phase_e/cross_arena_geometry_v3e005/queue.jsonl", "sha256": queue_sha256, "rows": 108},
    }


def main() -> None:
    for relative, expected in SOURCE_BINDINGS.items():
        observed = sha256_file(ROOT / relative)
        if observed != expected:
            raise SystemExit(f"source hash mismatch: {relative}: {observed} != {expected}")
    prompts = historical_prompts()
    rows = build_queue(prompts)
    payload = jsonl(rows)
    if len(rows) != 108 or len({row["cell_id"] for row in rows}) != 108:
        raise SystemExit("queue is not exactly 108 unique cells")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "queue.jsonl").write_bytes(payload)
    registration = build_registration(sha256_bytes(payload), prompts)
    (OUT / "registration.json").write_bytes(canonical_json(registration))
    print(f"registration_sha256={sha256_file(OUT / 'registration.json')}")
    print(f"queue_sha256={sha256_file(OUT / 'queue.jsonl')}")
    print("cells=108")


if __name__ == "__main__":
    main()
