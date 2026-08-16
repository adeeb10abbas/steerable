#!/usr/bin/env python3
"""Freeze the post-result, zero-model V3-E007 reachability computation."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/zero_model_reachability_v3e007"
OUTPUT = BASE / "registration.json"
E004_LAYOUT = ROOT / "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004/layout/candidate.json"
B005_LAYOUT = ROOT / "artifacts/vla_wam_shared_v3/phase_b/nano_lateral_sweep_v3b005/model_blind_lateral_calibration_report.json"


SOURCES = (
    "experiments/v3/phase_e/model_blind_ik_gate.py",
    "artifacts/vla_wam_shared_v3/phase_e/reference_controller_symmetry_v3e002/model_blind_gate/ik_gate_control_final.json",
    "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r005/gates/candidate_schedule.json",
    "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004/layout/candidate.json",
    "artifacts/vla_wam_shared_v3/phase_b/nano_lateral_sweep_v3b005/model_blind_lateral_calibration_report.json",
    "artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/results/pi05_v3b002_episodes.jsonl",
    "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/results/nano_v3b001_episodes.jsonl",
    "artifacts/vla_wam_shared_v3/phase_b/dreamzero_mirror_v3b003/results/dreamzero_v3b003_episodes.jsonl",
    "artifacts/vla_wam_shared_v3/phase_b/nano_lateral_sweep_v3b005/results/nano_v3b005_dose_response_report.json",
    "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004/results/episodes.jsonl",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(relative: str) -> dict[str, object]:
    path = ROOT / relative
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}


def layout(layout_id: str, family: str, level: object, pose: dict[str, object]) -> dict[str, object]:
    return {
        "layout_id": layout_id,
        "family": family,
        "level": level,
        "reference_object": "bowl",
        "reference_position_world_m": [pose["x_m"], pose["y_m"], pose["z_m"]],
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite frozen registration: {OUTPUT}")
    e004 = json.loads(E004_LAYOUT.read_text(encoding="utf-8"))
    b005 = json.loads(B005_LAYOUT.read_text(encoding="utf-8"))
    control_pose = e004["levels"]["0.00"]["realised_object_poses"]["bowl"]
    layouts = [
        layout("reflection_control", "reflection", "control", control_pose),
        layout(
            "reflection_mirrored",
            "reflection",
            "position_mirrored",
            {**control_pose, "y_m": -float(control_pose["y_m"])},
        ),
    ]
    for index, y_m in enumerate(b005["selection"]["ordered_seven_levels_y_m"]):
        layouts.append(
            layout(
                f"nano_sweep_level_{index}",
                "nano_lateral_sweep",
                index,
                {**control_pose, "y_m": y_m},
            )
        )
    for key, row in e004["levels"].items():
        layouts.append(
            layout(
                f"symmetry_s_{key.replace('.', '_')}",
                "symmetric_scene",
                float(key),
                row["realised_object_poses"]["bowl"],
            )
        )

    value = {
        "schema_version": "vla-wam-shared-v3e007-zero-model-reachability-registration-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-E007",
        "status": "frozen_before_reachability_computation",
        "registered_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "analysis_character": "disclosed_post_result_mechanism_analysis",
        "question": "Does the side with more strict IK-feasible placement-target volume match the side favored by the policy in each registered layout?",
        "layouts": layouts,
        "workspace": {
            "coordinate_frame": "robot-root world axes; +Y is robot LEFT",
            "target_body": "Robotiq_2F_85/base_link",
            "target_orientation": "exact FK orientation of the frozen RoboLab reset joint vector; identical for every voxel and layout",
            "reset_arm_joint_position_rad": [
                0.0,
                -0.2 * 3.141592653589793,
                0.0,
                -0.8 * 3.141592653589793,
                0.0,
                0.6 * 3.141592653589793,
                0.0,
            ],
            "reference_relative_domain_m": {
                "forward_x_edges": [-0.1, -0.075, -0.05, -0.025, 0.0, 0.025, 0.05, 0.075, 0.1],
                "absolute_lateral_y_edges": [0.075, 0.1, 0.125, 0.15, 0.175, 0.2],
                "vertical_z_edges": [0.09, 0.105, 0.12, 0.135, 0.15],
                "voxel_centers": "midpoints of consecutive edges",
                "relation_filter": "abs(forward_x_center) <= absolute_lateral_y_center (closed 45-degree cone)",
                "left_rule": "+absolute_lateral_y_center",
                "right_rule": "-absolute_lateral_y_center",
                "basis": "E002 depths 0.075-0.200 m; E002 height 0.120 m plus/minus approximately one measured cube half-extent; symmetric forward span",
            },
        },
        "ik": {
            "robot_asset": {
                "path": "/data/users/ali/vla_wam/external/RoboLab-11142d4/assets/robots/franka_robotiq_2f_85_flattened.usd",
                "bytes": 14156362,
                "sha256": "f555695465687548a1bd31b5e3f30385182d476a67c17080b7820ad0ef747e41",
            },
            "robolab_robot_source": {
                "path": "/data/users/ali/vla_wam/external/RoboLab-11142d4/robolab/robots/droid.py",
                "sha256": "3c43b562cc22476135b7cd82c9c4c01ed361350eda25f3ef6669408ffce53e5d",
            },
            "chain": "seven revolute panda joints plus panda_joint8 and panda_hand_joint fixed frames read directly from USD",
            "joint_limits": "exact USD revolute limits",
            "solver": "scipy.optimize.least_squares bounded TRF",
            "deterministic_starts": 17,
            "start_rule": "frozen reset q plus 16 unscrambled Halton points mapped to the interior 90 percent of the exact joint-limit box",
            "max_function_evaluations_per_start": 200,
            "position_error_m_inclusive": 0.001,
            "orientation_error_deg_inclusive": 1.0,
            "known_pose_fk_validation": "all four R005 known-reachable diagnostic source q/base-link poses must reproduce within 1e-5 m and 1e-3 degree before voxel solving",
            "collision_scope": "joint-limit pose IK only; no collision, contact, dynamics, or policy inference",
        },
        "outcome_blind_analysis_rules": {
            "volume_per_voxel": "product of the three registered edge spacings",
            "primary_layout_statistic": "right_feasible_volume_m3 - left_feasible_volume_m3",
            "policy_favorite_primary": "mean requested-side depth RIGHT minus LEFT within model and layout",
            "policy_favorite_secondary": "binary success rate RIGHT minus LEFT within model and layout",
            "reflection_test": "sign concordance between feasible-volume advantage and requested-depth advantage for every model-layout row; ties reported",
            "graded_sweep_test": "Spearman correlation over the seven registered levels, with exact two-sided permutation p over 7! level permutations",
            "symmetric_scene_test": "report whether absolute feasible-volume asymmetry and absolute policy depth disparity both decrease from s=0 to s=1; residuals retained",
            "claim_rule": "name kinematic reachability as a supported mechanism only if signs align in both reflection layouts and the graded volume contrast covaries positively with Nano depth contrast; otherwise scope the paper to scene configuration without a kinematic-cause claim",
        },
        "source_bindings": [binding(relative) for relative in SOURCES],
        "learned_model_request_count": 0,
        "behavioral_episode_count": 0,
        "gpu_required": False,
        "prohibitions": [
            "no policy server, checkpoint load, model request, or behavioral episode",
            "no adjustment of grid, IK tolerance, start set, or claim rule after reading computed alignment",
            "no pooling DROID and RoboTwin outcomes",
            "no interpretation of IK feasibility as collision-free execution or task success",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(OUTPUT), "sha256": sha256(OUTPUT), "layouts": len(layouts)}, indent=2))


if __name__ == "__main__":
    main()

