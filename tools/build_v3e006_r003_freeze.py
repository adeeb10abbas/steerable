#!/usr/bin/env python3
"""Freeze the prospective V3-E006-R003 historical-seeded waypoint repair."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r003"
R002 = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r002"
ORIGINAL = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006"
R001 = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r001"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind(path: Path) -> dict[str, Any]:
    path = path.resolve()
    require(path.is_file(), f"missing frozen input: {path}")
    try:
        name = str(path.relative_to(ROOT))
    except ValueError:
        name = str(path)
    return {"path": name, "bytes": path.stat().st_size, "sha256": sha(path)}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"not an object: {path}")
    return value


def write_new(path: Path, value: Any) -> None:
    require(not path.exists(), f"refusing to overwrite prospective freeze: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize(q: Sequence[float]) -> list[float]:
    value = [float(item) for item in q]
    norm = math.sqrt(sum(item * item for item in value))
    require(len(value) == 4 and math.isfinite(norm) and norm > 0, "invalid quaternion")
    value = [item / norm for item in value]
    for item in value:
        if abs(float(item)) > 1e-15:
            return [-child for child in value] if item < 0 else value
    raise RuntimeError("zero quaternion")


def slerp(left: Sequence[float], right: Sequence[float], fraction: float) -> list[float]:
    a, b = normalize(left), normalize(right)
    dot = sum(left_item * right_item for left_item, right_item in zip(a, b))
    if dot < 0:
        b, dot = [-item for item in b], -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 1 - 1e-12:
        out = normalize([(1 - fraction) * x + fraction * y for x, y in zip(a, b)])
    else:
        theta = math.acos(dot)
        left_scale = math.sin((1 - fraction) * theta) / math.sin(theta)
        right_scale = math.sin(fraction * theta) / math.sin(theta)
        out = normalize([left_scale * x + right_scale * y for x, y in zip(a, b)])
    return [float(item) for item in out]


def waypoints(source: dict[str, Any], target: dict[str, Any]) -> list[dict[str, Any]]:
    p0 = [float(item) for item in source["eef_position_world_m"]]
    p1 = [float(item) for item in target["position_world_m"]]
    rows = []
    for index in range(1, 9):
        fraction = index / 8.0
        rows.append(
            {
                "waypoint_index_one_based": index,
                "fraction": fraction,
                "position_world_m": [(1 - fraction) * x + fraction * y for x, y in zip(p0, p1)],
                "quaternion_world_wxyz": slerp(
                    source["eef_quaternion_world_wxyz"], target["quaternion_world_wxyz"], fraction
                ),
                "hold_steps": 30,
                "required_final_consecutive_steps": 10,
                "position_error_m_inclusive": 0.001,
                "orientation_geodesic_error_deg_inclusive": 1.0,
            }
        )
    require(rows[-1]["position_world_m"] == target["position_world_m"], "waypoint endpoint changed")
    return rows


def qmul(left: Sequence[float], right: Sequence[float]) -> list[float]:
    lw, lx, ly, lz = normalize(left)
    rw, rx, ry, rz = normalize(right)
    return normalize(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def qrotate(quaternion: Sequence[float], vector: Sequence[float]) -> list[float]:
    w, x, y, z = normalize(quaternion)
    vx, vy, vz = (float(item) for item in vector)
    # q * (0,v) * inv(q), expanded to avoid an external math dependency.
    tx, ty, tz = 2.0 * (y * vz - z * vy), 2.0 * (z * vx - x * vz), 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


EEF_OFFSET_POS = [0.0, 0.0, 0.0]
EEF_OFFSET_ROT = [0.5, -0.5, 0.5, -0.5]


def add_eef_diagnostic_from_base(source: dict[str, Any]) -> None:
    source["base_link_position_world_m"] = source.pop("eef_position_world_m")
    source["base_link_quaternion_world_wxyz"] = source.pop(
        "eef_quaternion_world_wxyz"
    )
    source["cube_in_base_link_translation_m"] = source.pop(
        "cube_in_eef_translation_m"
    )
    source["cube_in_base_link_quaternion_wxyz"] = source.pop(
        "cube_in_eef_quaternion_wxyz"
    )
    base_position = source["base_link_position_world_m"]
    base_quaternion = source["base_link_quaternion_world_wxyz"]
    rotated_offset = qrotate(base_quaternion, EEF_OFFSET_POS)
    source["expected_eef_frame_position_world_m"] = [
        float(a) + float(b) for a, b in zip(base_position, rotated_offset)
    ]
    source["expected_eef_frame_quaternion_world_wxyz"] = qmul(
        base_quaternion, EEF_OFFSET_ROT
    )


def relabel_r002_base_frame(stage: dict[str, Any]) -> None:
    """Correct R002's mislabeled HDF5 ee_pose fields without changing numbers."""

    stage["selected_observed_cube_in_base_link_transform"] = stage.pop(
        "selected_observed_cube_in_eef_transform"
    )
    stage["unselected_observed_cube_in_base_link_transform"] = stage.pop(
        "unselected_observed_cube_in_eef_transform"
    )
    stage["centerline_constrained_base_link_ik_target"] = stage.pop(
        "centerline_constrained_eef_ik_target"
    )
    stage.pop("open_approach_targets", None)
    for source in stage["both_direction_sources"].values():
        source["base_link_position_world_m"] = source.pop("eef_position_world_m")
        source["base_link_quaternion_world_wxyz"] = source.pop(
            "eef_quaternion_world_wxyz"
        )
        source["cube_in_base_link_translation_m"] = source.pop(
            "cube_in_eef_translation_m"
        )
        source["cube_in_base_link_quaternion_wxyz"] = source.pop(
            "cube_in_eef_quaternion_wxyz"
        )
    reflection = stage.get("reflection_definition", {})
    if "right_world_eef_reflected" in reflection:
        reflection["right_world_base_link_reflected"] = reflection.pop(
            "right_world_eef_reflected"
        )


def main() -> None:
    require(not OUT.exists(), f"R003 freeze already exists: {OUT}")
    r002_schedule_path = R002 / "gates/candidate_schedule.json"
    r002_results_path = R002 / "results/results.json"
    r002_manifest_path = R002 / "results/evidence_manifest.json"
    r002_memo_path = R002 / "results/DECISION_MEMO.md"
    r002_schedule = load(r002_schedule_path)
    r002_results = load(r002_results_path)
    require(r002_results.get("status") == "r002_candidate_budget_exhausted_no_valid_state_pair", "R002 is not immutable exhaustion")
    require(r002_results.get("model_request_count") == r002_results.get("behavioral_episode_count") == 0, "R002 was not zero-request")

    state_contract = ROOT / "experiments/v3/phase_e/canonical_stage_localization_v3e006/state_contract.py"
    ood_source = ROOT / "experiments/v3/phase_e/canonical_stage_localization_v3e006/ood_reference.py"
    ood_freeze = ORIGINAL / "gates/ood_reference_freeze.json"
    reset = ORIGINAL / "gates/e004_full_reset_reference.json"
    runtime = ORIGINAL / "gates/exact_pi05_runtime_contract.json"
    layout = ROOT / "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004/layout/candidate.json"
    original_closure = R001 / "gates/original_v3e006_closure_binding.json"
    diagnostic_sources = {
        "robolab_droid_action_config": {
            "path": "/data/users/ali/vla_wam/external/RoboLab-11142d4/robolab/robots/droid.py",
            "bytes": 16098,
            "sha256": "3c43b562cc22476135b7cd82c9c4c01ed361350eda25f3ef6669408ffce53e5d",
        },
        "robolab_abs_ik_demo": {
            "path": "/data/users/ali/vla_wam/external/RoboLab-11142d4/examples/run_abs_ik_demo.py",
            "bytes": 10177,
            "sha256": "c18ab8b4174d342e33ff4f8d2d0a9a913dd5af7daeb2ca2e731b44b82f3d9778",
        },
        "isaaclab_differential_ik": {
            "path": "/data/users/ali/vla_wam/envs/robolab-v2-isaac50/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/controllers/differential_ik.py",
            "bytes": 10808,
            "sha256": "307dde7c76b4b9d0834361cf8d53f8653b538c15041da1df725c3bd6d32666c4",
        },
        "isaaclab_differential_ik_config": {
            "path": "/data/users/ali/vla_wam/envs/robolab-v2-isaac50/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/controllers/differential_ik_cfg.py",
            "bytes": 2955,
            "sha256": "3e8fc09c6d428c4880f9badbc7ac8f82fe19362ef6c472e74d8db1c35dca0f08",
        },
        "robolab_post_step_end_effector_pose_recorder": {
            "path": "/data/users/ali/vla_wam/external/RoboLab-11142d4/robolab/core/events/basic_recorders.py",
            "bytes": 15427,
            "sha256": "84944e9eedb664ba8f34eee40e2b0efe064d5460a942eecb68ad7d491a5ffacd",
            "semantic_assertion": (
                "PostStepEndEffectorPoseRecorderCfg.ee_body_name defaults to base_link and "
                "the ee_pose HDF5 group records robot.data.body_pos_w/body_quat_w for that body"
            ),
        },
    }
    candidate_search = {
        "algorithm_version": "historical-q-seeded-abs-ik-waypoint-repair-v1",
        "maximum_complete_candidate_pairs": 4,
        "rank_order": [
            {"rank": row["candidate_rank"], "grasp_contact": row["canonical_grasp"]["contact_transform_selector"], "carry_contact": row["canonical_carry"]["contact_transform_selector"]}
            for row in r002_schedule["candidate_pairs"][:4]
        ],
        "acceptance": "The lowest numeric complete pair whose grasp and carry both pass every unchanged gate is accepted.",
        "first_pass_rule": "Evaluate complete ranks 1 through 4 serially; stop only after both stages of the first passing rank.",
        "exhaustion": "Four valid rejected pairs close R003 at zero model requests and zero behavioral episodes. Infrastructure-invalid attempts are retried identically and are not rank rejection. No seed, waypoint, target, contact transform, solver parameter, or gate may be changed after the diagnostic or candidate run begins.",
    }
    registration = {
        "schema_version": "vla-wam-shared-v3e006-r003-prospective-construction-repair-registration-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "repair_amendment_id": "V3-E006-R003",
        "predecessor_repair_amendment_id": "V3-E006-R002",
        "original_amendment_id": "V3-E006",
        "registered_at_utc": "2026-08-12T23:28:16Z",
        "status": "prospectively_registered_before_any_r003_live_diagnostic_candidate_or_model_request",
        "user_authorized_override": "The user explicitly directed the team to solve construction gates and run E006. R003 is a new prospective solver-only repair after the byte-preserved R002 finite search exhausted.",
        "counts_at_registration": {"r003_live_diagnostics": 0, "r003_live_candidate_evaluations": 0, "model_requests": 0, "behavioral_episodes": 0},
        "predecessor_outcomes": {
            "original_v3e006": "failed canonical construction; no model request",
            "r001": "eight candidate pairs exhausted; no model request",
            "r002": "eight candidate pairs/16 stages all failed the frozen IK convergence boundary before scientific gate evaluation; no model request",
            "claim_boundary": "No predecessor canonical-state gate passed; every predecessor artifact remains immutable.",
        },
        "engineering_rationale_only": "R002 direct reset-to-final commands reached neither the position nor orientation target and often saturated soft limits. Source audit then established that RoboLab's HDF5 ee_pose field is base_link, while R002 labeled and commanded it as the offset eef_frame. R003 corrects only this frame semantics plus historical-q waypoint initialization; it does not motivate any world cube target or scientific-threshold change.",
        "frozen_inputs": {
            "r002_results": bind(r002_results_path), "r002_evidence_manifest": bind(r002_manifest_path),
            "r002_decision_memo": bind(r002_memo_path), "r002_candidate_schedule": bind(r002_schedule_path),
            "r002_registration": bind(R002 / "repair_registration.json"),
            "original_v3e006_closure_binding": bind(original_closure),
            "state_contract": bind(state_contract), "ood_reference": bind(ood_source),
            "ood_freeze": bind(ood_freeze), "e004_full_reset_reference": bind(reset),
            "e004_candidate": bind(layout), "runtime_contract": bind(runtime),
        },
        "known_reachable_diagnostic": {
            "required_before_candidate_rank_one": True,
            "anchors": "canonical_grasp and canonical_carry, LEFT and RIGHT: exact hash-bound historical 13-joint q, cube pose, and recorded base_link pose from the HDF5 ee_pose group",
            "state_write": "fresh exact E004 environment; atomically write exact historical robot q and cube pose with zero velocities, set joint targets, forward without integration; normal gripper command remains closed",
            "controller_hold": "command the recorded base_link pose directly as the DroidIKActionCfg absolute quaternion for 30 physics steps; separately assert live eef_frame pose equals base_link pose composed with EEF_OFFSET_POS/EEF_OFFSET_ROT",
            "acceptance": "all four anchors must be finite, inside live soft joint limits, nonterminal, and inside <=1mm and <=1deg for each of the final 10 consecutive steps",
            "failure": "any diagnostic failure closes R003 diagnostic as failed and prohibits candidate evaluation; no adaptive edit/retry under this registration",
        },
        "solver_contract": {
            "controller": "RoboLab DroidIKActionCfg absolute pose, scale 1.0, body base_link; command quaternion is direct WXYZ with no EEF offset conversion",
            "method": "damped least squares",
            "dls_lambda": 0.01,
            "actuator_gains": {"panda_joints_1_4": {"stiffness": 400.0, "damping": 80.0}, "panda_joints_5_7": {"stiffness": 400.0, "damping": 80.0}},
            "controller_source_bindings": diagnostic_sources,
            "initialization": "selected exact historical 13-joint q and cube pose, all velocities zero, closed normal gripper; recorded HDF5 ee_pose is explicitly base_link; no policy action/prompt history",
            "waypoints": "8 exact base_link-pose fractions 1/8..8/8; world position linear interpolation and WXYZ shortest-arc quaternion SLERP, flipping target sign iff dot<0; each held 30 steps",
            "waypoint_acceptance": "each waypoint requires its final 10 consecutive steps at <=1mm position and <=1deg geodesic orientation error; any termination, nonfinite state, or soft-limit violation rejects that stage",
            "fresh_lifecycle": "one diagnostic environment per anchor and one solve plus separate materialization environment per candidate rank/stage; every environment starts from verified E004 reset and closes before the next",
        },
        "target_and_contact_contract": "R002 direct ranks1-4 world target cube poses and rank order are unchanged. Numerically unchanged HDF5 ee_pose-derived contacts/targets are prospectively relabeled as cube-in-base_link/base_link target transforms; direct normal-contact materialization is unchanged.",
        "unchanged_scientific_gate": {
            "source": bind(state_contract), "ood_source": bind(ood_source), "ood_freeze": bind(ood_freeze),
            "window_steps": 10,
            "thresholds": {"cube_abs_y_m": "<0.001", "relative_drift_m": "<0.002", "arm_speed_rad_s": "<0.01", "cube_linear_speed_m_s": "<0.01", "cube_angular_speed_rad_s": "<0.05", "intended_contact_n": ">1.0", "unintended_contact_n": "<=1.0"},
            "other": "exact unchanged E004 reset comparison, contact inventory, camera/visibility gate, companion pose gate, and stage OOD thresholds. Live base_link pose is supplied to the historically named state.eef OOD feature because the frozen reference's HDF5 ee_pose was base_link; both frames and the exact offset residual are retained.",
        },
        "candidate_search": candidate_search,
        "downstream_behavioral_design_if_and_only_if_state_pair_passes": "unchanged original 341 seeds x 3 stages x 2 directions = 2046; exact E004 pi0.5 runtime/prompts/scorer; no behavior authorized here",
        "release_boundary": "One zero-model diagnostic plus conditional finite R003 search only after final pushed source gate and independent audit. No model request, smoke, isolation, registration activation, or inference is authorized.",
    }
    registration_path = OUT / "repair_registration.json"
    write_new(registration_path, registration)

    diagnostics: list[dict[str, Any]] = []
    rank_one = r002_schedule["candidate_pairs"][0]
    for stage in ("canonical_grasp", "canonical_carry"):
        for side in ("left", "right"):
            source = json.loads(json.dumps(rank_one[stage]["both_direction_sources"][side]))
            add_eef_diagnostic_from_base(source)
            diagnostics.append(
                {
                    "diagnostic_index_one_based": len(diagnostics) + 1,
                    "stage": stage, "source_side": side,
                    "source_environment_seed": rank_one[stage]["source_environment_seed"],
                    "source": source,
                    "hold_steps": 30, "required_final_consecutive_steps": 10,
                    "position_error_m_inclusive": 0.001,
                    "orientation_geodesic_error_deg_inclusive": 1.0,
                    "recorded_pose_frame": "robot/base_link",
                    "command_frame_conversion": "target_base_link quaternion is sent directly as the DroidIKActionCfg absolute quaternion, WXYZ",
                }
            )
    candidates = []
    for old in r002_schedule["candidate_pairs"][:4]:
        row = json.loads(json.dumps(old))
        require(row["construction_method"] == "direct_contact_initialization", "R003 target overlay is not R002 direct family")
        for stage in ("canonical_grasp", "canonical_carry"):
            entry = row[stage]
            relabel_r002_base_frame(entry)
            side = entry["selected_historical_source_side"]
            source = entry["both_direction_sources"][side]
            entry["r003_solver_initialization"] = {
                "source_side": side,
                "exact_historical_joint_position_rad": source["joint_position_rad"],
                "exact_historical_cube_pose_world_wxyz": source["cube_pose_world_wxyz"],
                "exact_historical_base_link_pose_world_wxyz": source["base_link_position_world_m"] + source["base_link_quaternion_world_wxyz"],
                "all_robot_and_cube_velocities": "exact_zero",
                "gripper_command": 1.0,
                "waypoints": waypoints(
                    {
                        "eef_position_world_m": source["base_link_position_world_m"],
                        "eef_quaternion_world_wxyz": source["base_link_quaternion_world_wxyz"],
                    },
                    entry["centerline_constrained_base_link_ik_target"],
                ),
            }
        candidates.append(row)
    schedule = {
        "schema_version": "vla-wam-shared-v3e006-r003-candidate-schedule-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "repair_amendment_id": "V3-E006-R003",
        "status": "frozen_before_any_r003_live_diagnostic_candidate_or_model_request",
        "candidate_budget": 4,
        "diagnostic_budget": 4,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "r003_live_diagnostic_count": 0,
        "r003_live_candidate_evaluation_count": 0,
        "repair_registration": bind(registration_path),
        "r002_predecessor": {"results": bind(r002_results_path), "evidence_manifest": bind(r002_manifest_path), "decision_memo": bind(r002_memo_path), "candidate_schedule": bind(r002_schedule_path)},
        "original_v3e006_closure_binding": bind(original_closure),
        "unchanged_gate_bindings": {"state_contract": bind(state_contract), "ood_reference": bind(ood_source), "ood_freeze": bind(ood_freeze), "e004_full_reset_reference": bind(reset), "e004_candidate": bind(layout), "runtime_contract": bind(runtime)},
        "controller_source_bindings": diagnostic_sources,
        "historical_policy_provenance_disclosure": (
            "The frozen source states were produced by historical E004 pi0.5 behavior. "
            "R003 performs no new model request and selects only through prospectively "
            "frozen reachability and unchanged physical/OOD/camera/companion gates."
        ),
        "selection_rule": candidate_search,
        "known_reachable_diagnostics": diagnostics,
        "candidate_pairs": candidates,
        "frame_correction": {
            "historical_hdf5_ee_pose_actual_frame": "robot/base_link",
            "r002_defect": "R002 mislabeled the recorder value as eef_frame and applied inv(EEF_OFFSET_ROT) again when commanding it",
            "r003_action_contract": "base_link target quaternion is sent directly; live eef_frame must equal base_link composed with EEF_OFFSET_POS/EEF_OFFSET_ROT",
            "ood_compatibility": "unchanged frozen OOD reference also came from HDF5 base_link; R003 captures live base_link into its historically named eef feature and retains both frames",
        },
        "r002_closure_commit": "27d1bfd844808f7f336bbb4e25552a9c859fd08a",
        "r002_raw_result": {
            "path": "/data/users/ali/vla_wam/raw/v3e006_r002/state_repair/39b18d9-a40r06-attempt01/raw/state_repair_result.json",
            "bytes": 7103108,
            "sha256": "afb8c3ba2b53f1513bd22fd6135b16cbfe3e4dd9de3fe3d818ac05e458311fe7",
        },
        "target_contact_and_rank_identity": "Each candidate preserves the R002 rank and world target cube pose; its mislabeled ee_pose-derived transforms are explicitly corrected to base_link semantics before the frozen R003 solver runs.",
    }
    core = json.dumps(schedule, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    schedule["schedule_canonical_sha256_without_this_field"] = hashlib.sha256(core).hexdigest()
    write_new(OUT / "gates/candidate_schedule.json", schedule)
    print(json.dumps({"registration": bind(registration_path), "schedule": bind(OUT / "gates/candidate_schedule.json")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
