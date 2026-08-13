#!/usr/bin/env python3
"""Hash-close R011's final fail-closed, pre-action scene-sync outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from tools.validate_v3e006_r011 import validate_geometry_attachment_preflight


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r011"
DEFAULT_OUTPUT = ARTIFACT / "results"
TERMINAL = "r011_geometry_attachment_preflight_failed_candidates_not_evaluated"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"not a JSON object: {path}")
    return value


def binding(path: Path, *, repo_relative: bool = False) -> dict[str, Any]:
    path = path.resolve()
    require(path.is_file(), f"missing bound file: {path}")
    return {
        "path": str(path.relative_to(ROOT)) if repo_relative else str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def summarize_preflight(
    value: Mapping[str, Any], schedule: Mapping[str, Any]
) -> dict[str, Any]:
    require(
        validate_geometry_attachment_preflight(value, schedule) is False,
        "R011 preflight unexpectedly passed",
    )
    require(
        value.get("physics_to_usd_setting_path") == "/physics/updateToUsd"
        and value.get("physics_to_usd_setting_before_one_shot_sync")
        == value.get("physics_to_usd_setting_after_one_shot_sync")
        and value.get("physics_to_usd_setting_unchanged") is True
        and value.get("physics_or_action_steps_between_tensor_snapshot_and_oracle")
        == 0,
        "R011 one-shot synchronization evidence differs",
    )
    stage = value["stage_and_scene_identity"]
    call = value["physics_to_usd_synchronization_call_arguments"]
    require(
        stage.get("configured_physics_scene_path") == "/physicsScene"
        and stage.get("resolved_physics_scene_paths") == ["/physicsScene"]
        and stage.get("unique_configured_physics_scene") is True
        and call
        == {
            "physics_scene_path": "/physicsScene",
            "physics_scene_path_integer": stage.get("physics_scene_path_integer"),
            "update_to_usd": True,
            "update_velocities_to_usd": False,
        },
        "R011 stage-specific call binding differs",
    )
    owners = value["owner_pose_oracle_rows"]
    shapes = value["oracle_rows"]
    require(
        [row.get("role") for row in owners] == ["left", "right", "cube"]
        and len(shapes) == value.get("collision_prim_count") == 5
        and all(row.get("passed") is False for row in owners)
        and all(row.get("passed") is False for row in shapes),
        "R011 oracle failure pattern differs",
    )
    owner_position = [float(row["maximum_absolute_position_error_m"]) for row in owners]
    owner_orientation = [float(row["maximum_axis_orientation_error_deg"]) for row in owners]
    shape_errors = [
        max(
            float(row["maximum_absolute_minimum_error_m"]),
            float(row["maximum_absolute_maximum_error_m"]),
        )
        for row in shapes
    ]
    return {
        "classification": "final_scene_specific_usd_tensor_oracle_synchronization_failure",
        "mechanically_valid_fail_closed_execution": True,
        "relative_bound_attachment_validated": False,
        "relative_bound_controller_evaluated": False,
        "intended_r011_construction_scientifically_exhausted": False,
        "behavioral_release_permitted": False,
        "root_cause_scope": (
            "the exact stage-identity-checked scene-specific PhysX-to-USD writeback "
            "still did not make the fresh-reset USD owner transforms and collision "
            "bounds agree with their simultaneous live tensor poses; the final "
            "infrastructure-only correction therefore blocked every action"
        ),
        "scene_specific_synchronization": {
            "call": value.get("physics_to_usd_synchronization_call"),
            "call_arguments": call,
            "call_count": value.get("physics_to_usd_synchronization_call_count"),
            "stage_and_scene_identity": stage,
            "setting_path": value.get("physics_to_usd_setting_path"),
            "setting_before": value.get(
                "physics_to_usd_setting_before_one_shot_sync"
            ),
            "setting_after": value.get(
                "physics_to_usd_setting_after_one_shot_sync"
            ),
            "setting_unchanged": value.get("physics_to_usd_setting_unchanged"),
            "reset_steps": value.get("physics_to_usd_synchronized_reset_steps"),
            "intervening_action_or_physics_steps": value.get(
                "physics_or_action_steps_between_tensor_snapshot_and_oracle"
            ),
        },
        "geometry_identity_sha256": value.get("geometry_identity_sha256"),
        "owner_pose_rows": [
            {
                key: row[key]
                for key in (
                    "role",
                    "owning_rigid_body_prim_path",
                    "maximum_absolute_position_error_m",
                    "maximum_axis_orientation_error_deg",
                    "passed",
                )
            }
            for row in owners
        ],
        "collision_oracle_rows": [
            {
                key: row[key]
                for key in (
                    "role",
                    "collision_prim_path",
                    "maximum_absolute_minimum_error_m",
                    "maximum_absolute_maximum_error_m",
                    "passed",
                )
            }
            for row in shapes
        ],
        "quantitative_signature": {
            "owner_position_error_m_min": min(owner_position),
            "owner_position_error_m_max": max(owner_position),
            "owner_orientation_error_deg_min": min(owner_orientation),
            "owner_orientation_error_deg_max": max(owner_orientation),
            "collision_aabb_error_m_min": min(shape_errors),
            "collision_aabb_error_m_max": max(shape_errors),
            "failed_owner_count": 3,
            "failed_collision_prim_count": 5,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--authoritative-validation-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    raw_root = args.candidate_root.resolve()
    output = args.output_root.resolve()
    require(not output.exists(), f"refusing to overwrite R011 closure: {output}")

    launch_path = raw_root / "launch.json"
    harness_path = raw_root / "harness_result.json"
    runtime_path = raw_root / "runtime.log"
    preflight_path = raw_root / "raw/geometry_attachment_preflight.json"
    raw_path = raw_root / "raw/state_repair_result.json"
    harness, raw, preflight = map(load, (harness_path, raw_path, preflight_path))
    receipt = load(args.authoritative_validation_receipt)
    schedule_path = Path(raw["candidate_schedule"]["path"])
    schedule = load(schedule_path)
    require(
        harness.get("process_completed") is True
        and harness.get("process_exit_code") == 0
        and harness.get("child_status") == TERMINAL
        and harness.get("scientific_gate_passed") is False,
        "R011 harness terminal differs",
    )
    require(
        raw.get("status") == TERMINAL
        and raw.get("passed") is False
        and raw.get("geometry_attachment_preflight_count") == 1
        and raw.get("r011_live_diagnostic_count") == 0
        and raw.get("repair_candidate_evaluation_count") == 0
        and raw.get("state_candidate_count") == 0
        and raw.get("accepted_candidate_rank") is None
        and raw.get("accepted_states") is None
        and raw.get("known_reachable_diagnostics") == []
        and raw.get("attempts") == []
        and raw.get("model_request_count") == raw.get("behavioral_episode_count") == 0,
        "R011 pre-action terminal/counts differ",
    )
    require(raw.get("geometry_attachment_preflight") == preflight, "preflight/result differ")
    finding = summarize_preflight(preflight, schedule)
    evidence = receipt.get("candidate_evidence")
    require(
        receipt.get("passed") is True
        and isinstance(evidence, Mapping)
        and evidence.get("passed") is True
        and evidence.get("geometry_attachment_preflight_passed") is False
        and evidence.get("child_report", {}).get("sha256") == sha256(raw_path)
        and evidence.get("geometry_attachment_preflight", {}).get("sha256")
        == sha256(preflight_path),
        "authoritative target receipt differs",
    )

    output.mkdir(parents=True)
    results = {
        "schema_version": "vla-wam-shared-v3e006-r011-state-repair-closure-v1",
        "amendment_id": "V3-E006-R011",
        "status": TERMINAL,
        "passed": False,
        "geometry_attachment_preflight_count": 1,
        "diagnostic_evaluation_count": 0,
        "candidate_pair_evaluation_count": 0,
        "accepted_candidate_rank": None,
        "accepted_state_hashes": None,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "behavioral_activation_released": False,
        "mechanically_valid_fail_closed_execution": True,
        "relative_bound_attachment_validated": False,
        "relative_bound_controller_evaluated": False,
        "intended_r011_construction_scientifically_exhausted": False,
        "final_state_construction_blocker": True,
        "geometry_oracle_finding": finding,
        "raw_result": binding(raw_path),
        "raw_preflight": binding(preflight_path),
        "raw_harness": binding(harness_path),
        "raw_launch": binding(launch_path),
        "raw_runtime_log": binding(runtime_path),
        "authoritative_target_validation_receipt": binding(
            args.authoritative_validation_receipt
        ),
        "registration": raw.get("repair_registration"),
        "candidate_schedule": raw.get("candidate_schedule"),
        "source_push_gate_at_execution": raw.get("source_push_gate"),
        "source_commit_at_execution": load(launch_path).get("study_commit"),
    }
    result_path = output / "results.json"
    result_path.write_bytes(canonical_bytes(results))
    receipt_path = output / "target_validation_receipt.json"
    receipt_path.write_bytes(args.authoritative_validation_receipt.read_bytes())
    signature = finding["quantitative_signature"]
    memo_path = output / "DECISION_MEMO.md"
    memo_path.write_text(
        "# V3-E006-R011 final state-construction decision\n\n"
        "R011 completed its one registered zero-model scene-specific attachment preflight "
        "and failed closed before every diagnostic, candidate, or construction action. No "
        "state was accepted; behavioral activation and inference remain blocked.\n\n"
        "The verified current/initial USD stage, unique `/physicsScene`, StageCache identity, "
        "structured path-integer call arguments, persistent update-to-USD setting, and exact "
        "single scene-specific writeback all passed their registered checks. Nevertheless, "
        "all three USD owner poses and all five collision AABBs disagreed with the simultaneous "
        "live tensors. Owner position errors ranged from "
        f"{signature['owner_position_error_m_min']:.15f} to "
        f"{signature['owner_position_error_m_max']:.15f} m; collision AABB errors ranged from "
        f"{signature['collision_aabb_error_m_min']:.15f} to "
        f"{signature['collision_aabb_error_m_max']:.15f} m. The cube owner discrepancy was "
        "about 126.9 mm. Thus the relative-bound attachment and controller were never evaluated. "
        "This is the final infrastructure blocker, not a scientific candidate exhaustion; no "
        "R012, B001 activation, model request, or behavioral episode is authorized.\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "vla-wam-shared-v3e006-r011-closure-manifest-v1",
        "status": TERMINAL,
        "repo_result": binding(result_path, repo_relative=True),
        "repo_target_validation_receipt": binding(receipt_path, repo_relative=True),
        "decision_memo": binding(memo_path, repo_relative=True),
        "closure_tool": binding(Path(__file__), repo_relative=True),
        "closure_validator": binding(
            ROOT / "tools/validate_v3e006_r011_closure.py", repo_relative=True
        ),
        "repair_registration": binding(
            ARTIFACT / "repair_registration.json", repo_relative=True
        ),
        "candidate_schedule": binding(
            ARTIFACT / "gates/candidate_schedule.json", repo_relative=True
        ),
        "source_push_gate": binding(
            ARTIFACT / "source_push_gate.json", repo_relative=True
        ),
        "raw_evidence": {
            "child_result": binding(raw_path),
            "geometry_attachment_preflight": binding(preflight_path),
            "harness": binding(harness_path),
            "launch": binding(launch_path),
            "runtime_log": binding(runtime_path),
            "authoritative_validation": binding(args.authoritative_validation_receipt),
        },
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "behavioral_release_permitted": False,
        "final_state_construction_blocker": True,
    }
    manifest_path = output / "evidence_manifest.json"
    manifest_path.write_bytes(canonical_bytes(manifest))
    print(
        json.dumps(
            {
                "results": binding(result_path),
                "manifest": binding(manifest_path),
                "memo": binding(memo_path),
                "receipt": binding(receipt_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
