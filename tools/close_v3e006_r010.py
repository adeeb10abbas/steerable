#!/usr/bin/env python3
"""Hash-close R010's fail-closed, pre-action geometry-oracle outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r010"
DEFAULT_OUTPUT = ARTIFACT / "results"
TERMINAL = "r010_geometry_attachment_preflight_failed_candidates_not_evaluated"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
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


def summarize_preflight(value: Mapping[str, Any]) -> dict[str, Any]:
    require(
        value.get("status") == "failed_r010_relative_bound_tensor_world_oracle_preflight"
        and value.get("passed") is False,
        "R010 preflight terminal differs",
    )
    require(
        value.get("physics_to_usd_setting_before_one_shot_sync")
        == value.get("physics_to_usd_setting_after_one_shot_sync")
        and value.get("physics_to_usd_setting_unchanged") is True
        and value.get("physics_or_action_steps_between_tensor_snapshot_and_oracle") == 0,
        "R010 one-shot synchronization evidence differs",
    )
    owners = value.get("owner_pose_oracle_rows")
    shapes = value.get("oracle_rows")
    require(
        isinstance(owners, list)
        and [row.get("role") for row in owners] == ["left", "right", "cube"]
        and isinstance(shapes, list)
        and len(shapes) == value.get("collision_prim_count") == 5,
        "R010 oracle inventory differs",
    )
    require(
        value.get("all_inventory_paths_evaluated_once") is True
        and all(row.get("passed") is False for row in owners)
        and all(row.get("passed") is False for row in shapes),
        "R010 oracle pass pattern differs",
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
        "classification": "pre_action_geometry_oracle_synchronization_failure",
        "mechanically_valid_fail_closed_execution": True,
        "relative_bound_attachment_validated": False,
        "relative_bound_controller_evaluated": False,
        "intended_r010_construction_scientifically_exhausted": False,
        "behavioral_release_permitted": False,
        "root_cause_scope": (
            "the registered global PhysX-to-USD writeback did not make the dedicated "
            "fresh-reset USD owner transforms agree with the simultaneously retained live "
            "tensor poses; therefore the independent attachment oracle correctly blocked all actions"
        ),
        "one_shot_synchronization": {
            "call": value.get("physics_to_usd_synchronization_call"),
            "setting_path": value.get("physics_to_usd_setting_path"),
            "setting_before": value.get("physics_to_usd_setting_before_one_shot_sync"),
            "setting_after": value.get("physics_to_usd_setting_after_one_shot_sync"),
            "setting_unchanged": value.get("physics_to_usd_setting_unchanged"),
            "reset_steps": value.get("physics_to_usd_synchronized_reset_steps"),
            "intervening_action_or_physics_steps": value.get(
                "physics_or_action_steps_between_tensor_snapshot_and_oracle"
            ),
        },
        "geometry_identity_sha256": value.get("geometry_identity_sha256"),
        "owner_pose_rows": [
            {
                "role": row["role"],
                "owning_rigid_body_prim_path": row["owning_rigid_body_prim_path"],
                "maximum_absolute_position_error_m": row["maximum_absolute_position_error_m"],
                "maximum_axis_orientation_error_deg": row[
                    "maximum_axis_orientation_error_deg"
                ],
                "passed": row["passed"],
            }
            for row in owners
        ],
        "collision_oracle_rows": [
            {
                "role": row["role"],
                "collision_prim_path": row["collision_prim_path"],
                "maximum_absolute_minimum_error_m": row[
                    "maximum_absolute_minimum_error_m"
                ],
                "maximum_absolute_maximum_error_m": row[
                    "maximum_absolute_maximum_error_m"
                ],
                "passed": row["passed"],
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
            "failed_owner_count": sum(row.get("passed") is False for row in owners),
            "failed_collision_prim_count": sum(row.get("passed") is False for row in shapes),
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
    require(not output.exists(), f"refusing to overwrite R010 closure: {output}")

    launch_path = raw_root / "launch.json"
    harness_path = raw_root / "harness_result.json"
    runtime_path = raw_root / "runtime.log"
    preflight_path = raw_root / "raw/geometry_attachment_preflight.json"
    raw_path = raw_root / "raw/state_repair_result.json"
    harness = load(harness_path)
    raw = load(raw_path)
    preflight = load(preflight_path)
    receipt = load(args.authoritative_validation_receipt)
    require(
        harness.get("process_completed") is True
        and harness.get("process_exit_code") == 0
        and harness.get("child_status") == TERMINAL
        and harness.get("scientific_gate_passed") is False,
        "R010 harness terminal differs",
    )
    require(
        raw.get("status") == TERMINAL
        and raw.get("passed") is False
        and raw.get("geometry_attachment_preflight_count") == 1
        and raw.get("r010_live_diagnostic_count") == 0
        and raw.get("repair_candidate_evaluation_count") == 0
        and raw.get("state_candidate_count") == 0
        and raw.get("accepted_candidate_rank") is None
        and raw.get("accepted_states") is None
        and raw.get("known_reachable_diagnostics") == []
        and raw.get("attempts") == []
        and raw.get("model_request_count") == raw.get("behavioral_episode_count") == 0,
        "R010 pre-action terminal/counts differ",
    )
    require(raw.get("geometry_attachment_preflight") == preflight, "preflight/result differ")
    finding = summarize_preflight(preflight)
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
        "schema_version": "vla-wam-shared-v3e006-r010-state-repair-closure-v1",
        "amendment_id": "V3-E006-R010",
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
        "intended_r010_construction_scientifically_exhausted": False,
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
        "# V3-E006-R010 state-construction decision\n\n"
        "R010 completed exactly one registered, zero-model attachment preflight and failed closed "
        "before every diagnostic and candidate action. No state was accepted, and behavioral "
        "activation remains blocked.\n\n"
        "The failure was in the validation oracle's dynamic synchronization, not a scientific "
        "candidate rejection. After the registered one-shot global PhysX-to-USD writeback, the "
        "persistent update-to-USD setting remained unchanged and no action or simulation step "
        "intervened, yet all three USD owner poses disagreed with their simultaneously retained "
        "live tensor poses. Owner position errors ranged from "
        f"{signature['owner_position_error_m_min']:.15f} to "
        f"{signature['owner_position_error_m_max']:.15f} m and orientation errors from "
        f"{signature['owner_orientation_error_deg_min']:.15f} to "
        f"{signature['owner_orientation_error_deg_max']:.15f} degrees. All five collision AABB "
        "comparisons failed. Consequently, R010 did not validate its relative-bound attachment "
        "and never exercised the collision-pinch controller; it is not a scientific exhaustion "
        "of that intended construction.\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "vla-wam-shared-v3e006-r010-closure-manifest-v1",
        "status": TERMINAL,
        "repo_result": binding(result_path, repo_relative=True),
        "repo_target_validation_receipt": binding(receipt_path, repo_relative=True),
        "decision_memo": binding(memo_path, repo_relative=True),
        "closure_tool": binding(Path(__file__), repo_relative=True),
        "closure_validator": binding(
            ROOT / "tools/validate_v3e006_r010_closure.py", repo_relative=True
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
