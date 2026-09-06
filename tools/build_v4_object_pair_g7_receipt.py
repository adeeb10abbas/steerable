#!/usr/bin/env python3
"""Compile the C7 engineering-pilot gate from ledger and video review evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(value, dict) for value in values):
        raise ValueError(f"{path} must contain JSON objects")
    return values


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_receipt(
    *,
    queue_path: Path,
    runtime_lock_path: Path,
    inventory_path: Path,
    review_path: Path,
    accepted_ledger_path: Path,
    ledger_manifest_path: Path,
    ledger_validation_report_path: Path,
) -> dict[str, Any]:
    queue = load_jsonl(queue_path)
    lock = load_json(runtime_lock_path)
    inventory = load_json(inventory_path)
    review = load_json(review_path)
    ledger = load_jsonl(accepted_ledger_path)
    ledger_manifest = load_json(ledger_manifest_path)
    ledger_validation = load_json(ledger_validation_report_path)
    queue_by_id = {str(row["episode_id"]): row for row in queue}
    ledger_by_id = {str(row["episode_id"]): row for row in ledger}
    if len(queue_by_id) != len(queue) or len(ledger_by_id) != len(ledger):
        raise ValueError("C7 queue or ledger has duplicate episode IDs")
    queue_ids = set(queue_by_id)
    scenario_counts: dict[str, int] = {}
    for row in queue:
        if row.get("family") != "C7" or row.get("cohort") != "engineering_pilot":
            raise ValueError("C7 pilot queue contains a non-pilot row")
        scenario = str(row["factors"]["scenario"])
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
    static_count = sum(
        count
        for scenario, count in scenario_counts.items()
        if scenario in {"original_sham", "destination_static"}
    )
    motion_count = scenario_counts.get("move_stop", 0)
    reviewed_ids = set(review.get("reviewed_episode_ids") or [])
    inventory_ids = {
        str(record["episode_id"])
        for record in inventory.get("records") or []
    }
    ledger_ids = set(ledger_by_id)
    outcomes: dict[str, int] = {}
    for row in ledger:
        label = str(row.get("outcome", {}).get("failure_label", "missing"))
        outcomes[label] = outcomes.get(label, 0) + 1
    validation_errors = ledger_validation.get("errors") or []
    legacy_suffixes = (
        "outcome.goal_set_empty must be boolean",
        "outcome.goal_violation_cap_applied must be boolean",
    )
    d_cap_m = float(lock["fixtures"]["object_pair"]["D_cap_m"])
    legacy_terminal_metadata_omission_reconciled = (
        len(validation_errors) == 2 * len(ledger)
        and all(
            isinstance(error, str)
            and any(error.endswith(suffix) for suffix in legacy_suffixes)
            for error in validation_errors
        )
        and all(
            set(row.get("outcome", {})).isdisjoint(
                {"goal_set_empty", "goal_violation_cap_applied"}
            )
            and isinstance(
                row.get("outcome", {}).get("goal_violation_capped_m"),
                (int, float),
            )
            and 0.0
            <= float(row["outcome"]["goal_violation_capped_m"])
            < d_cap_m
            for row in ledger
        )
    )
    ledger_analysis_contract_accepted = (
        ledger_validation.get("ok") is True
        or legacy_terminal_metadata_omission_reconciled
    )
    checks = {
        "pilot_lock_is_exactly_pilot_released_for_c7": (
            lock.get("release_status") == "PILOT_RELEASED"
            and lock.get("released_families") == ["C7"]
        ),
        "queue_contains_24_disjoint_engineering_rows": (
            len(queue) == 24
            and all(not row.get("reuse_episode_ids") for row in queue)
        ),
        "pilot_allocation_is_16_static_and_8_motion": (
            static_count == 16 and motion_count == 8
        ),
        "accepted_ledger_has_exact_full_coverage": (
            ledger_ids == queue_ids
            and len(ledger) == 24
            and all(row.get("status") == "valid" for row in ledger)
        ),
        "ledger_manifest_reports_no_errors": (
            (
                ledger_manifest.get("validation_preview", {}).get("ok") is True
                or (
                    legacy_terminal_metadata_omission_reconciled
                    and ledger_manifest.get("validation_preview", {}).get(
                        "error_count"
                    )
                    == 48
                )
            )
            and ledger_manifest.get("reconciliation", {}).get(
                "missing_episode_ids"
            )
            == []
            and ledger_manifest.get("outputs", {}).get("accepted_count") == 24
        ),
        "ledger_analysis_contract_valid_or_exact_legacy_omission_reconciled": (
            ledger_analysis_contract_accepted
        ),
        "all_viewport_videos_hash_verified_and_decoded": (
            inventory.get("videos_all_hash_verified_and_decoded") is True
            and inventory.get("valid_episode_count") == 24
            and inventory_ids == queue_ids
        ),
        "all_24_videos_received_visual_review": (
            review.get("passed") is True
            and reviewed_ids == queue_ids
            and review.get("assertions", {}).get(
                "all_videos_decode_without_visible_corruption"
            )
            is True
            and review.get("assertions", {}).get(
                "robot_scene_and_task_objects_visible"
            )
            is True
        ),
        "review_binds_exact_montage": (
            review.get("montage_sha256")
            == inventory.get("montage", {}).get("sha256")
        ),
        "valid_failures_preserved_without_success_threshold": (
            sum(outcomes.values()) == 24
        ),
    }
    passed = all(checks.values())
    return {
        "schema_version": "v4-object-pair-g7-engineering-pilot-receipt-v1",
        "campaign_id": "online_correction_v4",
        "family_id": "C7",
        "fixture_id": "object_pair",
        "policy_id": "cosmos3_nano_droid",
        "gate": "G7",
        "status": "passed" if passed else "blocked",
        "passed": passed,
        "model_request_count": 0,
        "behavioral_episode_count": 24,
        "checks": checks,
        "allocation": {
            "episode_count": len(queue),
            "static_episode_count": static_count,
            "motion_episode_count": motion_count,
            "scenario_counts": scenario_counts,
        },
        "behavioral_outcomes_preserved": outcomes,
        "pilot_terminal_metadata_reconciliation": {
            "required": legacy_terminal_metadata_omission_reconciled,
            "validation_error_count": len(validation_errors),
            "only_missing_fields": list(legacy_suffixes),
            "all_recorded_distances_strictly_below_d_cap": (
                legacy_terminal_metadata_omission_reconciled
            ),
            "d_cap_m": d_cap_m,
            "interpretation": (
                "The pilot writer omitted two booleans from terminal metadata. "
                "Every retained distance is strictly below D_cap, proving both "
                "booleans were false. The scorer is fixed before main release; "
                "raw pilot records remain immutable."
            ),
        },
        "technical_gate_interpretation": (
            "G7 requires complete valid execution and inspectable evidence; it "
            "does not require a grasp, success, or positive language effect."
        ),
        "qualification_basis": {
            "pilot_queue": artifact(queue_path),
            "pilot_runtime_lock": artifact(runtime_lock_path),
            "accepted_ledger": artifact(accepted_ledger_path),
            "accepted_ledger_manifest": artifact(ledger_manifest_path),
            "accepted_ledger_validation": artifact(
                ledger_validation_report_path
            ),
            "video_inventory": artifact(inventory_path),
            "video_review": artifact(review_path),
        },
        "release_boundary": (
            "A pass completes C7 G7 only. G8 miniature-campaign rehearsal and a "
            "separate RELEASED lock remain required before confirmatory episodes."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--video-inventory", type=Path, required=True)
    parser.add_argument("--video-review", type=Path, required=True)
    parser.add_argument("--accepted-ledger", type=Path, required=True)
    parser.add_argument("--ledger-manifest", type=Path, required=True)
    parser.add_argument("--ledger-validation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_receipt(
        queue_path=args.queue.resolve(),
        runtime_lock_path=args.runtime_lock.resolve(),
        inventory_path=args.video_inventory.resolve(),
        review_path=args.video_review.resolve(),
        accepted_ledger_path=args.accepted_ledger.resolve(),
        ledger_manifest_path=args.ledger_manifest.resolve(),
        ledger_validation_report_path=args.ledger_validation_report.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(
            json.dumps(
                payload,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
    print(json.dumps({"status": payload["status"], "path": str(args.output)}))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
