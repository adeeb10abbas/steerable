#!/usr/bin/env python3
"""Fail-closed validation for the repaired horizontal geometry inventory."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.online_correction_v4.horizontal_geometry_repair import (  # noqa: E402
    COHORT,
    FIXTURE_VERSION,
    minimum_cube_repair_offset_m,
)

EXPECTED_ROW_COUNT = 9728
AFFECTED_FAMILIES = ("C1", "C3", "C4")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(
    *,
    amendment_path: Path,
    inventory_manifest_path: Path,
    queue_path: Path,
    reset_registry_path: Path,
    g3_plan_path: Path,
    historical_queue_path: Path,
) -> list[str]:
    errors: list[str] = []
    amendment = load_json(amendment_path)
    manifest = load_json(inventory_manifest_path)
    rows = load_jsonl(queue_path)
    registry = load_json(reset_registry_path)
    plan = load_json(g3_plan_path)
    historical_rows = load_jsonl(historical_queue_path)
    historical_ids = {row["episode_id"] for row in historical_rows}

    if amendment.get("fixture_version") != FIXTURE_VERSION:
        errors.append("amendment fixture_version differs")
    if manifest.get("row_count") != EXPECTED_ROW_COUNT:
        errors.append("inventory manifest row_count differs")
    if len(rows) != EXPECTED_ROW_COUNT:
        errors.append("repaired queue row_count differs")
    if {row["episode_id"] for row in rows} & historical_ids:
        errors.append("repaired queue episode_id collides with historical queue")

    repaired_ids = {row["episode_id"] for row in rows}
    for row in rows:
        if row.get("family") not in AFFECTED_FAMILIES:
            errors.append(f"unexpected family in repaired queue: {row.get('family')}")
        if row.get("fixture_version") != FIXTURE_VERSION:
            errors.append("repaired queue fixture_version differs")
            break
        if row.get("cohort") != COHORT:
            errors.append("repaired queue cohort differs")
            break
        for reuse_id in row.get("reuse_episode_ids", []):
            if reuse_id not in repaired_ids:
                errors.append(f"{row['episode_id']} reuses non-repaired episode {reuse_id}")

    offset = float(amendment["repair"]["cube_robot_base_x_offset_m"])
    if registry.get("fixture_version") != FIXTURE_VERSION:
        errors.append("reset registry fixture_version differs")
    if registry.get("geometry_repair", {}).get("cube_robot_base_x_offset_m") != offset:
        errors.append("reset registry repair offset differs from amendment")
    if manifest["reset_registry"]["sha256"] != sha256_file(reset_registry_path):
        errors.append("inventory manifest reset_registry hash differs")
    if manifest["queue"]["sha256"] != sha256_file(queue_path):
        errors.append("inventory manifest queue hash differs")

    selected_offset, _audit = minimum_cube_repair_offset_m(
        base_positions_robot_base_m=registry["source_identity"]["base_positions_robot_base_m"],
        resets_by_env_seed=registry["resets_by_env_seed"],
    )
    if abs(selected_offset - offset) > 1e-12:
        errors.append("amendment offset differs from recomputed clearance selection")

    if plan.get("fixture_version") != FIXTURE_VERSION:
        errors.append("g3 plan fixture_version differs")
    if plan.get("geometry_repair_mode") is not True:
        errors.append("g3 plan is not marked geometry_repair_mode")
    if plan.get("plan_status") != "pending_repaired_g2_prerequisite":
        errors.append("repaired g3 plan is not blocked pending G2")

    original_registry_path = ROOT / "artifacts/online_correction_v4/setup/horizontal_reset_registry.candidate.json"
    if sha256_file(original_registry_path) == sha256_file(reset_registry_path):
        errors.append("repaired reset registry is byte-identical to original layout")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--amendment",
        type=Path,
        default=ROOT
        / "artifacts/online_correction_v4/setup/horizontal_geometry_repair_amendment.candidate.json",
    )
    parser.add_argument(
        "--inventory-manifest",
        type=Path,
        default=ROOT
        / "artifacts/online_correction_v4/setup/horizontal_geometry_repair_inventory_v1.json",
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=ROOT / "artifacts/online_correction_v4/queue_horizontal_geometry_repair_v1.jsonl",
    )
    parser.add_argument(
        "--reset-registry",
        type=Path,
        default=ROOT
        / "artifacts/online_correction_v4/setup/horizontal_reset_registry.geometry_repair_v1.candidate.json",
    )
    parser.add_argument(
        "--g3-plan",
        type=Path,
        default=ROOT
        / "artifacts/online_correction_v4/setup/horizontal_g3_plan.geometry_repair_v1.candidate.json",
    )
    parser.add_argument(
        "--historical-queue",
        type=Path,
        default=ROOT / "artifacts/online_correction_v4/queue.jsonl",
    )
    args = parser.parse_args()
    errors = validate(
        amendment_path=args.amendment.resolve(),
        inventory_manifest_path=args.inventory_manifest.resolve(),
        queue_path=args.queue.resolve(),
        reset_registry_path=args.reset_registry.resolve(),
        g3_plan_path=args.g3_plan.resolve(),
        historical_queue_path=args.historical_queue.resolve(),
    )
    if errors:
        print(json.dumps({"passed": False, "errors": errors}, indent=2))
        return 1
    print(json.dumps({"passed": True, "errors": []}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
