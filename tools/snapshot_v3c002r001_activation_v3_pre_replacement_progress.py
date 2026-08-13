#!/usr/bin/env python3
"""Capture outcome-blind completed/partial/unstarted activation-v3 block state.

This snapshot is taken before the replacement-source gate.  It reads only
completion markers and raw-file hashes; it never opens a behavioral outcome.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import ContractError, file_binding, read_finite_json, require, sha256_file, validate_file_binding
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.contract import validate_assignment


ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002r001/activation_v3"
SCHEMA = "vla-wam-shared-v3c002r001-activation-v3-pre-replacement-progress-snapshot-v1"
PARTIAL_SLOTS = {"repair-lane-00": 12060, "repair-lane-01": 12101}


def _slots(values: list[str], expected: set[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        slot, sep, raw = value.partition("=")
        require(sep == "=" and slot in expected and raw and slot not in result, f"{label} must be a unique SLOT=PATH")
        result[slot] = Path(raw).resolve()
    require(set(result) == expected, f"{label} does not have the required slots")
    return result


def _partial_seed(path: Path, slot: str) -> int:
    require(path.is_file() and path.stat().st_size > 0, f"partial ledger missing: {slot}")
    seeds = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        require(isinstance(row, dict) and row.get("schema_version") == "vla-wam-shared-v3c002-infrastructure-attempt-v1" and row.get("infrastructure_status") == "infrastructure_invalid_excluded" and row.get("denominator_eligible") is False and row.get("authorization_mode") == "behavioral" and row.get("entire_partial_block_invalidated") is True, f"partial ledger does not retain infrastructure-invalid attempt: {slot}")
        seed_block = str(row.get("seed_block_id", ""))
        prefix = "v3c002:seed"
        require(seed_block.startswith(prefix) and seed_block[len(prefix):].isdigit(), f"partial ledger seed invalid: {slot}")
        seeds.add(int(seed_block[len(prefix):]))
    require(seeds == {PARTIAL_SLOTS[slot]}, f"partial ledger does not describe exactly the registered retry seed: {slot}")
    return next(iter(seeds))


def _marker(path: Path, seed: int, slot: str) -> dict:
    value = read_finite_json(path)
    require(isinstance(value, dict) and value.get("schema_version") == "vla-wam-shared-v3c002-completed-block-v1" and value.get("status") == "completed_behavioral_block" and value.get("authorization_mode") == "behavioral" and value.get("episode_seed") == seed, f"completion marker invalid: {slot}/{seed}")
    raws = value.get("raw_episodes")
    require(isinstance(raws, list) and len(raws) == 4, f"completion marker raw count invalid: {slot}/{seed}")
    for raw in raws:
        validate_file_binding(raw, f"completed raw {slot}/{seed}")
    return file_binding(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-root", type=Path, default=V3)
    parser.add_argument("--lane-raw-root", action="append", default=[], required=True, help="SLOT=the exact runner --raw-root (without /behavioral)")
    parser.add_argument("--partial-ledger", action="append", default=[], required=True, help="repair-lane-00/01 only")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"refusing to overwrite progress snapshot: {args.output}")
    all_slots = {f"repair-lane-{n:02d}" for n in range(8)}
    roots = _slots(args.lane_raw_root, all_slots, "lane raw roots")
    ledgers = _slots(args.partial_ledger, set(PARTIAL_SLOTS), "partial ledgers")
    release = read_finite_json(args.activation_root / "release_gate.released.json")
    require(isinstance(release, dict) and release.get("passed") is True, "activation-v3 release is not passed")
    assignments = validate_assignment(release.get("assignment_manifest"))
    lane_records = {}
    for record in release.get("lane_manifests", []):
        binding = validate_file_binding(record, "activation-v3 lane manifest")
        lane = read_finite_json(Path(binding["path"]))
        require(isinstance(lane, dict) and lane.get("lane_slot") not in lane_records, "activation-v3 lane manifest set invalid")
        lane_records[lane["lane_slot"]] = (binding, lane)
    require(set(lane_records) == all_slots, "activation-v3 lane manifest coverage invalid")
    lanes, total = {}, 0
    for slot in sorted(all_slots):
        old_binding, lane = lane_records[slot]
        require(str(roots[slot]) == lane.get("raw_root"), f"raw root does not match frozen lane: {slot}")
        assigned = [int(row["episode_seed"]) for row in assignments if row["lane_slot"] == slot]
        completed, markers = [], []
        for seed in assigned:
            marker = roots[slot] / "behavioral" / f"seed{seed}" / "completed_block.json"
            if marker.is_file():
                completed.append(seed); markers.append({"episode_seed": seed, "marker": _marker(marker, seed, slot)})
        partial = [_partial_seed(ledgers[slot], slot)] if slot in PARTIAL_SLOTS else []
        require(not (set(completed) & set(partial)), f"partial seed already has a completed marker: {slot}")
        unstarted = sorted(set(assigned) - set(completed) - set(partial))
        lanes[slot] = {"old_lane_manifest": old_binding, "raw_root": str(roots[slot]), "assigned_seed_blocks": assigned, "completed_seed_blocks": completed, "partial_seed_blocks": partial, "unstarted_seed_blocks": unstarted, "completed_block_markers": markers}
        total += len(completed)
    value = {"schema_version": SCHEMA, "repair_id": "V3-C002-R001", "activation_id": "V3-C002-R001-A002", "status": "captured_pre_replacement_activation_progress", "passed": True, "captured_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "outcome_blind": True, "release_gate": file_binding(args.activation_root / "release_gate.released.json"), "queue": validate_file_binding(release.get("queue"), "activation-v3 queue"), "assignment_manifest": validate_file_binding(release.get("assignment_manifest"), "activation-v3 assignment"), "partial_infrastructure_ledgers": {slot: file_binding(path) for slot, path in ledgers.items()}, "lanes": lanes, "completed_behavioral_block_count": total, "replacement_gate_model_request_count": 0, "replacement_gate_behavioral_action_count": 0, "replacement_gate_behavioral_episode_count": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": value["status"], "completed_behavioral_block_count": total, "sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
