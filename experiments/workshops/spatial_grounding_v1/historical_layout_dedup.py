"""Fail-closed historical-layout deduplication for frozen SGW LAT proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .fixtures import RESET_ANGLE_TOLERANCE_DEGREES, RESET_POSITION_TOLERANCE_M, Pose, pose_error


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dedup_ledger(proposals: Mapping[str, Any], historical: Mapping[str, Any] | None, *, proposal_sha256: str) -> dict[str, Any]:
    candidates = proposals.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 100:
        raise ValueError("dedup gate requires exactly the frozen 100 LAT proposals")
    base = {
        "schema_version": "sgw-01-lat-historical-layout-dedup-v1",
        "proposal_sha256": proposal_sha256,
        "proposal_count": len(candidates),
        "coordinate_contract": {
            "reset": "actor-root position plus quaternion; <=3 mm and <=2 degrees",
            "scoring": "geometric center reconstructed from root pose plus measured root-local offset",
            "historical_record_requirement": "hash-bound actor-root poses and root-local scoring-center offsets for cube and bowl",
        },
        "model_request_count": 0,
        "behavioral_episode_count": 0,
    }
    if not historical:
        return {**base, "status": "blocked_missing_historical_layout_evidence",
                "coverage": {"historical_record_count": 0, "complete": False},
                "missing_evidence": ["No hash-bound historical LAT actor-root/scoring-center layout registry was supplied."],
                "release_permitted": False, "matches": []}
    records = historical.get("layouts")
    if not isinstance(records, list):
        return {**base, "status": "blocked_invalid_historical_layout_evidence",
                "coverage": {"historical_record_count": 0, "complete": False},
                "missing_evidence": ["Historical registry lacks a layouts list."],
                "release_permitted": False, "matches": []}
    missing = []
    matches = []
    valid = []
    for row in records:
        if not isinstance(row, Mapping) or not isinstance(row.get("sha256"), str):
            missing.append("Historical record lacks a hash binding.")
            continue
        try:
            roots = _poses(row["object_root_poses"])
            offsets = row["scoring_center_offsets_root_local_m"]
            if set(roots) != {"rubiks_cube", "bowl"} or set(offsets) != set(roots):
                raise ValueError("object coverage differs")
            valid.append((str(row.get("layout_id", row["sha256"])), roots, offsets, row["sha256"]))
        except (KeyError, TypeError, ValueError):
            missing.append("Historical record lacks complete cube/bowl root poses and center offsets.")
    for candidate in candidates:
        roots = _poses(candidate["object_poses"])
        offsets = candidate["metadata"]["scoring_center_offsets_root_local_m"]
        for layout_id, old_roots, old_offsets, evidence_hash in valid:
            if _same_layout(roots, offsets, old_roots, old_offsets):
                matches.append({"candidate_id": candidate["candidate_id"], "historical_layout_id": layout_id,
                                "historical_evidence_sha256": evidence_hash})
    complete = not missing
    return {**base,
            "status": "passed_no_historical_duplicates" if complete and not matches else (
                "blocked_historical_duplicate_detected" if matches else "blocked_incomplete_historical_layout_evidence"),
            "coverage": {"historical_record_count": len(records), "valid_record_count": len(valid), "complete": complete},
            "missing_evidence": sorted(set(missing)), "matches": matches,
            "release_permitted": bool(complete and not matches)}


def _poses(value: Mapping[str, Any]) -> dict[str, Pose]:
    return {name: Pose.from_json(row) for name, row in value.items()}


def _same_layout(left: Mapping[str, Pose], left_offsets: Mapping[str, Any], right: Mapping[str, Pose],
                 right_offsets: Mapping[str, Any]) -> bool:
    for name in ("rubiks_cube", "bowl"):
        position, angle = pose_error(left[name], right[name])
        if position > RESET_POSITION_TOLERANCE_M or angle > RESET_ANGLE_TOLERANCE_DEGREES:
            return False
        if any(abs(float(a) - float(b)) > RESET_POSITION_TOLERANCE_M
               for a, b in zip(left_offsets[name], right_offsets[name], strict=True)):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--expected-proposals-sha256", required=True)
    parser.add_argument("--historical-registry", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite dedup ledger: {args.output}")
    observed = sha256(args.proposals)
    if observed != args.expected_proposals_sha256:
        raise ValueError("frozen proposal artifact hash differs")
    historical = json.loads(args.historical_registry.read_text()) if args.historical_registry else None
    args.output.write_text(json.dumps(dedup_ledger(json.loads(args.proposals.read_text()), historical, proposal_sha256=observed),
                                     sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
