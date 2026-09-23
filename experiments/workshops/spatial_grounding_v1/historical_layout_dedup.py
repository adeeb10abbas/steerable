"""Fail-closed historical-layout deduplication for frozen SGW LAT proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
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
            "historical_record_requirement": "independently verified source/hash bindings, actor-root poses, and root-local scoring-center offsets for cube and bowl",
        },
        "coverage_policy": (
            "This initial gate is diagnostic-only. A supplied registry is not proof that "
            "all historical layouts were supplied, so it never authorizes release."
        ),
        "model_request_count": 0,
        "behavioral_episode_count": 0,
    }
    if not historical:
        return {**base, "status": "blocked_missing_historical_layout_evidence",
                "coverage": {"historical_record_count": 0, "valid_geometry_record_count": 0,
                             "exhaustive_source_verified": False},
                "missing_evidence": ["No hash-bound historical LAT actor-root/scoring-center layout registry was supplied."],
                "release_permitted": False, "matches": []}
    records = historical.get("layouts")
    if not isinstance(records, list):
        return {**base, "status": "blocked_invalid_historical_layout_evidence",
                "coverage": {"historical_record_count": 0, "valid_geometry_record_count": 0,
                             "exhaustive_source_verified": False},
                "missing_evidence": ["Historical registry lacks a layouts list."],
                "release_permitted": False, "matches": []}
    missing = []
    matches = []
    valid = []
    for row in records:
        if not isinstance(row, Mapping):
            missing.append("Historical record is not an object.")
            continue
        try:
            roots = _poses(row["object_root_poses"])
            offsets = _offsets(row["scoring_center_offsets_root_local_m"])
            if set(roots) != {"rubiks_cube", "bowl"} or set(offsets) != set(roots):
                raise ValueError("object coverage differs")
            valid.append((str(row.get("layout_id", "unidentified")), roots, offsets, row.get("sha256")))
        except (KeyError, TypeError, ValueError):
            missing.append("Historical record lacks complete cube/bowl root poses and center offsets.")
            continue
        if not _has_unverified_source_binding(row):
            missing.append("Historical record lacks a syntactically valid source/hash binding.")
    for candidate in candidates:
        try:
            roots = _poses(candidate["object_poses"])
            offsets = _offsets(candidate["metadata"]["scoring_center_offsets_root_local_m"])
        except (KeyError, TypeError, ValueError):
            missing.append(f"Frozen candidate {candidate.get('candidate_id', '<unknown>')} has invalid root/center geometry.")
            continue
        for layout_id, old_roots, old_offsets, evidence_hash in valid:
            if _same_layout(roots, offsets, old_roots, old_offsets):
                matches.append({"candidate_id": candidate["candidate_id"], "historical_layout_id": layout_id,
                                "historical_evidence_sha256": evidence_hash})
    return {**base,
            "status": "blocked_historical_duplicate_detected" if matches else (
                "blocked_missing_historical_layout_evidence" if not records else
                "blocked_unverified_historical_layout_coverage"),
            "coverage": {"historical_record_count": len(records), "valid_geometry_record_count": len(valid),
                         "exhaustive_source_verified": False},
            "missing_evidence": sorted(set(missing)), "matches": matches,
            "release_permitted": False}


def _poses(value: Mapping[str, Any]) -> dict[str, Pose]:
    return {name: Pose.from_json(row) for name, row in value.items()}


def _offsets(value: Mapping[str, Any]) -> dict[str, tuple[float, float, float]]:
    if not isinstance(value, Mapping):
        raise ValueError("center offsets must be an object")
    result = {}
    for name, raw in value.items():
        if isinstance(raw, (str, bytes)):
            raise ValueError("center offset must be a three-vector")
        offset = tuple(float(component) for component in raw)
        if len(offset) != 3 or not all(math.isfinite(component) for component in offset):
            raise ValueError("center offset must be a finite three-vector")
        result[str(name)] = offset
    return result


def _has_unverified_source_binding(row: Mapping[str, Any]) -> bool:
    """Validate syntax only; this gate deliberately does not claim source verification."""
    digest = row.get("sha256")
    source = row.get("source")
    return (
        isinstance(digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
        and isinstance(source, Mapping)
        and isinstance(source.get("path"), str)
        and bool(source["path"])
        and source.get("sha256") == digest
    )


def _same_layout(left: Mapping[str, Pose], left_offsets: Mapping[str, Any], right: Mapping[str, Pose],
                 right_offsets: Mapping[str, Any]) -> bool:
    for name in ("rubiks_cube", "bowl"):
        position, angle = pose_error(left[name], right[name])
        if position > RESET_POSITION_TOLERANCE_M or angle > RESET_ANGLE_TOLERANCE_DEGREES:
            return False
        if any(abs(a - b) > RESET_POSITION_TOLERANCE_M
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
