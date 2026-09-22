"""Materialize SGW LAT candidates only from a measured RoboLab workspace receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .fixtures import MAX_CANDIDATES_PER_FAMILY, FixtureCandidate, write_json


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize_lat_candidates(workspace: Mapping[str, Any], *, seed: int) -> list[dict[str, Any]]:
    """Select at most 100 candidate poses from measured workspace slots.

    The input receipt is expected to be produced by an assigned RoboLab lane.
    It supplies actual settled poses and validated Abs-IK waypoint sequences;
    this function deliberately has no numeric pose generator.
    """

    if workspace.get("schema_version") != "sgw-01-lat-measured-workspace-v1":
        raise ValueError("LAT candidates require a measured-workspace v1 receipt")
    if workspace.get("model_request_count") != 0 or workspace.get("behavioral_episode_count") != 0:
        raise ValueError("workspace receipt must remain model blind")
    asset_manifest_sha256 = str(workspace.get("asset_manifest_sha256", ""))
    if len(asset_manifest_sha256) != 64:
        raise ValueError("workspace receipt lacks an immutable asset manifest hash")
    slots = workspace.get("validated_slots")
    if not isinstance(slots, list) or not slots:
        raise ValueError("workspace receipt has no validated LAT slots")
    if len(slots) > MAX_CANDIDATES_PER_FAMILY:
        raise ValueError("measured workspace exceeds the frozen 100-candidate cap")
    ordered = sorted(
        slots,
        key=lambda row: hashlib.sha256(f"{seed}|{json.dumps(row, sort_keys=True)}".encode()).hexdigest(),
    )
    candidates: list[dict[str, Any]] = []
    for index, slot in enumerate(ordered, 1):
        candidate = {
            "candidate_id": f"LAT-{index:03d}",
            "family": "LAT",
            "seed": seed,
            "asset_manifest_sha256": asset_manifest_sha256,
            "task_asset": str(workspace["task_asset"]),
            "object_poses": slot["object_poses"],
            "metadata": {
                "workspace_receipt_sha256": workspace["receipt_sha256"],
                "source_slot_id": slot["slot_id"],
                "abs_ik_waypoints": slot["abs_ik_waypoints"],
                "historical_layout_fingerprint": None,
            },
        }
        FixtureCandidate.from_json(candidate)
        candidates.append(candidate)
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260922)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite candidate directory: {args.output_dir}")
    workspace = json.loads(args.workspace_receipt.read_text(encoding="utf-8"))
    expected = workspace.get("receipt_sha256")
    observed = _sha256(args.workspace_receipt)
    if expected != observed:
        raise ValueError("workspace receipt self-hash does not match its bytes")
    candidates = materialize_lat_candidates(workspace, seed=args.seed)
    for candidate in candidates:
        write_json(args.output_dir / "LAT" / f"{candidate['candidate_id']}.json", candidate)
    write_json(args.output_dir / "LAT" / "manifest.json", {
        "schema_version": "sgw-01-lat-candidate-manifest-v1",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "workspace_receipt_sha256": observed,
        "seed": args.seed,
        "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
    })


if __name__ == "__main__":
    main()
