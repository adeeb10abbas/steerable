#!/usr/bin/env python3
"""Record an explicit review of a live V4 DROID task-frame overlay."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.online_correction_v4.model_blind_g2 import (  # noqa: E402
    axis_review_schema,
    canonical_json_bytes,
    seed_receipt_schema,
    sha256_file,
)


def _write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = canonical_json_bytes(payload)
    with path.open("xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-id", default="horizontal")
    parser.add_argument("--seed-receipt", type=Path, required=True)
    parser.add_argument("--axis-overlay", type=Path, required=True)
    parser.add_argument("--reviewer-identity", required=True)
    parser.add_argument("--review-notes", default="")
    parser.add_argument("--left-axis-matches-fixed-robot-viewpoint", action="store_true")
    parser.add_argument("--front-axis-points-toward-robot", action="store_true")
    parser.add_argument("--up-axis-opposes-gravity", action="store_true")
    parser.add_argument("--labels-and-arrow-origins-visible", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    seed_path = args.seed_receipt.resolve()
    overlay_path = args.axis_overlay.resolve()
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    if seed.get("schema_version") != seed_receipt_schema(args.fixture_id):
        raise ValueError("seed receipt schema differs")
    if seed.get("fixture_id") != args.fixture_id:
        raise ValueError("seed receipt fixture differs")
    if seed.get("model_request_count") != 0 or seed.get("behavioral_episode_count") != 0:
        raise ValueError("seed receipt is not model-blind")
    montage = (
        ((seed.get("artifacts") or {}).get("axis_overlay_images") or {}).get(
            "montage"
        )
    )
    if not isinstance(montage, dict):
        raise ValueError("seed receipt lacks axis overlay montage identity")
    if (
        not overlay_path.is_file()
        or sha256_file(overlay_path) != montage.get("sha256")
        or overlay_path.stat().st_size != montage.get("bytes")
    ):
        raise ValueError("reviewed axis overlay differs from seed receipt")
    reviewer = args.reviewer_identity.strip()
    if not reviewer:
        raise ValueError("reviewer identity must be nonempty")
    assertions = {
        "left_axis_matches_fixed_robot_viewpoint": bool(
            args.left_axis_matches_fixed_robot_viewpoint
        ),
        "front_axis_points_toward_robot": bool(
            args.front_axis_points_toward_robot
        ),
        "up_axis_opposes_gravity": bool(args.up_axis_opposes_gravity),
        "labels_and_arrow_origins_visible": bool(
            args.labels_and_arrow_origins_visible
        ),
    }
    if not all(assertions.values()):
        raise ValueError("all rendered-axis review assertions must be explicit")
    payload = {
        "schema_version": axis_review_schema(args.fixture_id),
        "campaign_id": "online_correction_v4",
        "fixture_id": args.fixture_id,
        "status": "passed",
        "passed": True,
        "rendered_left_front_up": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "reviewer_identity": reviewer,
        "reviewed_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "review_notes": args.review_notes,
        "source_seed_receipt": {
            "path": str(seed_path),
            "sha256": sha256_file(seed_path),
            "bytes": seed_path.stat().st_size,
            "environment_seed": seed.get("environment_seed"),
        },
        "source_axis_overlay": {
            "path": str(overlay_path),
            "sha256": sha256_file(overlay_path),
            "bytes": overlay_path.stat().st_size,
        },
        "assertions": assertions,
        "release_boundary": (
            "This review contributes only the G2 rendered-axis check. It does not "
            "authorize policy inference without complete seed coverage and later gates."
        ),
    }
    _write_exclusive(args.out.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
