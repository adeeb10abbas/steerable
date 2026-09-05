#!/usr/bin/env python3
"""Build the formula-closed, zero-inference horizontal G3 check plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.online_correction_v4.model_blind_g2 import (  # noqa: E402
    canonical_json_bytes,
)
from experiments.online_correction_v4.model_blind_g3 import (  # noqa: E402
    build_plan_payload,
    validate_plan_payload,
)

DEFAULT_CAMPAIGN = ROOT / "docs/online_correction_v4/campaign.json"
DEFAULT_QUEUE = ROOT / "artifacts/online_correction_v4/queue.jsonl"
DEFAULT_MOTION = ROOT / "artifacts/online_correction_v4/motion_manifest.json"
DEFAULT_REGISTRY = (
    ROOT
    / "artifacts/online_correction_v4/setup/horizontal_reset_registry.candidate.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/online_correction_v4/setup/horizontal_g3_plan.candidate.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def build(
    *,
    campaign_path: Path,
    queue_path: Path,
    motion_path: Path,
    registry_path: Path,
    output_path: Path,
) -> dict:
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    motion = json.loads(motion_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if campaign.get("campaign_id") != "online_correction_v4":
        raise ValueError("campaign identity differs")
    if registry.get("status") != "model_blind_candidate_not_released_for_inference":
        raise ValueError("G3 plan requires the unreleased model-blind reset candidate")
    if registry.get("model_request_count") != 0 or registry.get(
        "behavioral_episode_count"
    ) != 0:
        raise ValueError("reset candidate is not model-blind")
    if motion.get("calibration", {}).get("status") != "pending_model_blind_geometry_gate":
        raise ValueError("motion manifest calibration state differs")
    rows = [
        json.loads(line)
        for line in queue_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    motion_config = campaign["motion"]
    nominal = float(campaign["fixtures"]["horizontal"]["nominal_translation_m"])
    plan = build_plan_payload(
        source_identity={
            "campaign": {
                "path": portable_path(campaign_path),
                "sha256": sha256_file(campaign_path),
            },
            "queue": {
                "path": portable_path(queue_path),
                "sha256": sha256_file(queue_path),
            },
            "motion_manifest": {
                "path": portable_path(motion_path),
                "sha256": sha256_file(motion_path),
            },
            "reset_registry": {
                "path": portable_path(registry_path),
                "sha256": sha256_file(registry_path),
            },
        },
        reset_registry=registry,
        queue_rows=rows,
        scale_candidates=motion_config["calibration_scale_candidates"],
        nominal_displacement_m=nominal,
        minimum_shrinking_area_fraction=float(
            motion_config["primary_horizontal_minimum_shrinking_goal_area_fraction"]
        ),
    )
    validate_plan_payload(plan)
    body = canonical_json_bytes(plan)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(body)
    return {
        "path": portable_path(output_path),
        "sha256": hashlib.sha256(body).hexdigest(),
        "bytes": len(body),
        "registered_reset_count": plan["registered_reset_count"],
        "path_checks_per_scale": plan["path_sweep"]["checks_per_scale"],
        "maximum_path_checks": plan["path_sweep"][
            "maximum_checks_across_scale_ladder"
        ],
        "scripted_checks_per_final_candidate": plan["scripted_controller"][
            "checks_per_final_geometry_candidate"
        ],
        "model_request_count": 0,
        "behavioral_episode_count": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--motion-manifest", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--reset-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = build(
        campaign_path=args.campaign.resolve(),
        queue_path=args.queue.resolve(),
        motion_path=args.motion_manifest.resolve(),
        registry_path=args.reset_registry.resolve(),
        output_path=args.out.resolve(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
