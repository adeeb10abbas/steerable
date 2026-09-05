#!/usr/bin/env python3
"""Build the prospective V4 horizontal reset registry without model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = ROOT / "docs/online_correction_v4/campaign.json"
DEFAULT_QUEUE = ROOT / "artifacts/online_correction_v4/queue.jsonl"
DEFAULT_SOURCE = (
    ROOT
    / "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/"
    "model_blind_calibration_report.json"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts/online_correction_v4/setup/horizontal_reset_registry.candidate.json"
)

SCHEMA_VERSION = "v4-droid-horizontal-reset-registry-v1"
STATUS = "model_blind_candidate_not_released_for_inference"
JITTER_ALGORITHM = "sha256_u64_independent_axes_common_scene_translation_v1"
JITTER_HALF_RANGE_X_M = 0.015
JITTER_HALF_RANGE_Y_M = 0.015
MOVABLE_OBJECTS = ("rubiks_cube", "bowl", "banana")


class ResetRegistryBuildError(ValueError):
    """Raised when prospective reset inputs are missing or inconsistent."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResetRegistryBuildError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResetRegistryBuildError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ResetRegistryBuildError(f"cannot read queue {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ResetRegistryBuildError(
                f"invalid queue JSON at {path}:{line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise ResetRegistryBuildError(
                f"queue row at {path}:{line_number} must be an object"
            )
        rows.append(row)
    return rows


def horizontal_env_seeds(
    *,
    campaign: Mapping[str, Any],
    queue_rows: Iterable[Mapping[str, Any]],
) -> tuple[int, ...]:
    seed_cfg = campaign.get("seed_reservation")
    fixture_cfg = (campaign.get("fixtures") or {}).get("horizontal")
    if not isinstance(seed_cfg, Mapping) or not isinstance(fixture_cfg, Mapping):
        raise ResetRegistryBuildError("campaign lacks horizontal seed reservation")
    base = seed_cfg.get("environment_base")
    stride = seed_cfg.get("fixture_stride")
    slot = fixture_cfg.get("seed_slot")
    if type(base) is not int or type(stride) is not int or type(slot) is not int:
        raise ResetRegistryBuildError("campaign horizontal seed parameters must be integers")
    seeds = sorted(
        {
            row.get("env_seed")
            for row in queue_rows
            if row.get("fixture") == "horizontal" and type(row.get("env_seed")) is int
        }
    )
    if not seeds:
        raise ResetRegistryBuildError("queue has no horizontal environment seeds")
    families = campaign.get("families")
    if not isinstance(families, list):
        raise ResetRegistryBuildError("campaign families must be a list")
    block_counts = [
        row.get("blocks")
        for row in families
        if isinstance(row, Mapping) and row.get("seed_group") == "horizontal"
    ]
    if not block_counts or any(type(count) is not int or count <= 0 for count in block_counts):
        raise ResetRegistryBuildError("campaign lacks valid horizontal block counts")
    expected = list(
        range(base + stride * slot, base + stride * slot + max(block_counts))
    )
    if seeds != expected:
        raise ResetRegistryBuildError(
            "horizontal environment seeds are not the contiguous registered block range"
        )
    return tuple(seeds)


def base_positions_from_v3_report(report: Mapping[str, Any]) -> dict[str, list[float]]:
    if report.get("model_request_count") != 0 or report.get("behavioral_episode_count") != 0:
        raise ResetRegistryBuildError("source calibration report is not model-blind")
    tasks = report.get("tasks")
    if not isinstance(tasks, list):
        raise ResetRegistryBuildError("source calibration report lacks tasks")
    matches = [
        task
        for task in tasks
        if isinstance(task, Mapping)
        and task.get("arm") == "control"
        and task.get("relation") == "left"
    ]
    if len(matches) != 1:
        raise ResetRegistryBuildError("source report lacks one control-left task")
    repeats = matches[0].get("repeat_resets")
    if not isinstance(repeats, list) or not repeats:
        raise ResetRegistryBuildError("source control-left task lacks reset evidence")
    positions = repeats[0].get("positions_robot_base_m")
    if not isinstance(positions, Mapping) or set(positions) != set(MOVABLE_OBJECTS):
        raise ResetRegistryBuildError("source reset movable-object inventory differs")
    result: dict[str, list[float]] = {}
    for name in MOVABLE_OBJECTS:
        vector = positions.get(name)
        if (
            not isinstance(vector, list)
            or len(vector) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in vector
            )
        ):
            raise ResetRegistryBuildError(f"source position for {name} is invalid")
        result[name] = [float(value) for value in vector]
    return result


def deterministic_axis_jitter(*, env_seed: int, axis: str, half_range_m: float) -> float:
    if axis not in {"x", "y"}:
        raise ResetRegistryBuildError(f"unsupported jitter axis: {axis}")
    digest = hashlib.sha256(
        f"online-correction-v4:horizontal:{env_seed}:{axis}:{JITTER_ALGORITHM}".encode(
            "utf-8"
        )
    ).digest()
    unit = int.from_bytes(digest[:8], "big") / float((1 << 64) - 1)
    return (2.0 * unit - 1.0) * half_range_m


def build_registry(
    *,
    campaign_path: Path,
    queue_path: Path,
    source_report_path: Path,
) -> dict[str, Any]:
    campaign = _read_json(campaign_path)
    if campaign.get("campaign_id") != "online_correction_v4":
        raise ResetRegistryBuildError("campaign identity differs")
    queue_rows = _read_jsonl(queue_path)
    seeds = horizontal_env_seeds(campaign=campaign, queue_rows=queue_rows)
    source_report = _read_json(source_report_path)
    base = base_positions_from_v3_report(source_report)
    scene_asset = "rubiks_cube_banana_bowl.usda"
    scene_metadata_sha256 = (
        "83ecf76a1fde9091b5db9012b76790aca36c2fe6b2c36a8885f4f98d7c4b7e1c"
    )

    def asset_identity(name: str) -> str:
        return f"{scene_asset}::{name}@{scene_metadata_sha256}"

    resets: dict[str, Any] = {}
    for env_seed in seeds:
        dx = deterministic_axis_jitter(
            env_seed=env_seed, axis="x", half_range_m=JITTER_HALF_RANGE_X_M
        )
        dy = deterministic_axis_jitter(
            env_seed=env_seed, axis="y", half_range_m=JITTER_HALF_RANGE_Y_M
        )
        positions = {
            name: [
                base[name][0] + dx,
                base[name][1] + dy,
                base[name][2],
            ]
            for name in MOVABLE_OBJECTS
        }
        resets[str(env_seed)] = {
            "block_index": env_seed - seeds[0],
            "jitter_robot_base_xy_m": [dx, dy],
            "positions_robot_base_m": positions,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": "online_correction_v4",
        "fixture_id": "horizontal",
        "status": STATUS,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "scene_asset": scene_asset,
        "scene_metadata_sha256": scene_metadata_sha256,
        "contact_objects": ["rubiks_cube", "banana", "bowl", "table"],
        "object_roles": {
            "target": {
                "scene_object": "rubiks_cube",
                "asset_identity": asset_identity("rubiks_cube"),
            },
            "reference": {
                "scene_object": "bowl",
                "asset_identity": asset_identity("bowl"),
            },
            "distractor": {
                "scene_object": "banana",
                "asset_identity": asset_identity("banana"),
            },
        },
        "source_identity": {
            "campaign_path": portable_path(campaign_path),
            "campaign_sha256": sha256_file(campaign_path),
            "queue_path": portable_path(queue_path),
            "queue_sha256": sha256_file(queue_path),
            "v3_model_blind_calibration_report_path": portable_path(
                source_report_path
            ),
            "v3_model_blind_calibration_report_sha256": sha256_file(
                source_report_path
            ),
            "base_positions_robot_base_m": base,
        },
        "reset_jitter": {
            "algorithm": JITTER_ALGORITHM,
            "independent_source": "registered_environment_seed",
            "x_half_range_m": JITTER_HALF_RANGE_X_M,
            "y_half_range_m": JITTER_HALF_RANGE_Y_M,
            "z_jitter_m": 0.0,
            "application": (
                "one common robot-base x/y translation is applied to all movable "
                "objects, preserving their relative geometry"
            ),
            "policy_outcome_used": False,
        },
        "registered_env_seed_count": len(seeds),
        "registered_env_seed_min": min(seeds),
        "registered_env_seed_max": max(seeds),
        "resets_by_env_seed": resets,
        "release_boundary": (
            "Prospective model-blind reset candidate only. Live repeated-reset, "
            "support/contact, camera-visibility, frame-axis, and swept-path gates "
            "must pass before any policy inference."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    payload = build_registry(
        campaign_path=args.campaign.resolve(),
        queue_path=args.queue.resolve(),
        source_report_path=args.source_report.resolve(),
    )
    body = canonical_json_bytes(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(body)
    print(
        json.dumps(
            {
                "path": str(args.out.resolve()),
                "sha256": sha256_bytes(body),
                "registered_env_seed_count": payload["registered_env_seed_count"],
                "status": payload["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
