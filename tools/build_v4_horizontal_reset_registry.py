#!/usr/bin/env python3
"""Build the prospective V4 horizontal reset registry without model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
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
DEFAULT_REPAIR_AMENDMENT = (
    ROOT
    / "artifacts/online_correction_v4/setup/horizontal_geometry_repair_amendment.candidate.json"
)
DEFAULT_REPAIRED_OUTPUT = (
    ROOT
    / "artifacts/online_correction_v4/setup/horizontal_reset_registry.geometry_repair_v1.candidate.json"
)

SCHEMA_VERSION = "v4-droid-horizontal-reset-registry-v1"
REPAIRED_SCHEMA_VERSION = "v4-droid-horizontal-reset-registry-geometry-repair-v1"
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
    seeds_by_block: dict[int, set[int]] = {}
    for row in queue_rows:
        block = row.get("block_id")
        env_seed = row.get("env_seed")
        if (
            row.get("fixture") == "horizontal"
            and type(block) is int
            and type(env_seed) is int
        ):
            seeds_by_block.setdefault(block, set()).add(env_seed)
    if not seeds_by_block:
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
    block_count = max(block_counts)
    if set(seeds_by_block) != set(range(block_count)) or any(
        len(values) != 1 for values in seeds_by_block.values()
    ):
        raise ResetRegistryBuildError(
            "horizontal queue does not bind exactly one environment seed per block"
        )
    substitutions = seed_cfg.get("post_result_environment_seed_substitutions", [])
    if not isinstance(substitutions, list):
        raise ResetRegistryBuildError("campaign seed substitutions must be a list")
    replacement_by_block = {
        row["block_id"]: row["replacement_seed"]
        for row in substitutions
        if isinstance(row, Mapping) and row.get("fixture") == "horizontal"
    }
    expected = tuple(
        replacement_by_block.get(block, base + stride * slot + block)
        for block in range(block_count)
    )
    observed = tuple(next(iter(seeds_by_block[block])) for block in range(block_count))
    if observed != expected or len(set(observed)) != block_count:
        raise ResetRegistryBuildError(
            "horizontal environment seeds differ from registered block substitutions"
        )
    return observed


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


def _load_geometry_repair_amendment(path: Path) -> dict[str, Any]:
    amendment = _read_json(path)
    if amendment.get("schema_version") not in {
        "v4-horizontal-geometry-repair-amendment-v1",
        "v4-horizontal-geometry-repair-amendment-v2",
    }:
        raise ResetRegistryBuildError("geometry repair amendment schema differs")
    if amendment.get("fixture_id") != "horizontal":
        raise ResetRegistryBuildError("geometry repair amendment fixture differs")
    repair = amendment.get("repair")
    if not isinstance(repair, Mapping):
        raise ResetRegistryBuildError("geometry repair amendment lacks repair block")
    offset = repair.get("cube_robot_base_x_offset_m")
    if (
        isinstance(offset, bool)
        or not isinstance(offset, (int, float))
        or not math.isfinite(float(offset))
        or float(offset) >= 0.0
    ):
        raise ResetRegistryBuildError("geometry repair offset must be a finite negative value")
    return amendment


def build_registry(
    *,
    campaign_path: Path,
    queue_path: Path,
    source_report_path: Path,
    geometry_repair_amendment_path: Path | None = None,
) -> dict[str, Any]:
    campaign = _read_json(campaign_path)
    if campaign.get("campaign_id") != "online_correction_v4":
        raise ResetRegistryBuildError("campaign identity differs")
    queue_rows = _read_jsonl(queue_path)
    seeds = horizontal_env_seeds(campaign=campaign, queue_rows=queue_rows)
    source_report = _read_json(source_report_path)
    base = base_positions_from_v3_report(source_report)
    repair_amendment: dict[str, Any] | None = None
    repair_offset_m = 0.0
    if geometry_repair_amendment_path is not None:
        from experiments.online_correction_v4.horizontal_geometry_repair import (
            FIXTURE_VERSION,
            apply_cube_repair_offset,
            minimum_cube_repair_offset_m,
        )

        repair_amendment = _load_geometry_repair_amendment(
            geometry_repair_amendment_path
        )
        frozen_offset = float(repair_amendment["repair"]["cube_robot_base_x_offset_m"])
        provisional_resets: dict[str, Any] = {}
        for block_index, env_seed in enumerate(seeds):
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
            provisional_resets[str(env_seed)] = {
                "block_index": block_index,
                "jitter_robot_base_xy_m": [dx, dy],
                "positions_robot_base_m": positions,
            }
        selected_offset, clearance_audit = minimum_cube_repair_offset_m(
            base_positions_robot_base_m=base,
            resets_by_env_seed=provisional_resets,
        )
        if abs(selected_offset - frozen_offset) > 1e-12:
            raise ResetRegistryBuildError(
                "geometry repair amendment offset differs from deterministic selection"
            )
        base = apply_cube_repair_offset(base, repair_offset_robot_base_x_m=frozen_offset)
        repair_offset_m = frozen_offset
        if repair_amendment.get("fixture_version") != FIXTURE_VERSION:
            raise ResetRegistryBuildError("geometry repair fixture_version differs")
    scene_asset = "rubiks_cube_banana_bowl.usda"
    scene_metadata_sha256 = (
        "83ecf76a1fde9091b5db9012b76790aca36c2fe6b2c36a8885f4f98d7c4b7e1c"
    )

    def asset_identity(name: str) -> str:
        return f"{scene_asset}::{name}@{scene_metadata_sha256}"

    resets: dict[str, Any] = {}
    for block_index, env_seed in enumerate(seeds):
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
            "block_index": block_index,
            "jitter_robot_base_xy_m": [dx, dy],
            "positions_robot_base_m": positions,
        }

    payload: dict[str, Any] = {
        "schema_version": (
            REPAIRED_SCHEMA_VERSION if repair_amendment is not None else SCHEMA_VERSION
        ),
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
    if repair_amendment is not None:
        payload["fixture_version"] = repair_amendment["fixture_version"]
        payload["geometry_repair"] = {
            "amendment_path": portable_path(geometry_repair_amendment_path),
            "amendment_sha256": sha256_file(geometry_repair_amendment_path),
            "cube_robot_base_x_offset_m": repair_offset_m,
            "application_order": "cube_offset_before_common_xy_jitter",
            "policy_outcome_used": False,
        }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--geometry-repair-amendment", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    payload = build_registry(
        campaign_path=args.campaign.resolve(),
        queue_path=args.queue.resolve(),
        source_report_path=args.source_report.resolve(),
        geometry_repair_amendment_path=(
            args.geometry_repair_amendment.resolve()
            if args.geometry_repair_amendment is not None
            else None
        ),
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
