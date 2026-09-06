#!/usr/bin/env python3
"""Build model-blind C5 vertical and C6 containment reset candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.online_correction_v4.droid_task_files.constants import (  # noqa: E402
    fixture_object_spec,
)


STATUS = "model_blind_candidate_not_released_for_inference"
JITTER_ALGORITHM = "sha256_u64_independent_axes_common_scene_translation_v1"
JITTER_HALF_RANGE_M = 0.015

FIXTURE_BUILD_SPECS: dict[str, dict[str, Any]] = {
    "vertical": {
        "family_id": "C5",
        "objects": {
            "cube": {
                "role": "target",
                "primitive": "box",
                "dimensions_m": [0.04, 0.04, 0.04],
                "mass_kg": 0.04,
                "diagonal_inertia_kg_m2": [
                    0.0000106667,
                    0.0000106667,
                    0.0000106667,
                ],
                "base_position_robot_m": [0.35, -0.18, 0.073],
            },
            "bowl": {
                "role": "reference",
                "primitive": "open_box_compound",
                "outer_dimensions_m": [0.12, 0.12, 0.06],
                "interior_dimensions_m": [0.10, 0.10, 0.0525],
                "mass_kg": 0.22,
                "diagonal_inertia_kg_m2": [0.00035, 0.00035, 0.00055],
                "base_position_robot_m": [0.53, 0.10, 0.2805],
            },
        },
        "support_geometry": {
            "kind": "three_stationary_shelves",
            "shelf_center_robot_x_m": 0.53,
            "shelf_width_y_m": 0.48,
            "shelf_depth_x_m": 0.28,
            "shelf_thickness_m": 0.02,
            "shelf_center_z_m": [0.11, 0.26, 0.41],
            "reference_support": "shelf_middle",
            "above_goal_support": "shelf_top",
            "below_goal_support": "shelf_bottom",
            "reference_motion_axis": "robot_base_y_left_right_only",
        },
    },
    "containment": {
        "family_id": "C6",
        "objects": {
            "cube": {
                "role": "target",
                "primitive": "box",
                "dimensions_m": [0.035, 0.035, 0.035],
                "mass_kg": 0.04,
                "diagonal_inertia_kg_m2": [
                    0.0000081667,
                    0.0000081667,
                    0.0000081667,
                ],
                "base_position_robot_m": [0.43, -0.11, 0.0705],
            },
            "bowl": {
                "role": "reference",
                "primitive": "open_box_compound",
                "outer_dimensions_m": [0.14, 0.14, 0.08],
                "interior_dimensions_m": [0.11, 0.11, 0.0725],
                "mass_kg": 0.28,
                "diagonal_inertia_kg_m2": [0.00055, 0.00055, 0.00082],
                "base_position_robot_m": [0.44, 0.13, 0.0605],
            },
        },
        "support_geometry": {
            "kind": "movable_open_container_on_table",
            "interior_reference_local_m": {
                "x": [-0.055, 0.055],
                "y": [-0.055, 0.055],
                "z": [0.0075, 0.08],
            },
            "wall_clearance_m": 0.005,
            "orientation": "identity_task_frame",
            "reference_motion_axis": "robot_base_y_left_right_only",
        },
    },
}


class FixtureRegistryBuildError(ValueError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FixtureRegistryBuildError(f"{path} must contain a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise FixtureRegistryBuildError(
                f"{path}:{line_number} must contain a JSON object"
            )
        result.append(value)
    return result


def fixture_env_seeds(
    *,
    fixture_id: str,
    campaign: Mapping[str, Any],
    queue_rows: Iterable[Mapping[str, Any]],
) -> tuple[int, ...]:
    build_spec = FIXTURE_BUILD_SPECS[fixture_id]
    fixture_cfg = (campaign.get("fixtures") or {}).get(fixture_id)
    seed_cfg = campaign.get("seed_reservation")
    families = campaign.get("families")
    if (
        not isinstance(fixture_cfg, Mapping)
        or not isinstance(seed_cfg, Mapping)
        or not isinstance(families, list)
    ):
        raise FixtureRegistryBuildError("campaign fixture seed allocation is missing")
    family_rows = [
        row
        for row in families
        if isinstance(row, Mapping) and row.get("id") == build_spec["family_id"]
    ]
    if len(family_rows) != 1:
        raise FixtureRegistryBuildError("campaign family allocation differs")
    block_count = family_rows[0].get("blocks")
    base = seed_cfg.get("environment_base")
    stride = seed_cfg.get("fixture_stride")
    slot = fixture_cfg.get("seed_slot")
    if any(type(value) is not int for value in (block_count, base, stride, slot)):
        raise FixtureRegistryBuildError("fixture seed allocation must be integral")
    observed: dict[int, set[int]] = {}
    for row in queue_rows:
        if row.get("fixture") != fixture_id:
            continue
        block = row.get("block_id")
        seed = row.get("env_seed")
        if type(block) is int and type(seed) is int:
            observed.setdefault(block, set()).add(seed)
    if set(observed) != set(range(block_count)) or any(
        len(values) != 1 for values in observed.values()
    ):
        raise FixtureRegistryBuildError(
            f"{fixture_id} queue lacks one environment seed per block"
        )
    seeds = tuple(next(iter(observed[index])) for index in range(block_count))
    expected = tuple(base + stride * slot + index for index in range(block_count))
    if seeds != expected:
        raise FixtureRegistryBuildError(
            f"{fixture_id} queue seeds differ from the frozen namespace"
        )
    return seeds


def deterministic_axis_jitter(
    *,
    fixture_id: str,
    env_seed: int,
    axis: str,
) -> float:
    if axis not in {"x", "y"}:
        raise FixtureRegistryBuildError("jitter axis must be x or y")
    digest = hashlib.sha256(
        f"online-correction-v4:{fixture_id}:{env_seed}:{axis}:{JITTER_ALGORITHM}".encode(
            "utf-8"
        )
    ).digest()
    unit = int.from_bytes(digest[:8], "big") / float((1 << 64) - 1)
    return (2.0 * unit - 1.0) * JITTER_HALF_RANGE_M


def validate_object_specs(objects: Mapping[str, Any]) -> None:
    if {row.get("role") for row in objects.values()} != {"target", "reference"}:
        raise FixtureRegistryBuildError("fixture roles must be target/reference")
    for name, row in objects.items():
        position = row.get("base_position_robot_m")
        inertia = row.get("diagonal_inertia_kg_m2")
        for field, values in (
            ("base_position_robot_m", position),
            ("diagonal_inertia_kg_m2", inertia),
        ):
            if (
                not isinstance(values, list)
                or len(values) != 3
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in values
                )
            ):
                raise FixtureRegistryBuildError(
                    f"{name}.{field} must be a finite 3-vector"
                )


def build_registry(
    *,
    fixture_id: str,
    campaign_path: Path,
    queue_path: Path,
    scene_path: Path | None = None,
) -> dict[str, Any]:
    if fixture_id not in FIXTURE_BUILD_SPECS:
        raise FixtureRegistryBuildError(f"unsupported fixture: {fixture_id}")
    fixture_spec = fixture_object_spec(fixture_id)
    build_spec = FIXTURE_BUILD_SPECS[fixture_id]
    objects = build_spec["objects"]
    validate_object_specs(objects)
    actual_scene_path = scene_path or (ROOT / fixture_spec.scene_asset)
    if not actual_scene_path.is_file():
        raise FixtureRegistryBuildError(f"scene asset is missing: {actual_scene_path}")
    scene_sha256 = sha256_file(actual_scene_path)
    if scene_sha256 != fixture_spec.scene_metadata_sha256:
        raise FixtureRegistryBuildError(
            f"{fixture_id} scene digest differs from the frozen code binding"
        )
    campaign = load_json(campaign_path)
    seeds = fixture_env_seeds(
        fixture_id=fixture_id,
        campaign=campaign,
        queue_rows=load_jsonl(queue_path),
    )

    def asset_identity(name: str) -> str:
        return (
            f"{portable_path(actual_scene_path)}::{name}@"
            f"{fixture_spec.scene_metadata_sha256}"
        )

    resets: dict[str, Any] = {}
    for block_index, env_seed in enumerate(seeds):
        dx = deterministic_axis_jitter(
            fixture_id=fixture_id,
            env_seed=env_seed,
            axis="x",
        )
        dy = deterministic_axis_jitter(
            fixture_id=fixture_id,
            env_seed=env_seed,
            axis="y",
        )
        resets[str(env_seed)] = {
            "block_index": block_index,
            "jitter_robot_base_xy_m": [dx, dy],
            "positions_robot_base_m": {
                name: [
                    float(row["base_position_robot_m"][0]) + dx,
                    float(row["base_position_robot_m"][1]) + dy,
                    float(row["base_position_robot_m"][2]),
                ]
                for name, row in objects.items()
            },
        }
    return {
        "schema_version": fixture_spec.reset_registry_schema,
        "campaign_id": "online_correction_v4",
        "fixture_id": fixture_id,
        "status": STATUS,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "scene_asset": portable_path(actual_scene_path),
        "scene_metadata_sha256": fixture_spec.scene_metadata_sha256,
        "scene_receipt": {
            "status": "model_blind_procedural_asset_candidate",
            "source": portable_path(actual_scene_path),
            "sha256": fixture_spec.scene_metadata_sha256,
            "object_specs": objects,
            "support_geometry": build_spec["support_geometry"],
            "policy_outcome_used": False,
        },
        "contact_objects": list(fixture_spec.contact_objects),
        "movable_objects": list(fixture_spec.movable_objects),
        "object_roles": {
            row["role"]: {
                "scene_object": name,
                "asset_identity": asset_identity(name),
            }
            for name, row in objects.items()
        },
        "source_identity": {
            "campaign_path": portable_path(campaign_path),
            "campaign_sha256": sha256_file(campaign_path),
            "queue_path": portable_path(queue_path),
            "queue_sha256": sha256_file(queue_path),
            "family_id": build_spec["family_id"],
        },
        "reset_jitter": {
            "algorithm": JITTER_ALGORITHM,
            "independent_source": "registered_environment_seed",
            "x_half_range_m": JITTER_HALF_RANGE_M,
            "y_half_range_m": JITTER_HALF_RANGE_M,
            "z_jitter_m": 0.0,
            "application": (
                "one common robot-base x/y translation is applied to the target "
                "and reference, preserving their relative geometry"
            ),
            "policy_outcome_used": False,
        },
        "registered_env_seed_count": len(seeds),
        "registered_env_seed_min": min(seeds),
        "registered_env_seed_max": max(seeds),
        "resets_by_env_seed": resets,
        "release_boundary": (
            f"Prospective model-blind {build_spec['family_id']} {fixture_id} "
            "scene/reset candidate only. G2 and G3 physical qualification must "
            "pass before policy inference."
        ),
    }


def output_path(fixture_id: str) -> Path:
    return (
        ROOT
        / "artifacts/online_correction_v4/setup"
        / f"{fixture_id}_reset_registry.candidate.json"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        choices=sorted(FIXTURE_BUILD_SPECS),
        action="append",
        dest="fixtures",
    )
    parser.add_argument(
        "--campaign",
        type=Path,
        default=ROOT / "docs/online_correction_v4/campaign.json",
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=ROOT / "artifacts/online_correction_v4/queue.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    fixtures = args.fixtures or sorted(FIXTURE_BUILD_SPECS)
    report: dict[str, Any] = {}
    for fixture_id in fixtures:
        payload = build_registry(
            fixture_id=fixture_id,
            campaign_path=args.campaign.resolve(),
            queue_path=args.queue.resolve(),
        )
        destination = (
            args.output_dir.resolve() / output_path(fixture_id).name
            if args.output_dir
            else output_path(fixture_id)
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        body = canonical_json_bytes(payload)
        destination.write_bytes(body)
        report[fixture_id] = {
            "path": str(destination.resolve()),
            "sha256": hashlib.sha256(body).hexdigest(),
            "scene_sha256": payload["scene_metadata_sha256"],
            "registered_env_seed_count": payload["registered_env_seed_count"],
            "status": payload["status"],
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
