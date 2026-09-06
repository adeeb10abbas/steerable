#!/usr/bin/env python3
"""Build the prospective V4 C7 sponge/tray reset registry without inference."""

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
DEFAULT_SCENE = (
    ROOT
    / "experiments/online_correction_v4/droid_task_files/scene_assets/"
    "sponge_tray_object_pair.usda"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts/online_correction_v4/setup/object_pair_reset_registry.candidate.json"
)

SCHEMA_VERSION = "v4-droid-object-pair-reset-registry-v1"
STATUS = "model_blind_candidate_not_released_for_inference"
FIXTURE_ID = "object_pair"
SEED_GROUP = "object_pair"
JITTER_ALGORITHM = "sha256_u64_independent_axes_common_scene_translation_v1"
JITTER_HALF_RANGE_X_M = 0.015
JITTER_HALF_RANGE_Y_M = 0.015

OBJECT_SPECS: dict[str, dict[str, Any]] = {
    "sponge": {
        "role": "target",
        "construction": "co_composed_usd_rigid_body",
        "primitive": "box",
        "dimensions_m": [0.08, 0.055, 0.035],
        "mass_kg": 0.045,
        "center_of_mass_m": [0.0, 0.0, 0.0],
        "principal_axes_wxyz": [1.0, 0.0, 0.0, 0.0],
        "diagonal_inertia_kg_m2": [0.0000159375, 0.00002859375, 0.00003534375],
        "linear_damping": 8.0,
        "angular_damping": 4.0,
        "display_color_rgb": [0.95, 0.78, 0.10],
        "base_position_robot_m": [0.43, -0.10, 0.0705],
    },
    "tray": {
        "role": "reference",
        "construction": "co_composed_usd_rigid_body",
        "primitive": "box",
        "dimensions_m": [0.14, 0.10, 0.018],
        "mass_kg": 0.30,
        "center_of_mass_m": [0.0, 0.0, 0.0],
        "principal_axes_wxyz": [1.0, 0.0, 0.0, 0.0],
        "diagonal_inertia_kg_m2": [0.0002581, 0.0004981, 0.00074],
        "linear_damping": 8.0,
        "angular_damping": 4.0,
        "display_color_rgb": [0.10, 0.34, 0.85],
        "base_position_robot_m": [0.44, 0.13, 0.062],
    },
}


class ObjectPairRegistryBuildError(ValueError):
    """Raised when prospective C7 reset inputs are missing or inconsistent."""


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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ObjectPairRegistryBuildError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ObjectPairRegistryBuildError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ObjectPairRegistryBuildError(f"cannot read queue {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ObjectPairRegistryBuildError(
                f"invalid queue JSON at {path}:{line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise ObjectPairRegistryBuildError(
                f"queue row at {path}:{line_number} must be an object"
            )
        rows.append(value)
    return rows


def object_pair_env_seeds(
    *,
    campaign: Mapping[str, Any],
    queue_rows: Iterable[Mapping[str, Any]],
) -> tuple[int, ...]:
    seed_cfg = campaign.get("seed_reservation")
    fixture_cfg = (campaign.get("fixtures") or {}).get(FIXTURE_ID)
    if not isinstance(seed_cfg, Mapping) or not isinstance(fixture_cfg, Mapping):
        raise ObjectPairRegistryBuildError("campaign lacks object-pair seed reservation")
    base = seed_cfg.get("environment_base")
    stride = seed_cfg.get("fixture_stride")
    slot = fixture_cfg.get("seed_slot")
    if type(base) is not int or type(stride) is not int or type(slot) is not int:
        raise ObjectPairRegistryBuildError("object-pair seed parameters must be integers")

    families = campaign.get("families")
    if not isinstance(families, list):
        raise ObjectPairRegistryBuildError("campaign families must be a list")
    counts = [
        row.get("blocks")
        for row in families
        if isinstance(row, Mapping) and row.get("seed_group") == SEED_GROUP
    ]
    if len(counts) != 1 or type(counts[0]) is not int or counts[0] <= 0:
        raise ObjectPairRegistryBuildError("campaign lacks one valid C7 block count")
    block_count = counts[0]

    observed: dict[int, set[int]] = {}
    for row in queue_rows:
        block = row.get("block_id")
        seed = row.get("env_seed")
        if (
            row.get("fixture") == FIXTURE_ID
            and type(block) is int
            and type(seed) is int
        ):
            observed.setdefault(block, set()).add(seed)
    if set(observed) != set(range(block_count)) or any(
        len(values) != 1 for values in observed.values()
    ):
        raise ObjectPairRegistryBuildError(
            "object-pair queue does not bind exactly one environment seed per block"
        )
    seeds = tuple(next(iter(observed[block])) for block in range(block_count))
    expected = tuple(base + stride * slot + block for block in range(block_count))
    if seeds != expected:
        raise ObjectPairRegistryBuildError(
            "object-pair queue seeds differ from the frozen fixture namespace"
        )
    return seeds


def deterministic_axis_jitter(*, env_seed: int, axis: str) -> float:
    if axis not in {"x", "y"}:
        raise ObjectPairRegistryBuildError(f"unsupported jitter axis: {axis}")
    digest = hashlib.sha256(
        f"online-correction-v4:{FIXTURE_ID}:{env_seed}:{axis}:{JITTER_ALGORITHM}".encode(
            "utf-8"
        )
    ).digest()
    unit = int.from_bytes(digest[:8], "big") / float((1 << 64) - 1)
    half_range = (
        JITTER_HALF_RANGE_X_M if axis == "x" else JITTER_HALF_RANGE_Y_M
    )
    return (2.0 * unit - 1.0) * half_range


def _validate_object_specs() -> None:
    if {row["role"] for row in OBJECT_SPECS.values()} != {"target", "reference"}:
        raise ObjectPairRegistryBuildError("object-pair roles must be target/reference")
    for name, row in OBJECT_SPECS.items():
        for field in (
            "dimensions_m",
            "base_position_robot_m",
            "center_of_mass_m",
            "diagonal_inertia_kg_m2",
        ):
            values = row.get(field)
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
                raise ObjectPairRegistryBuildError(
                    f"object specification {name}.{field} must be a finite 3-vector"
                )
        axes = row.get("principal_axes_wxyz")
        if (
            not isinstance(axes, list)
            or len(axes) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in axes
            )
        ):
            raise ObjectPairRegistryBuildError(
                f"object specification {name}.principal_axes_wxyz must be a finite quaternion"
            )


def build_registry(
    *,
    campaign_path: Path,
    queue_path: Path,
    scene_path: Path,
) -> dict[str, Any]:
    campaign = _read_json(campaign_path)
    if campaign.get("campaign_id") != "online_correction_v4":
        raise ObjectPairRegistryBuildError("campaign identity differs")
    if not scene_path.is_file():
        raise ObjectPairRegistryBuildError(f"scene asset does not exist: {scene_path}")
    _validate_object_specs()
    seeds = object_pair_env_seeds(
        campaign=campaign,
        queue_rows=_read_jsonl(queue_path),
    )
    scene_sha256 = sha256_file(scene_path)

    def asset_identity(name: str) -> str:
        return f"{portable_path(scene_path)}::{name}@{scene_sha256}"

    resets: dict[str, Any] = {}
    for block_index, env_seed in enumerate(seeds):
        dx = deterministic_axis_jitter(env_seed=env_seed, axis="x")
        dy = deterministic_axis_jitter(env_seed=env_seed, axis="y")
        resets[str(env_seed)] = {
            "block_index": block_index,
            "jitter_robot_base_xy_m": [dx, dy],
            "positions_robot_base_m": {
                name: [
                    float(row["base_position_robot_m"][0]) + dx,
                    float(row["base_position_robot_m"][1]) + dy,
                    float(row["base_position_robot_m"][2]),
                ]
                for name, row in OBJECT_SPECS.items()
            },
        }

    roles = {
        row["role"]: {
            "scene_object": name,
            "asset_identity": asset_identity(name),
        }
        for name, row in OBJECT_SPECS.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": "online_correction_v4",
        "fixture_id": FIXTURE_ID,
        "status": STATUS,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "scene_asset": portable_path(scene_path),
        "scene_metadata_sha256": scene_sha256,
        "scene_receipt": {
            "status": "model_blind_procedural_asset_candidate",
            "source": portable_path(scene_path),
            "sha256": scene_sha256,
            "external_payload_root": (
                "/data/users/ali/vla_wam/external/RoboLab-11142d4"
            ),
            "object_specs": OBJECT_SPECS,
            "support_surface": {
                "construction": "qualified_dynamic_table_payload",
                "prim": "table",
                "top_surface_z_m": 0.05,
                "initial_clearance_m": 0.003,
            },
            "policy_outcome_used": False,
        },
        "contact_objects": ["sponge", "tray", "table"],
        "movable_objects": ["sponge", "tray"],
        "object_roles": roles,
        "source_identity": {
            "campaign_path": portable_path(campaign_path),
            "campaign_sha256": sha256_file(campaign_path),
            "queue_path": portable_path(queue_path),
            "queue_sha256": sha256_file(queue_path),
            "base_positions_robot_base_m": {
                name: row["base_position_robot_m"]
                for name, row in OBJECT_SPECS.items()
            },
        },
        "reset_jitter": {
            "algorithm": JITTER_ALGORITHM,
            "independent_source": "registered_environment_seed",
            "x_half_range_m": JITTER_HALF_RANGE_X_M,
            "y_half_range_m": JITTER_HALF_RANGE_Y_M,
            "z_jitter_m": 0.0,
            "application": (
                "one common robot-base x/y translation is applied to both movable "
                "objects, preserving their relative geometry"
            ),
            "policy_outcome_used": False,
        },
        "registered_env_seed_count": len(seeds),
        "registered_env_seed_min": min(seeds),
        "registered_env_seed_max": max(seeds),
        "resets_by_env_seed": resets,
        "release_boundary": (
            "Prospective model-blind C7 scene/reset candidate only. USD import, "
            "live repeated-reset, support/contact, camera/frame, swept-path, and "
            "scripted-controller gates must pass before policy inference."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    payload = build_registry(
        campaign_path=args.campaign.resolve(),
        queue_path=args.queue.resolve(),
        scene_path=args.scene.resolve(),
    )
    body = canonical_json_bytes(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(body)
    print(
        json.dumps(
            {
                "path": str(args.out.resolve()),
                "sha256": hashlib.sha256(body).hexdigest(),
                "scene_sha256": payload["scene_metadata_sha256"],
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
