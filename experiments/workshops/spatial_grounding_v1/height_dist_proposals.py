"""Fail-closed HEIGHT/DIST candidate materialization from measured layouts.

Unlike the early LAT geometry sampler, these families require geometry that is
not present in the measured LAT receipt: HEIGHT needs two released landing
surfaces and DIST needs a measured plate plus two released landing regions.
This module never estimates those missing objects from a current USD.
"""

from __future__ import annotations

import hashlib
import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .fixtures import FixtureCandidate


def propose_family_layouts(workspace: Mapping[str, Any], *, family: str, seed: int, count: int = 100) -> list[dict[str, Any]]:
    """Materialize bounded, unqualified candidates from measurement-supplied rows."""

    if family not in {"HEIGHT", "DIST"}:
        raise ValueError("only HEIGHT and DIST use this measured-layout proposal surface")
    if not 1 <= count <= 100:
        raise ValueError("proposal count must be within the frozen 1..100 cap")
    _require_common_workspace(workspace)
    source_rows = workspace.get(f"{family.lower()}_layout_measurements")
    if not isinstance(source_rows, list):
        raise ValueError(_missing_capture_requirement(family))
    layout_ids = [row.get("layout_id") for row in source_rows if isinstance(row, Mapping)]
    if len(layout_ids) != len(source_rows) or any(not isinstance(layout_id, str) or not layout_id for layout_id in layout_ids):
        raise ValueError(f"{family} capture has a missing layout_id")
    if len(set(layout_ids)) != len(layout_ids):
        raise ValueError(f"{family} capture has duplicate layout IDs")
    candidates = [_candidate_from_measurement(row, workspace, family=family, seed=seed) for row in source_rows]
    if len(candidates) < count:
        raise ValueError(f"{family} capture contains {len(candidates)} usable measured layouts; {count} requested")
    ordered = sorted(candidates, key=lambda row: hashlib.sha256(
        f"{seed}|{family}|{row['candidate_id']}".encode()
    ).hexdigest())
    return ordered[:count]


def _candidate_from_measurement(row: Any, workspace: Mapping[str, Any], *, family: str, seed: int) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise ValueError(f"{family} measured layout is not an object")
    required = {"layout_id", "object_root_poses", "scoring_center_offsets_root_local_m", "native_scene"}
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"{family} measured layout lacks {', '.join(missing)}")
    poses = row["object_root_poses"]
    offsets = row["scoring_center_offsets_root_local_m"]
    names = {"rubiks_cube", "bowl"} | ({"plate"} if family == "DIST" else set())
    if not isinstance(poses, Mapping) or set(poses) != names or not isinstance(offsets, Mapping) or set(offsets) != names:
        raise ValueError(f"{family} measured layout has an incomplete object inventory")
    _validate_native_scene(row["native_scene"], names)
    if family == "HEIGHT":
        _validate_height_measurements(row)
        counterbalance_key = "upper_support_side"
    else:
        _validate_dist_measurements(row)
        counterbalance_key = "bowl_side"
    candidate = {
        "candidate_id": f"{family}-CANDIDATE-{str(row['layout_id'])}",
        "family": family,
        "seed": seed,
        "task_asset": str(workspace["task_asset"]),
        "asset_manifest_sha256": str(workspace["asset_manifest_sha256"]),
        "object_poses": poses,
        "metadata": {
            "status": "unqualified_proposal_starting_physical_validation",
            "workspace_receipt_sha256": str(workspace["receipt_sha256"]),
            "source_measurement_layout_id": str(row["layout_id"]),
            "scoring_center_offsets_root_local_m": offsets,
            "native_scene": row["native_scene"],
            counterbalance_key: row[counterbalance_key],
            "goal_supports": row["goal_supports"],
            "controller_status": "requires_measured_abs_ik_calibration_before_qualification",
            "geometric_screen_status": "passed",
            "historical_layout_fingerprint": None,
        },
    }
    # This enforces the frozen neutral relation using measured center offsets.
    parsed = FixtureCandidate.from_json(candidate)
    declared_centers = row.get("scoring_centers_env_local_xyz_m")
    if not isinstance(declared_centers, Mapping):
        raise ValueError(f"{family} measured layout lacks scoring centers")
    for name, pose in parsed.scoring_poses().items():
        center = declared_centers.get(name)
        _finite_vector(center, f"{name} measured scoring center")
        if max(abs(observed - expected) for observed, expected in zip(center, pose.position_m, strict=True)) > 1e-6:
            raise ValueError(f"{family} {name} scoring center does not close from root pose and offset")
    return candidate


def _require_common_workspace(workspace: Mapping[str, Any]) -> None:
    for key in ("receipt_sha256", "asset_manifest_sha256", "task_asset"):
        if not isinstance(workspace.get(key), str) or not workspace[key]:
            raise ValueError(f"workspace lacks required measured {key}")


def _validate_native_scene(scene: Any, object_names: set[str]) -> None:
    if not isinstance(scene, Mapping):
        raise ValueError("measured native_scene is required")
    asset = scene.get("asset")
    names = scene.get("object_names")
    if not isinstance(asset, str) or not asset or not isinstance(names, list):
        raise ValueError("native_scene requires measured asset and object_names")
    if not object_names.issubset(set(names)) or "table" not in names:
        raise ValueError("native_scene omits a scored object or table")


def _validate_height_measurements(row: Mapping[str, Any]) -> None:
    if row.get("upper_support_side") not in {"left", "right"}:
        raise ValueError("HEIGHT requires measured upper_support_side")
    supports = row.get("goal_supports")
    if not isinstance(supports, Mapping) or set(supports) != {"higher", "lower"}:
        raise ValueError("HEIGHT requires measured higher and lower landing supports")
    for sign, support in supports.items():
        if (not isinstance(support, Mapping) or not support.get("support_surface_id")
                or not support.get("contact_sensor_id")):
            raise ValueError(f"HEIGHT {sign} support lacks an immutable surface identity")
        _finite_vector(support.get("cube_center_env_local_xyz_m"), f"HEIGHT {sign} support center")
    cube, bowl = _centers(row)
    if abs(cube[2] - bowl[2]) > 0.005:
        raise ValueError("HEIGHT measured start is not neutral")
    higher = supports["higher"]["cube_center_env_local_xyz_m"][2] - bowl[2]
    lower = supports["lower"]["cube_center_env_local_xyz_m"][2] - bowl[2]
    if higher < 0.03 or lower > -0.03:
        raise ValueError("HEIGHT supports do not expose both released 30 mm goals")


def _validate_dist_measurements(row: Mapping[str, Any]) -> None:
    if row.get("bowl_side") not in {"left", "right"}:
        raise ValueError("DIST requires measured bowl_side")
    supports = row.get("goal_supports")
    if not isinstance(supports, Mapping) or set(supports) != {"near_bowl", "near_plate"}:
        raise ValueError("DIST requires measured near-bowl and near-plate landing supports")
    for name, support in supports.items():
        if (not isinstance(support, Mapping) or not support.get("support_surface_id")
                or not support.get("contact_sensor_id")):
            raise ValueError(f"DIST {name} support lacks an immutable surface identity")
        _finite_vector(support.get("cube_center_env_local_xyz_m"), f"DIST {name} support center")
    cube, bowl, plate = _centers(row, include_plate=True)
    if abs(_distance(cube, bowl) - _distance(cube, plate)) > 0.005:
        raise ValueError("DIST measured start is not on the anchors' perpendicular bisector")
    near_bowl = supports["near_bowl"]["cube_center_env_local_xyz_m"]
    near_plate = supports["near_plate"]["cube_center_env_local_xyz_m"]
    if _distance(near_bowl, plate) - _distance(near_bowl, bowl) < 0.03:
        raise ValueError("DIST near-bowl support does not expose the requested margin")
    if _distance(near_plate, plate) - _distance(near_plate, bowl) > -0.03:
        raise ValueError("DIST near-plate support does not expose the requested margin")


def _centers(row: Mapping[str, Any], *, include_plate: bool = False) -> tuple[list[float], ...]:
    offsets = row["scoring_center_offsets_root_local_m"]
    poses = row["object_root_poses"]
    result = []
    for name in ("rubiks_cube", "bowl") + (("plate",) if include_plate else ()):
        # We intentionally require capture to supply center coordinates too: deriving
        # them here would conflate a future asset revision with the historical capture.
        center = row.get("scoring_centers_env_local_xyz_m", {}).get(name)
        _finite_vector(center, f"{name} measured scoring center")
        _finite_vector(offsets[name], f"{name} center offset")
        result.append(center)
    return tuple(result)


def _finite_vector(value: Any, label: str) -> None:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must be a finite 3-vector")
    if not all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value):
        raise ValueError(f"{label} must be a finite 3-vector")


def _distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right, strict=True)))


def _missing_capture_requirement(family: str) -> str:
    if family == "HEIGHT":
        return (
            "HEIGHT requires a measured height_layout_measurements capture with cube/bowl roots and center offsets, "
            "intermediate bowl center, immutable higher/lower support identities and landing centers, "
            "upper_support_side, and a native scene asset/object inventory."
        )
    return (
        "DIST requires a measured dist_layout_measurements capture with cube/bowl/plate roots and center offsets, "
        "all three scoring centers, immutable near-bowl/near-plate support identities and landing centers, "
        "bowl_side, and a native scene asset/object inventory."
    )


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--workspace-receipt", type=Path, required=True)
    parser.add_argument("--family", choices=("HEIGHT", "DIST"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite proposal artifact: {args.output}")
    workspace = json.loads(args.workspace_receipt.read_text(encoding="utf-8"))
    candidates = propose_family_layouts(workspace, family=args.family, seed=args.seed, count=args.count)
    value = {
        "schema_version": "sgw-01-measured-family-proposals-v1",
        "family": args.family,
        "status": "proposed_unqualified",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "workspace_receipt_sha256": workspace["receipt_sha256"],
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
