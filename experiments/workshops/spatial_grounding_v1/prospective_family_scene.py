"""Build hash-bound prospective USD overlays for SGW HEIGHT/DIST zero-model capture.

The base RoboLab scene is sublayered unchanged.  Dimensions in this module are
design inputs, not measured geometry or qualified fixture positions; the
subsequent native capture is the sole source for geometry evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .lat_candidate_generator import workspace_digest
from .lat_workspace_capture import _rotate_wxyz

SCHEMA = "sgw-01-prospective-family-overlay-v1"
BASE_WORKSPACE_SCHEMAS = {
    "sgw-01-lat-measured-workspace-v1",
    "sgw-01-lat-measured-workspace-v2",
}
BANANA_CENTER_XY_M = (0.80, 0.39)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_overlay(
    *, family: str, base_scene: Path, workspace_receipt: Path, output: Path, upper_side: str | None = None,
    bowl_side: str | None = None,
) -> dict[str, Any]:
    """Write one source-controlled USD overlay without changing the base scene."""

    if family not in {"HEIGHT", "DIST"}:
        raise ValueError("prospective overlay supports only HEIGHT or DIST")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prospective overlay: {output}")
    if not base_scene.is_file():
        raise FileNotFoundError(f"base scene is missing: {base_scene}")
    workspace = json.loads(workspace_receipt.read_text(encoding="utf-8"))
    _validate_base_receipt(workspace, workspace_receipt)
    if family == "HEIGHT":
        side = upper_side
        if side not in {"left", "right"}:
            raise ValueError("HEIGHT overlay requires --upper-side left|right")
        specs, overrides = _height_specs(side, workspace)
        counterbalance = {"upper_support_side": side}
    else:
        side = bowl_side
        if side not in {"left", "right"}:
            raise ValueError("DIST overlay requires --bowl-side left|right")
        specs, overrides = _dist_specs(side, workspace)
        counterbalance = {"bowl_side": side}
    output.parent.mkdir(parents=True, exist_ok=True)
    banana_root = _banana_root_on_table(workspace)
    _validate_authored_banana_geometry(workspace, banana_root, specs)
    overrides["banana"] = banana_root
    orientations = {
        name: workspace["objects"][name]["root_quaternion_world_wxyz"] for name in overrides
    }
    output.write_text(_usda(base_scene.resolve(), specs, overrides, orientations), encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA,
        "status": "prospective_scene_design_not_measured_or_qualified",
        "family": family,
        "overlay_usda": {"path": str(output.resolve()), "sha256": sha256(output), "bytes": output.stat().st_size},
        "base_scene": {"path": str(base_scene.resolve()), "sha256": sha256(base_scene)},
        "base_workspace_receipt": {
            "path": str(workspace_receipt.resolve()), "sha256": sha256(workspace_receipt),
            "receipt_sha256": workspace["receipt_sha256"],
            "asset_manifest_sha256": workspace["asset_manifest_sha256"],
            "robolab_commit": workspace["robolab_commit"],
        },
        "native_import_contract": {
            "robolab_utils_sha256": "6562517740be7c60e24e964afced5eac63bd00b4c5de0b6fee92295b87c81110",
            "dynamic_bodies": ["plate"] if family == "DIST" else [],
            "kinematic_or_static_bodies": [spec["name"] for spec in specs if spec["name"] != "plate"],
            "kinematic_bodies": [spec["name"] for spec in specs if spec.get("kinematic_body")],
            "objects_of_interest": ["rubiks_cube", "bowl", "banana", "table", *(spec["name"] for spec in specs)],
            "banana_contact_bodies": ["table", *(spec["name"] for spec in specs)],
        },
        "prospective_design": {
            "units": "meters",
            "dimensions_and_poses": specs,
            "authored_actor_root_overrides_env_local_xyz_m": overrides,
            "authored_actor_quaternion_overrides_wxyz": orientations,
            "claim_boundary": "Authoring values are prospective design inputs. Native capture must measure root poses, centers, offsets, banana contacts, and rendered views before any candidate proposal.",
            "plate_category_caveat": (
                "The DIST plate is a simplified off-white disc. Its category remains a caveat until "
                "a future actual policy-sized view check; this overlay does not assert recognition."
            ) if family == "DIST" else None,
        },
        "counterbalance": counterbalance,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
    }
    manifest["manifest_sha256"] = _digest(manifest)
    return manifest


def _validate_base_receipt(value: Mapping[str, Any], path: Path) -> None:
    if value.get("measurement_schema_version") not in BASE_WORKSPACE_SCHEMAS:
        raise ValueError("prospective overlay requires a supported measured LAT workspace receipt")
    if value.get("model_request_count") != 0 or value.get("behavioral_episode_count") != 0:
        raise ValueError("base workspace receipt must remain model blind")
    objects = value.get("objects")
    if not isinstance(objects, Mapping) or not {"rubiks_cube", "bowl", "banana", "table"}.issubset(objects):
        raise ValueError("base workspace receipt lacks measured cube/bowl/banana/table")
    for name in ("rubiks_cube", "bowl", "banana", "table"):
        row = objects[name]
        if not isinstance(row, Mapping):
            raise ValueError(f"base workspace receipt lacks measured {name}")
        for key, length in (
            ("root_position_env_local_xyz_m", 3),
            ("root_quaternion_world_wxyz", 4),
            ("geometric_center_offset_root_local_xyz_m", 3),
            ("bbox_env_local_min_xyz_m", 3),
            ("bbox_env_local_max_xyz_m", 3),
        ):
            if not _finite_vector(row.get(key), length):
                raise ValueError(f"base workspace receipt lacks finite {name} {key}")
    if _table_top(workspace=value) <= value["objects"]["table"]["bbox_env_local_min_xyz_m"][2]:
        raise ValueError("base workspace receipt has invalid measured table bounds")
    for key in ("receipt_sha256", "asset_manifest_sha256", "robolab_commit", "task_asset"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError(f"base workspace receipt lacks {key}")
    if value["receipt_sha256"] != workspace_digest(value):
        raise ValueError("base workspace receipt content digest differs")
    if len(value["asset_manifest_sha256"]) != 64 or len(value["receipt_sha256"]) != 64:
        raise ValueError("base workspace receipt has invalid hash binding")
    if not path.is_file() or not sha256(path):
        raise ValueError("base workspace receipt bytes are unavailable")


def _height_specs(upper_side: str, workspace: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    """Author neutral and goal supports from measured root/center/bounds transforms."""

    sign = 1 if upper_side == "left" else -1
    cube, bowl = workspace["objects"]["rubiks_cube"], workspace["objects"]["bowl"]
    intermediate_center_z = 0.16
    bowl_center = [0.45, 0.13, intermediate_center_z]
    cube_center = [0.29, -0.10, intermediate_center_z]
    bowl_root = _root_for_center(bowl, bowl_center)
    cube_root = _root_for_center(cube, cube_center)
    bowl_bottom_z = _bottom_z(bowl, bowl_root)
    cube_bottom_z = _bottom_z(cube, cube_root)
    table_top = _table_top(workspace)
    lower_center_z = intermediate_center_z - 0.04
    upper_center_z = intermediate_center_z + 0.04
    lower_cube_root = _root_for_center(cube, [0.65, -0.23 * sign, lower_center_z])
    upper_cube_root = _root_for_center(cube, [0.65, 0.23 * sign, upper_center_z])
    specs = [
        _pedestal("height_reference_support", bowl_center[:2], bowl_bottom_z, (0.16, 0.16), (0.20, 0.45, 0.95), table_top),
        _pedestal("height_neutral_cube_support", cube_center[:2], cube_bottom_z, (0.10, 0.10), (0.35, 0.85, 0.45), table_top),
        _pedestal("height_lower_support", [0.65, -0.23 * sign], _bottom_z(cube, lower_cube_root), (0.14, 0.14), (0.20, 0.45, 0.95), table_top),
        _pedestal("height_upper_support", [0.65, 0.23 * sign], _bottom_z(cube, upper_cube_root), (0.14, 0.14), (0.95, 0.40, 0.20), table_top),
    ]
    return specs, {"bowl": bowl_root, "rubiks_cube": cube_root}


def _dist_specs(bowl_side: str, workspace: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    """Counterbalance anchors and put the cube center on their perpendicular bisector."""

    sign = 1 if bowl_side == "left" else -1
    cube, bowl = workspace["objects"]["rubiks_cube"], workspace["objects"]["bowl"]
    bowl_center = [0.44, 0.16 * sign, 0.12]
    plate_center = [0.66, -0.16 * sign, 0.12]
    cube_center = [(bowl_center[index] + plate_center[index]) / 2 for index in range(3)]
    bowl_root = _root_for_center(bowl, bowl_center)
    cube_root = _root_for_center(cube, cube_center)
    table_top = _table_top(workspace)
    return [
        _disc("plate", tuple(plate_center), radius_m=0.11, thickness_m=0.012, color=(0.94, 0.94, 0.90), rigid=True),
        _pedestal("dist_bowl_support", bowl_center[:2], _bottom_z(bowl, bowl_root), (0.18, 0.18), (0.25, 0.70, 0.35), table_top),
        _pedestal("dist_plate_support", plate_center[:2], plate_center[2] - 0.012 / 2, (0.24, 0.24), (0.95, 0.70, 0.30), table_top),
        _pedestal("dist_neutral_cube_support", cube_center[:2], _bottom_z(cube, cube_root), (0.08, 0.08), (0.35, 0.85, 0.45), table_top),
        *[
            _pedestal(
                f"dist_{name}_landing_support", center[:2], _bottom_z(cube, _root_for_center(cube, center)),
                (0.08, 0.08), color, table_top,
            )
            for name, center, color in (
                ("bowl", [0.30, 0.16 * sign, 0.12], (0.25, 0.70, 0.35)),
                ("plate", [0.66, -0.34 * sign, 0.12], (0.95, 0.70, 0.30)),
            )
        ],
    ], {"bowl": bowl_root, "rubiks_cube": cube_root}


def _box(name: str, center_m: tuple[float, float, float], size_m: tuple[float, float, float],
         color: tuple[float, float, float]) -> dict[str, Any]:
    return {
        "name": name, "center_m": list(center_m), "size_m": list(size_m),
        "display_color_rgb": list(color), "rigid_body": True, "kinematic_body": True,
    }


def _pedestal(
    name: str, center_xy_m: list[float], original_top_z_m: float, footprint_m: tuple[float, float],
    color: tuple[float, float, float], table_top_z_m: float,
) -> dict[str, Any]:
    """Extend a support down to the measured tabletop without moving its original top plane."""

    if not math.isfinite(original_top_z_m) or original_top_z_m <= table_top_z_m:
        raise ValueError(f"{name} top plane must be strictly above the measured table top")
    height = original_top_z_m - table_top_z_m
    return _box(
        name, (center_xy_m[0], center_xy_m[1], table_top_z_m + height / 2),
        (footprint_m[0], footprint_m[1], height), color,
    )


def _table_top(workspace: Mapping[str, Any]) -> float:
    table = workspace["objects"]["table"]
    maximum = table["bbox_env_local_max_xyz_m"]
    minimum = table["bbox_env_local_min_xyz_m"]
    if not _finite_vector(minimum, 3) or not _finite_vector(maximum, 3) or maximum[2] <= minimum[2]:
        raise ValueError("base workspace receipt has invalid measured table bounds")
    return maximum[2]


def _banana_root_on_table(workspace: Mapping[str, Any]) -> list[float]:
    """Move the inherited banana by measured geometry, preserving its orientation."""

    banana = workspace["objects"]["banana"]
    root = banana["root_position_env_local_xyz_m"]
    offset = banana["geometric_center_offset_root_local_xyz_m"]
    quaternion = banana["root_quaternion_world_wxyz"]
    rotated = _rotate_wxyz(quaternion, offset)
    bottom_relative_to_root = banana["bbox_env_local_min_xyz_m"][2] - root[2]
    return [
        BANANA_CENTER_XY_M[0] - rotated[0],
        BANANA_CENTER_XY_M[1] - rotated[1],
        _table_top(workspace) - bottom_relative_to_root,
    ]


def _validate_authored_banana_geometry(
    workspace: Mapping[str, Any], banana_root: list[float], specs: list[Mapping[str, Any]],
) -> None:
    """Reject an authored banana that leaves the table or nears an added support."""

    banana = workspace["objects"]["banana"]
    original_root = banana["root_position_env_local_xyz_m"]
    delta = [banana_root[index] - original_root[index] for index in range(3)]
    minimum = [banana["bbox_env_local_min_xyz_m"][index] + delta[index] for index in range(3)]
    maximum = [banana["bbox_env_local_max_xyz_m"][index] + delta[index] for index in range(3)]
    table_minimum = workspace["objects"]["table"]["bbox_env_local_min_xyz_m"]
    table_maximum = workspace["objects"]["table"]["bbox_env_local_max_xyz_m"]
    if (
        minimum[0] < table_minimum[0] or maximum[0] > table_maximum[0]
        or minimum[1] < table_minimum[1] or maximum[1] > table_maximum[1]
    ):
        raise ValueError("authored banana does not remain within measured table XY bounds")
    for spec in specs:
        if spec["name"] == "plate":
            continue
        center, size = spec["center_m"], spec["size_m"]
        support_minimum = [center[0] - size[0] / 2, center[1] - size[1] / 2]
        support_maximum = [center[0] + size[0] / 2, center[1] + size[1] / 2]
        dx = max(support_minimum[0] - maximum[0], minimum[0] - support_maximum[0], 0.0)
        dy = max(support_minimum[1] - maximum[1], minimum[1] - support_maximum[1], 0.0)
        if math.hypot(dx, dy) < 0.02:
            raise ValueError(f"authored banana clearance is below 20 mm for {spec['name']}")


def _disc(name: str, center_m: tuple[float, float, float], *, radius_m: float, thickness_m: float,
          color: tuple[float, float, float], rigid: bool) -> dict[str, Any]:
    return {
        "name": name, "center_m": list(center_m), "radius_m": radius_m, "thickness_m": thickness_m,
        "display_color_rgb": list(color), "rigid_body": rigid, "shape": "cylinder",
    }


def _finite_vector(value: Any, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(isinstance(item, (int, float)) and math.isfinite(item) for item in value)


def _root_for_center(object_row: Mapping[str, Any], center: list[float]) -> list[float]:
    offset = object_row["geometric_center_offset_root_local_xyz_m"]
    quaternion = object_row["root_quaternion_world_wxyz"]
    rotated = _rotate_wxyz(quaternion, offset)
    return [center[index] - rotated[index] for index in range(3)]


def _bottom_z(object_row: Mapping[str, Any], root: list[float]) -> float:
    """Use captured lowest geometry relative to the retained base orientation."""

    return root[2] + (
        object_row["bbox_env_local_min_xyz_m"][2] - object_row["root_position_env_local_xyz_m"][2]
    )


def _usda(
    base_scene: Path,
    specs: list[dict[str, Any]],
    overrides: Mapping[str, list[float]],
    orientations: Mapping[str, list[float]],
) -> str:
    def prim(spec: Mapping[str, Any]) -> str:
        center, color = spec["center_m"], spec["display_color_rgb"]
        rigid = '        prepend apiSchemas = ["PhysicsRigidBodyAPI"]\n' if spec["rigid_body"] else ""
        kinematic = '        bool physics:kinematicEnabled = true\n' if spec.get("kinematic_body") else ""
        if spec.get("shape") == "cylinder":
            geometry = f'''        def Cylinder "geometry" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        ) {{
            uniform token axis = "Z"
            double height = {spec["thickness_m"]}
            double radius = {spec["radius_m"]}
            color3f[] primvars:displayColor = [({color[0]}, {color[1]}, {color[2]})]
        }}'''
        else:
            size = spec["size_m"]
            geometry = f'''        def Cube "geometry" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        ) {{
            double size = 1
            double3 xformOp:scale = ({size[0]}, {size[1]}, {size[2]})
            uniform token[] xformOpOrder = ["xformOp:scale"]
            color3f[] primvars:displayColor = [({color[0]}, {color[1]}, {color[2]})]
        }}'''
        return f'''    def Xform "{spec["name"]}" (
{rigid}    )
    {{
{kinematic}        double3 xformOp:translate = ({center[0]}, {center[1]}, {center[2]})
        uniform token[] xformOpOrder = ["xformOp:translate"]
{geometry}
    }}'''
    escaped = str(base_scene).replace("\\", "\\\\")
    authored = "\n".join(
        f'    over "{name}" {{\n'
        f'        double3 xformOp:translate = ({pose[0]}, {pose[1]}, {pose[2]})\n'
        f'        quatf xformOp:orient = ({orientations[name][0]}, {orientations[name][1]}, '
        f'{orientations[name][2]}, {orientations[name][3]})\n'
        '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]\n'
        '    }'
        for name, pose in overrides.items()
    )
    return "#usda 1.0\n(\n    defaultPrim = \"World\"\n    subLayers = [@" + escaped + "@]\n)\n\nover \"World\" {\n" + authored + "\n" + "\n".join(prim(item) for item in specs) + "\n}\n"


def _digest(value: Mapping[str, Any]) -> str:
    material = dict(value)
    material.pop("manifest_sha256", None)
    return hashlib.sha256((json.dumps(material, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--family", choices=("HEIGHT", "DIST"), required=True)
    parser.add_argument("--base-scene", type=Path, required=True)
    parser.add_argument("--base-workspace-receipt", type=Path, required=True)
    parser.add_argument("--output-usda", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--upper-side", choices=("left", "right"))
    parser.add_argument("--bowl-side", choices=("left", "right"))
    args = parser.parse_args()
    if args.manifest_output.exists():
        raise FileExistsError(f"refusing to overwrite overlay manifest: {args.manifest_output}")
    manifest = build_overlay(
        family=args.family, base_scene=args.base_scene, workspace_receipt=args.base_workspace_receipt,
        output=args.output_usda, upper_side=args.upper_side, bowl_side=args.bowl_side,
    )
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
