"""Build hash-bound prospective USD overlays for SGW HEIGHT/DIST zero-model capture.

The base RoboLab scene is sublayered unchanged.  Dimensions in this module are
design inputs, not measured geometry or qualified fixture positions; the
subsequent native capture is the sole source for geometry evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .lat_candidate_generator import workspace_digest

SCHEMA = "sgw-01-prospective-family-overlay-v1"
BASE_WORKSPACE_SCHEMA = "sgw-01-lat-measured-workspace-v2"


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
    output.write_text(_usda(base_scene.resolve(), specs, overrides), encoding="utf-8")
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
            "objects_of_interest": ["rubiks_cube", "bowl", "table", *(spec["name"] for spec in specs)],
        },
        "prospective_design": {
            "units": "meters",
            "dimensions_and_poses": specs,
            "claim_boundary": "Authoring values are prospective design inputs. Native capture must measure root poses, centers, offsets, contacts, and rendered views before any candidate proposal.",
        },
        "counterbalance": counterbalance,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
    }
    manifest["manifest_sha256"] = _digest(manifest)
    return manifest


def _validate_base_receipt(value: Mapping[str, Any], path: Path) -> None:
    if value.get("measurement_schema_version") != BASE_WORKSPACE_SCHEMA:
        raise ValueError("prospective overlay requires the measured LAT workspace-v2 receipt")
    if value.get("model_request_count") != 0 or value.get("behavioral_episode_count") != 0:
        raise ValueError("base workspace receipt must remain model blind")
    objects = value.get("objects")
    if not isinstance(objects, Mapping) or not {"rubiks_cube", "bowl", "table"}.issubset(objects):
        raise ValueError("base workspace receipt lacks measured cube/bowl/table")
    for name in ("rubiks_cube", "bowl"):
        row = objects[name]
        if not isinstance(row, Mapping) or not isinstance(row.get("geometric_center_offset_root_local_xyz_m"), list):
            raise ValueError(f"base workspace receipt lacks measured {name} center offset")
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
    """Place bowl/cube at a measured-offset neutral intermediate height."""

    sign = 1 if upper_side == "left" else -1
    cube_offset = workspace["objects"]["rubiks_cube"]["geometric_center_offset_root_local_xyz_m"]
    bowl = workspace["objects"]["bowl"]
    bowl_root = [0.44258353114128113, 0.0, 0.15]
    cube_root = [0.32 - cube_offset[0], -0.10 - cube_offset[1], bowl_root[2] - cube_offset[2]]
    specs = [
        _box("height_reference_support", (bowl_root[0], bowl_root[1], 0.10), (0.16, 0.16, 0.10), (0.20, 0.45, 0.95)),
        _box("height_lower_support", (0.62, -0.23 * sign, 0.075), (0.14, 0.14, 0.05), (0.20, 0.45, 0.95)),
        _box("height_upper_support", (0.62, 0.23 * sign, 0.22), (0.14, 0.14, 0.34), (0.95, 0.40, 0.20)),
    ]
    return specs, {"bowl": bowl_root, "rubiks_cube": cube_root}


def _dist_specs(bowl_side: str, workspace: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    """Counterbalance anchors and put the cube center on their perpendicular bisector."""

    sign = 1 if bowl_side == "left" else -1
    cube_offset = workspace["objects"]["rubiks_cube"]["geometric_center_offset_root_local_xyz_m"]
    bowl_root = [0.44, 0.16 * sign, 0.07732785493135452]
    plate_center = [0.62, -0.16 * sign, 0.072]
    cube_center = [(bowl_root[0] + plate_center[0]) / 2, 0.0, bowl_root[2]]
    cube_root = [cube_center[index] - cube_offset[index] for index in range(3)]
    return [
        _box("plate", tuple(plate_center), (0.16, 0.16, 0.015), (0.95, 0.75, 0.15), rigid=True),
        _box("dist_bowl_landing_support", (0.48, 0.18 * sign, 0.055), (0.12, 0.12, 0.01), (0.25, 0.70, 0.35)),
        _box("dist_plate_landing_support", (0.62, 0.18 * -sign, 0.055), (0.12, 0.12, 0.01), (0.95, 0.70, 0.30)),
    ], {"bowl": bowl_root, "rubiks_cube": cube_root}


def _box(name: str, center_m: tuple[float, float, float], size_m: tuple[float, float, float],
         color: tuple[float, float, float], *, rigid: bool = False) -> dict[str, Any]:
    return {
        "name": name, "center_m": list(center_m), "size_m": list(size_m),
        "display_color_rgb": list(color), "rigid_body": rigid,
    }


def _usda(base_scene: Path, specs: list[dict[str, Any]], overrides: Mapping[str, list[float]]) -> str:
    def prim(spec: Mapping[str, Any]) -> str:
        center, size, color = spec["center_m"], spec["size_m"], spec["display_color_rgb"]
        rigid = '        prepend apiSchemas = ["PhysicsRigidBodyAPI"]\n' if spec["rigid_body"] else ""
        return f'''    def Xform "{spec["name"]}" (
{rigid}    )
    {{
        double3 xformOp:translate = ({center[0]}, {center[1]}, {center[2]})
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Cube "geometry" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        ) {{
            double size = 1
            double3 xformOp:scale = ({size[0]}, {size[1]}, {size[2]})
            uniform token[] xformOpOrder = ["xformOp:scale"]
            color3f[] primvars:displayColor = [({color[0]}, {color[1]}, {color[2]})]
        }}
    }}'''
    escaped = str(base_scene).replace("\\", "\\\\")
    authored = "\n".join(
        f'    over "{name}" {{\n        double3 xformOp:translate = ({pose[0]}, {pose[1]}, {pose[2]})\n    }}'
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
