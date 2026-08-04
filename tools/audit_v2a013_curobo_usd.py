#!/usr/bin/env python3
"""Read-only V2-A013 USD/CuRobo kinematics audit.

This intentionally does not import Isaac Sim or CuRobo, initialize CUDA, write
files, or execute a robot action.  It uses usd-core/PXR to compare the exact
USD frame transform with the pinned CuRobo USD parser's documented joint-only
transform calculation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path


PARSER_RELATIVE_PATH = Path("src/curobo/cuda_robot_model/usd_kinematics_parser.py")
EXPECTED_ASSET_SHA256 = "f555695465687548a1bd31b5e3f30385182d476a67c17080b7820ad0ef747e41"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(path: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def matrix_rows(matrix) -> list[list[float]]:
    return [[float(matrix[row][column]) for column in range(4)] for row in range(4)]


def make_transform(gf, position, quaternion):
    matrix = gf.Matrix4d(1)
    matrix.SetRotate(gf.Quatd(quaternion.GetReal(), gf.Vec3d(*quaternion.GetImaginary())))
    matrix.SetTranslate(gf.Vec3d(*position))
    return matrix


def joint_only_transform(gf, usd_physics, stage, joint_paths: list[str]):
    """Reproduce `UsdKinematicsParser._get_joint_transform` across a chain."""
    result = gf.Matrix4d(1)
    for joint_path in joint_paths:
        joint = usd_physics.Joint(stage.GetPrimAtPath(joint_path))
        local0 = make_transform(gf, joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
        local1 = make_transform(gf, joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())
        result = result * local0 * local1.GetInverse()
    return result


def joint_chain(usd_physics, stage, base_path: str, ee_path: str) -> list[str]:
    """Return parent-to-child joint prims connecting the two explicit USD paths."""
    parent_joint_by_child: dict[str, tuple[str, str]] = {}
    for prim in stage.Traverse():
        if not prim.IsA(usd_physics.Joint):
            continue
        joint = usd_physics.Joint(prim)
        parents = joint.GetBody0Rel().GetTargets()
        children = joint.GetBody1Rel().GetTargets()
        if len(parents) == 1 and len(children) == 1:
            parent_joint_by_child[str(children[0])] = (str(parents[0]), str(prim.GetPath()))

    current = ee_path
    reverse_chain: list[str] = []
    while current != base_path:
        if current not in parent_joint_by_child:
            raise ValueError(f"no unique parent joint from {current} toward {base_path}")
        current, joint_path = parent_joint_by_child[current]
        reverse_chain.append(joint_path)
    return list(reversed(reverse_chain))


def joint_summary(usd_physics, stage, joint_path: str) -> dict[str, object]:
    prim = stage.GetPrimAtPath(joint_path)
    joint = usd_physics.Joint(prim)
    record: dict[str, object] = {
        "name": prim.GetName(),
        "path": joint_path,
        "type": prim.GetTypeName(),
        "body0": [str(target) for target in joint.GetBody0Rel().GetTargets()],
        "body1": [str(target) for target in joint.GetBody1Rel().GetTargets()],
        "local_pos0": [float(value) for value in joint.GetLocalPos0Attr().Get()],
        "local_pos1": [float(value) for value in joint.GetLocalPos1Attr().Get()],
    }
    if prim.IsA(usd_physics.RevoluteJoint):
        revolute = usd_physics.RevoluteJoint(prim)
        record["axis"] = str(revolute.GetAxisAttr().Get())
        record["limits_degrees"] = [
            float(revolute.GetLowerLimitAttr().Get()),
            float(revolute.GetUpperLimitAttr().Get()),
        ]
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", required=True, type=Path, help="Hydrated RoboLab USD; read only.")
    parser.add_argument("--curobo-root", required=True, type=Path, help="Pinned CuRobo source root; read only.")
    parser.add_argument("--robot-root", default="/panda")
    parser.add_argument("--base-path", default="/panda/panda_link0")
    parser.add_argument("--ee-path", default="/panda/Gripper/Robotiq_2F_85/base_link")
    args = parser.parse_args()

    if not args.usd.is_file():
        parser.error(f"USD does not exist: {args.usd}")
    parser_source = args.curobo_root / PARSER_RELATIVE_PATH
    if not parser_source.is_file():
        parser.error(f"pinned parser does not exist: {parser_source}")
    try:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
    except ImportError as error:
        parser.error(f"usd-core/PXR is required for static inspection: {error}")

    stage = Usd.Stage.Open(str(args.usd))
    if stage is None:
        parser.error(f"USD stage could not be opened: {args.usd}")
    if not stage.GetPrimAtPath(args.robot_root).IsValid():
        parser.error(f"robot root is not valid: {args.robot_root}")
    if not stage.GetPrimAtPath(args.base_path).IsValid() or not stage.GetPrimAtPath(args.ee_path).IsValid():
        parser.error("base or end-effector prim is not valid")

    chain = joint_chain(UsdPhysics, stage, args.base_path, args.ee_path)
    xform_cache = UsdGeom.XformCache()
    base_world = xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(args.base_path))
    ee_world = xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(args.ee_path))
    usd_transform = base_world.GetInverse() * ee_world
    parser_transform = joint_only_transform(Gf, UsdPhysics, stage, chain)
    delta = parser_transform * usd_transform.GetInverse()
    translation_error = math.sqrt(sum(float(delta[3][index]) ** 2 for index in range(3)))
    rotation_trace = sum(float(delta[index][index]) for index in range(3))
    rotation_error_degrees = math.degrees(math.acos(max(-1.0, min(1.0, (rotation_trace - 1.0) / 2.0))))

    all_joints = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)]
    collision_prims = [
        {"path": str(prim.GetPath()), "type": prim.GetTypeName()}
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    report = {
        "audit": "cosmos3_edge_base_v2a013_curobo_usd_static",
        "read_only": True,
        "isaac_initialized": False,
        "cuda_initialized": False,
        "usd": {
            "path": str(args.usd),
            "bytes": args.usd.stat().st_size,
            "sha256": sha256(args.usd),
            "expected_v2a013_sha256": EXPECTED_ASSET_SHA256,
            "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
            "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
            "default_prim": str(stage.GetDefaultPrim().GetPath()),
            "robot_root": args.robot_root,
        },
        "curobo": {
            "source_root": str(args.curobo_root),
            "git_head": git_head(args.curobo_root),
            "usd_parser": str(parser_source),
            "usd_parser_sha256": sha256(parser_source),
            "known_limitations": [
                "experimental USD parser",
                "does not account for link geometry transformations after joints",
                "cannot read mimic joints",
                "does not create collision spheres or link meshes",
            ],
        },
        "frame_comparison": {
            "base_path": args.base_path,
            "ee_path": args.ee_path,
            "joint_chain_parent_to_child": chain,
            "usd_xform_cache_base_to_ee_row_major": matrix_rows(usd_transform),
            "curobo_joint_only_reproduction_row_major": matrix_rows(parser_transform),
            "joint_only_minus_usd_delta_row_major": matrix_rows(delta),
            "translation_error_m": translation_error,
            "rotation_error_degrees": rotation_error_degrees,
        },
        "joint_facts": [joint_summary(UsdPhysics, stage, str(prim.GetPath())) for prim in all_joints],
        "collision_prims": collision_prims,
        "decision": "blocked_no_behavior_direct_usd_parser_frame_mismatch_and_no_collision_model",
    }
    json.dump(report, sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
