"""Compile captured live geometry into a fail-closed V3-E004 scene gate.

The simulator adapter writes one finite JSON snapshot containing settled
object poses, arm reset pose, camera centers, the live bowl bounding box, and
rendered instance-segmentation counts.  This compiler performs no inference
and is safe to run independently of Isaac.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .layout_contract import LayoutContractError, PoseSE2, evaluate_layout, load_candidate
from .occlusion import (
    CameraEvidence,
    YawOrientedBox,
    evaluate_all_cameras,
    project_world_target_to_pixel,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_finite(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )


def compile_live_gate(
    *,
    candidate_path: Path,
    candidate_sha256: str,
    snapshot_path: Path,
    snapshot_sha256: str,
    minimum_visible_target_pixels: int,
) -> dict[str, Any]:
    candidate = load_candidate(candidate_path, candidate_sha256)
    if _sha(snapshot_path) != snapshot_sha256:
        raise LayoutContractError("live snapshot SHA-256 mismatch")
    snapshot = _load_finite(snapshot_path)
    if snapshot.get("schema_version") != "vla-wam-shared-v3e004-live-scene-snapshot-v1":
        raise LayoutContractError("live snapshot schema changed")
    if snapshot.get("model_request_count") != 0 or snapshot.get("behavioral_episode_count") != 0:
        raise LayoutContractError("live scene gate is not model-blind")
    poses = {
        name: PoseSE2.from_json(value, f"realised_object_poses.{name}")
        for name, value in snapshot.get("realised_object_poses", {}).items()
    }
    camera_rows: Mapping[str, Mapping[str, Any]] = snapshot.get("cameras", {})
    evidence: dict[str, CameraEvidence] = {}
    for name, row in camera_rows.items():
        bounds = row.get("reference_bounds_world", {})
        geometry_payload = {
            "camera_center_world_m": row.get("camera_center_world_m"),
            "camera_quaternion_world_wxyz_ros": row.get("camera_quaternion_world_wxyz_ros"),
            "target_center_world_m": row.get("target_center_world_m"),
            "intrinsic_matrix_3x3": row.get("intrinsic_matrix_3x3"),
            "image_size_wh": row.get("image_size_wh"),
            "reference_bounds_world": bounds,
        }
        geometry_sha = hashlib.sha256(
            json.dumps(
                geometry_payload,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if row.get("camera_geometry_source_sha256") != geometry_sha:
            raise LayoutContractError(f"{name}: live camera geometry digest mismatch")
        projected = project_world_target_to_pixel(
            camera_center_world_m=row.get("camera_center_world_m", []),
            camera_quaternion_world_wxyz_ros=row.get("camera_quaternion_world_wxyz_ros", []),
            target_center_world_m=row.get("target_center_world_m", []),
            intrinsic_matrix_3x3=row.get("intrinsic_matrix_3x3", []),
        )
        if row.get("target_projected_pixel_uv") is not None:
            supplied = row["target_projected_pixel_uv"]
            if len(supplied) != 2 or any(abs(float(a) - b) > 1e-6 for a, b in zip(supplied, projected)):
                raise LayoutContractError(f"{name}: supplied target projection differs from live geometry")
        evidence[name] = CameraEvidence(
            camera_name=name,
            camera_center_world_m=tuple(row.get("camera_center_world_m", [])),
            target_center_world_m=tuple(row.get("target_center_world_m", [])),
            reference_bounds_world=YawOrientedBox(
                center_world_m=tuple(bounds.get("center_world_m", [])),
                half_extents_m=tuple(bounds.get("half_extents_m", [])),
                yaw_world_rad=bounds.get("yaw_world_rad"),
            ),
            target_instance_visible_pixels=row.get("target_instance_visible_pixels"),
            segmentation_source_sha256=row.get("segmentation_source_sha256"),
            target_projected_pixel_uv=projected,
            image_size_wh=(tuple(row["image_size_wh"]) if row.get("image_size_wh") is not None else None),
            camera_geometry_source_sha256=geometry_sha,
        )
    camera_gate = evaluate_all_cameras(
        evidence,
        expected_cameras=candidate.expected_cameras,
        minimum_visible_target_pixels=minimum_visible_target_pixels,
    )
    scene = evaluate_layout(
        candidate,
        symmetry_level_s=snapshot.get("symmetry_level_s"),
        realised_object_poses=poses,
        occlusion_check_by_camera={name: row["occlusion_check"] for name, row in camera_gate.items()},
        target_visible_by_camera={name: row["target_visible"] for name, row in camera_gate.items()},
        arm_reset_pose=snapshot.get("arm_reset_pose", {}),
    )
    return {
        "schema_version": "vla-wam-shared-v3e004-live-scene-gate-v1",
        "status": "passed_model_blind_not_released_for_inference",
        "passed": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "candidate_sha256": candidate_sha256,
        "snapshot_sha256": snapshot_sha256,
        "minimum_visible_target_pixels": minimum_visible_target_pixels,
        "scene": scene,
        "cameras": camera_gate,
        "scope_caveat": "The object layout is assessed relative to the robot midline; arm reset and camera/embodiment symmetry are not assumed.",
        "release_boundary": "This model-blind gate does not by itself release policy inference; registration and runtime gates remain required.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--snapshot-sha256", required=True)
    parser.add_argument("--minimum-visible-target-pixels", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite retained gate: {args.output}")
    report = compile_live_gate(
        candidate_path=args.candidate,
        candidate_sha256=args.candidate_sha256,
        snapshot_path=args.snapshot,
        snapshot_sha256=args.snapshot_sha256,
        minimum_visible_target_pixels=args.minimum_visible_target_pixels,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "sha256": _sha(args.output), "passed": True}, indent=2))


if __name__ == "__main__":
    main()
