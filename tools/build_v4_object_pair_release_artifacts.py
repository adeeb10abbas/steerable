#!/usr/bin/env python3
"""Promote qualified C7 reset and scoring artifacts for policy gates."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any


CANDIDATE_SCHEMA = "v4-droid-object-pair-reset-registry-v1"
G2_SCHEMA = "v4-object-pair-g2-aggregate-receipt-v1"
G3_SCHEMA = "v4-object-pair-g3-aggregate-receipt-v1"
G4_SCHEMA = "v4-object-pair-g4-nano-policy-session-receipt-v1"
FIXTURE_ID = "object_pair"
POLICY_ID = "cosmos3_nano_droid"
CHECKPOINT_REVISION = "6706d7680581c255ff61e0f3bb49d90eac55c79e"
MODEL_BLIND_STATUS = "model_blind_candidate_not_released_for_inference"
RELEASED_STATUS = "released_for_policy_inference"
SUPPORT_EDGE_MARGIN_M = 0.005
RELATION_CLEARANCE_M = 0.01


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def require_pass(
    payload: dict[str, Any],
    *,
    schema: str,
    path: Path,
) -> None:
    if payload.get("schema_version") != schema:
        raise ValueError(f"{path} schema mismatch")
    if payload.get("fixture_id") != FIXTURE_ID:
        raise ValueError(f"{path} fixture mismatch")
    if payload.get("passed") is not True or payload.get("status") != "passed":
        raise ValueError(f"{path} is not a passing receipt")
    if payload.get("behavioral_episode_count") != 0:
        raise ValueError(f"{path} unexpectedly contains behavioral episodes")


def world_aabb_to_task(
    bounds: dict[str, Any],
    *,
    axes: tuple[tuple[float, float, float], ...],
    origin: tuple[float, float, float],
) -> list[float]:
    lo = bounds.get("min_xyz")
    hi = bounds.get("max_xyz")
    if not isinstance(lo, list) or not isinstance(hi, list):
        raise ValueError("live scene AABB is incomplete")
    corners = itertools.product(
        (float(lo[0]), float(hi[0])),
        (float(lo[1]), float(hi[1])),
        (float(lo[2]), float(hi[2])),
    )
    transformed: list[tuple[float, float, float]] = []
    for point in corners:
        delta = tuple(value - base for value, base in zip(point, origin))
        transformed.append(
            tuple(sum(a * b for a, b in zip(axis, delta)) for axis in axes)
        )
    return [
        min(point[0] for point in transformed),
        max(point[0] for point in transformed),
        min(point[1] for point in transformed),
        max(point[1] for point in transformed),
        min(point[2] for point in transformed),
        max(point[2] for point in transformed),
    ]


def build_scoring_geometry(
    *,
    initial_scene: dict[str, Any],
    initial_scene_path: Path,
    g2_path: Path,
    g3_path: Path,
) -> dict[str, Any]:
    axes = (
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    origin = (0.0, 0.0, 0.0)
    objects = initial_scene.get("objects")
    if not isinstance(objects, dict):
        raise ValueError("initial scene lacks object states")
    sponge = objects.get("sponge")
    tray = objects.get("tray")
    if not isinstance(sponge, dict) or not isinstance(tray, dict):
        raise ValueError("initial scene lacks sponge/tray states")
    table = world_aabb_to_task(
        initial_scene.get("table_world_aabb_m", {}),
        axes=axes,
        origin=origin,
    )
    sponge_bounds = world_aabb_to_task(
        sponge.get("world_aabb_m", {}),
        axes=axes,
        origin=origin,
    )
    tray_bounds = world_aabb_to_task(
        tray.get("world_aabb_m", {}),
        axes=axes,
        origin=origin,
    )
    target_half = [
        (sponge_bounds[1] - sponge_bounds[0]) / 2.0,
        (sponge_bounds[3] - sponge_bounds[2]) / 2.0,
        (sponge_bounds[5] - sponge_bounds[4]) / 2.0,
    ]
    reference_half = [
        (tray_bounds[1] - tray_bounds[0]) / 2.0,
        (tray_bounds[3] - tray_bounds[2]) / 2.0,
        (tray_bounds[5] - tray_bounds[4]) / 2.0,
    ]
    workspace = [
        table[0] + target_half[0] + SUPPORT_EDGE_MARGIN_M,
        table[1] - target_half[0] - SUPPORT_EDGE_MARGIN_M,
        table[2] + target_half[1] + SUPPORT_EDGE_MARGIN_M,
        table[3] - target_half[1] - SUPPORT_EDGE_MARGIN_M,
        table[5],
        table[5] + 2.0 * target_half[2],
    ]
    if any(
        workspace[index] >= workspace[index + 1]
        for index in (0, 2, 4)
    ):
        raise ValueError("qualified object-pair scoring workspace is empty")
    d_cap_m = math.dist(
        (workspace[0], workspace[2], workspace[4]),
        (workspace[1], workspace[3], workspace[5]),
    )
    return {
        "schema_version": "v4-object-pair-scoring-geometry-v1",
        "campaign_id": "online_correction_v4",
        "fixture_id": FIXTURE_ID,
        "status": "released_for_policy_qualification",
        "behavioral_episode_count": 0,
        "model_request_count": 0,
        "task_frame": {
            "basis_convention": "task coordinates are left, front, up",
            "u_left": list(axes[0]),
            "u_front": list(axes[1]),
            "u_up": list(axes[2]),
            "origin": list(origin),
        },
        "workspace": {
            "x_min": workspace[0],
            "x_max": workspace[1],
            "y_min": workspace[2],
            "y_max": workspace[3],
            "z_min": workspace[4],
            "z_max": workspace[5],
        },
        "object_footprint": {
            "half_left": target_half[0],
            "half_front": target_half[1],
            "half_up": target_half[2],
        },
        "reference_footprint": {
            "half_left": reference_half[0],
            "half_front": reference_half[1],
            "half_up": reference_half[2],
        },
        "clearance_m": RELATION_CLEARANCE_M,
        "d_cap_m": d_cap_m,
        "geometry_provenance": {
            "target_object": "sponge",
            "reference_object": "tray",
            "support_edge_margin_m": SUPPORT_EDGE_MARGIN_M,
        },
        "qualification_basis": {
            "g2_aggregate": artifact(g2_path),
            "g3_aggregate": artifact(g3_path),
            "live_initial_scene": artifact(initial_scene_path),
        },
        "release_boundary": (
            "This geometry is released for C7 G5-G8 policy qualification. "
            "Confirmatory episodes still require every downstream gate and a "
            "released immutable runtime lock."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-reset-registry", type=Path, required=True)
    parser.add_argument("--g2-aggregate", type=Path, required=True)
    parser.add_argument("--g3-aggregate", type=Path, required=True)
    parser.add_argument("--g4-receipt", type=Path, required=True)
    parser.add_argument("--initial-scene", type=Path, required=True)
    parser.add_argument("--initial-scene-sha256", required=True)
    parser.add_argument("--released-reset-registry", type=Path, required=True)
    parser.add_argument("--scoring-geometry", type=Path, required=True)
    args = parser.parse_args()

    for output in (args.released_reset_registry, args.scoring_geometry):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite release artifact: {output}")

    candidate = load_json(args.candidate_reset_registry)
    g2 = load_json(args.g2_aggregate)
    g3 = load_json(args.g3_aggregate)
    g4 = load_json(args.g4_receipt)
    initial_scene = load_json(args.initial_scene)
    if candidate.get("schema_version") != CANDIDATE_SCHEMA:
        raise ValueError("candidate reset registry schema mismatch")
    if candidate.get("fixture_id") != FIXTURE_ID:
        raise ValueError("candidate reset registry fixture mismatch")
    if candidate.get("status") != MODEL_BLIND_STATUS:
        raise ValueError("candidate reset registry is not model-blind")
    if candidate.get("registered_env_seed_count") != 64:
        raise ValueError("candidate reset registry must contain all 64 C7 seeds")
    require_pass(g2, schema=G2_SCHEMA, path=args.g2_aggregate)
    require_pass(g3, schema=G3_SCHEMA, path=args.g3_aggregate)
    require_pass(g4, schema=G4_SCHEMA, path=args.g4_receipt)
    candidate_sha256 = sha256_file(args.candidate_reset_registry)
    if g2.get("reset_registry", {}).get("sha256") != candidate_sha256:
        raise ValueError("G2 aggregate does not bind the candidate reset registry")
    if g3.get("selected_scale") != 0.5 or g3.get("selected_displacement_m") != 0.06:
        raise ValueError("G3 aggregate does not bind the qualified C7 scale")
    if g4.get("policy_id") != POLICY_ID:
        raise ValueError("G4 policy mismatch")
    if g4.get("checkpoint_revision") != CHECKPOINT_REVISION:
        raise ValueError("G4 checkpoint revision mismatch")
    if g4.get("model_request_count") != 3:
        raise ValueError("G4 receipt must contain exactly three qualification requests")
    if sha256_file(args.initial_scene) != args.initial_scene_sha256:
        raise ValueError("live initial-scene digest mismatch")

    release_basis = {
        "candidate_reset_registry": artifact(args.candidate_reset_registry),
        "g2_aggregate": artifact(args.g2_aggregate),
        "g3_aggregate": artifact(args.g3_aggregate),
        "g4_policy_session": artifact(args.g4_receipt),
    }
    released = {
        **candidate,
        "status": RELEASED_STATUS,
        "scene_receipt": {
            **candidate["scene_receipt"],
            "status": "qualified_object_pair_asset",
        },
        "qualification_release_basis": release_basis,
        "release_boundary": (
            "Released for C7 Nano G5-G8 policy qualification after passing G2, "
            "G3, and G4. Confirmatory policy episodes remain blocked until G5-G8 "
            "pass and the immutable runtime lock is released."
        ),
    }
    scoring_geometry = build_scoring_geometry(
        initial_scene=initial_scene,
        initial_scene_path=args.initial_scene,
        g2_path=args.g2_aggregate,
        g3_path=args.g3_aggregate,
    )
    args.released_reset_registry.parent.mkdir(parents=True, exist_ok=True)
    args.scoring_geometry.parent.mkdir(parents=True, exist_ok=True)
    args.released_reset_registry.write_bytes(canonical_json_bytes(released))
    args.scoring_geometry.write_bytes(canonical_json_bytes(scoring_geometry))
    print(
        json.dumps(
            {
                "released_reset_registry": artifact(args.released_reset_registry),
                "scoring_geometry": artifact(args.scoring_geometry),
                "d_cap_m": scoring_geometry["d_cap_m"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
