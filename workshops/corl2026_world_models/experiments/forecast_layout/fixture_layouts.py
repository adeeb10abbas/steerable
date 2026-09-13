#!/usr/bin/env python3
"""Build and validate unreleased, model-blind forecast-layout candidates.

The generator intentionally has no model, action, success, or outcome input.  It
maps a frozen namespace and layout ID through SHA-256 onto a quantized geometry
grid.  Generated coordinates are candidate inputs to a live physical gate, not
an inference release.  A separate frozen pose manifest can only be assembled
from accepted, hash-chained live-gate records.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


NAMESPACE = "wmf_ablation_001_20260912"
SOURCE_SCHEMA = "wmf-forecast-layout-source-contract-v1"
POOL_SCHEMA = "wmf-forecast-layout-candidate-pool-v1"
POSE_MANIFEST_SCHEMA = "wmf-forecast-layout-frozen-pose-manifest-v1"
SOURCE_STATUS = "FROZEN_MODEL_BLIND_SOURCE_INPUT"
CANDIDATE_STATUS = "NUMERIC_CANDIDATE_NOT_RELEASED_PENDING_LIVE_GATE"
POOL_STATUS = "NUMERIC_CANDIDATE_POOL_NOT_RELEASED_PENDING_LIVE_GATE"
POSE_MANIFEST_STATUS = "LIVE_GATE_QUALIFIED_POSES_FROZEN_MODEL_EXECUTION_NOT_RELEASED"
MOVABLE_OBJECTS = ("banana", "bowl", "rubiks_cube")
LAYOUT_ARMS = ("original", "reflected")
COMMANDS = ("left", "right")
PLANNED_LAYOUT_IDS = (
    "P00",
    *(f"D{index:02d}" for index in range(1, 5)),
    *(f"C{index:02d}" for index in range(1, 25)),
)
DEFAULT_CANDIDATES_PER_LAYOUT = 4  # one deterministic first choice plus three spares
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA1 = re.compile(r"^[0-9a-f]{40}$")


class LayoutContractError(ValueError):
    """Raised when layout evidence cannot be trusted fail-closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LayoutContractError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise LayoutContractError(f"value is not finite canonical JSON: {error}") from error


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise LayoutContractError(f"non-finite JSON constant is prohibited: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in output, f"duplicate JSON key is prohibited: {key}")
        output[key] = value
    return output


def strict_json_loads(payload: bytes, label: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LayoutContractError(f"{label} is not strict UTF-8 JSON: {error}") from error


def load_json_file(path: Path, expected_sha256: str | None = None) -> tuple[Any, bytes]:
    path = Path(path)
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise LayoutContractError(f"cannot read {path}: {error}") from error
    if expected_sha256 is not None:
        require(_SHA256.fullmatch(expected_sha256) is not None, "expected digest must be lowercase SHA-256")
        require(sha256_bytes(payload) == expected_sha256, f"SHA-256 mismatch for {path}")
    return strict_json_loads(payload, str(path)), payload


def _finite_vector(value: Any, length: int, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == length, f"{label} must be a {length}-vector")
    require(
        all(type(item) in (int, float) and math.isfinite(float(item)) for item in value),
        f"{label} must contain only finite numbers",
    )
    return [float(item) for item in value]


def _validate_sha(value: Any, label: str) -> str:
    require(isinstance(value, str) and _SHA256.fullmatch(value) is not None, f"{label} is not SHA-256")
    return value


def _validate_git_oid(value: Any, label: str) -> str:
    require(isinstance(value, str) and _GIT_SHA1.fullmatch(value) is not None, f"{label} is not a Git SHA-1 object ID")
    return value


def _grid_values(specification: Sequence[Any], label: str) -> tuple[float, ...]:
    require(len(specification) == 3, f"{label} grid must be [minimum, maximum, step]")
    low, high, step = (Decimal(str(value)) for value in specification)
    require(step > 0 and high >= low, f"{label} grid bounds are invalid")
    span = (high - low) / step
    require(span == span.to_integral_value(), f"{label} grid is not exactly divisible")
    return tuple(float(low + index * step) for index in range(int(span) + 1))


def _hash_pick(namespace: str, key: str, values: Sequence[Any]) -> Any:
    require(bool(values), f"empty deterministic choice set for {key}")
    digest = hashlib.sha256(f"{namespace}:{key}".encode("ascii")).digest()
    return values[int.from_bytes(digest, "big") % len(values)]


def reflect_position_y(position: Sequence[Any]) -> list[float]:
    x, y, z = _finite_vector(list(position), 3, "position")
    return [x, 0.0 if y == 0.0 else -y, z]


def _candidate_payload_sha(candidate: Mapping[str, Any]) -> str:
    core = dict(candidate)
    core.pop("candidate_payload_sha256", None)
    return sha256_bytes(canonical_json_bytes(core))


def validate_source_contract(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "source contract must be an object")
    require(value.get("schema_version") == SOURCE_SCHEMA, "source contract schema changed")
    require(value.get("study_namespace") == NAMESPACE, "source namespace changed")
    require(value.get("status") == SOURCE_STATUS, "source contract is not frozen")
    require(value.get("model_request_count") == 0, "source contract contains model requests")
    require(value.get("behavioral_episode_count") == 0, "source contract contains behavioral episodes")
    require(value.get("generation_uses_model_outcomes") is False, "candidate generation is not model-blind")
    scene = value.get("scene")
    require(isinstance(scene, dict) and scene.get("name") == "rubiks_cube_banana_bowl.usda", "scene changed")
    _validate_git_oid(scene.get("robolab_commit"), "RoboLab commit")
    for label in ("scene_asset", "scene_metadata"):
        row = scene.get(label)
        require(isinstance(row, dict) and isinstance(row.get("path"), str), f"{label} provenance is missing")
        _validate_git_oid(row.get("git_blob"), f"{label} Git blob")
    _validate_sha(scene["scene_asset"].get("lfs_oid_sha256"), "scene asset LFS OID")
    _validate_sha(scene["scene_metadata"].get("sha256"), "scene metadata digest")
    objects = value.get("objects")
    require(isinstance(objects, dict) and set(objects) == set(MOVABLE_OBJECTS), "movable-object source inventory changed")
    for name in MOVABLE_OBJECTS:
        row = objects[name]
        require(isinstance(row, dict), f"object source is invalid: {name}")
        quaternion = _finite_vector(row.get("quaternion_wxyz"), 4, f"{name} source quaternion")
        norm = math.sqrt(sum(item * item for item in quaternion))
        require(abs(norm - 1.0) <= 1e-3, f"{name} source quaternion is not normalized")
        require(isinstance(row.get("quaternion_source"), str) and row["quaternion_source"], f"{name} quaternion provenance missing")
        asset = row.get("asset")
        require(isinstance(asset, dict) and isinstance(asset.get("path"), str), f"{name} asset provenance missing")
        for field in ("git_blob", "git_payload_sha256", "lfs_oid_sha256"):
            (_validate_git_oid if field == "git_blob" else _validate_sha)(
                asset.get(field), f"{name} asset {field}"
            )
        require(type(row.get("z_m")) in (int, float), f"{name} source height missing")
    p00 = value.get("p00_original_positions_robot_base_m")
    require(isinstance(p00, dict) and set(p00) == set(MOVABLE_OBJECTS), "P00 source inventory changed")
    for name in MOVABLE_OBJECTS:
        position = _finite_vector(p00[name], 3, f"P00 {name} position")
        require(position[2] == float(objects[name]["z_m"]), f"P00 {name} height differs from its source")
    gate = value.get("live_gate")
    require(isinstance(gate, dict), "live-gate contract is missing")
    require(gate.get("repeat_resets_per_condition") >= 2, "live gate needs repeated resets")
    require(set(gate.get("required_cameras", [])) == {
        "head_camera", "over_shoulder_left_camera", "over_shoulder_right_camera", "wrist_cam"
    }, "required live cameras changed")
    require(set(gate.get("minimum_visible_pixels_by_object", {})) == set(MOVABLE_OBJECTS), "visibility inventory changed")
    return value


def _analytic_precheck(
    positions: Mapping[str, Sequence[Any]], generation: Mapping[str, Any]
) -> dict[str, Any]:
    require(set(positions) == set(MOVABLE_OBJECTS), "candidate position inventory changed")
    normalized = {name: _finite_vector(list(positions[name]), 3, f"{name} position") for name in MOVABLE_OBJECTS}
    minimum = float(generation["minimum_planar_center_separation_m"])
    separations: dict[str, float] = {}
    for left_index, left in enumerate(MOVABLE_OBJECTS):
        for right in MOVABLE_OBJECTS[left_index + 1 :]:
            distance = math.hypot(
                normalized[left][0] - normalized[right][0],
                normalized[left][1] - normalized[right][1],
            )
            separations[f"{left}|{right}"] = distance
    cube = normalized["rubiks_cube"]
    bowl = normalized["bowl"]
    longitudinal = abs(cube[0] - bowl[0])
    lateral = abs(cube[1] - bowl[1])
    neutrality_margin = float(generation["neutrality_margin_m"])
    maximum_abs_y = float(generation["maximum_abs_y_m"])
    checks = {
        "all_pairwise_planar_center_separations_at_least_frozen_minimum": all(
            value >= minimum for value in separations.values()
        ),
        "cube_outside_both_45_degree_lateral_success_cones_with_margin": (
            lateral + neutrality_margin < longitudinal
        ),
        "all_positions_within_frozen_lateral_bound": all(
            abs(position[1]) <= maximum_abs_y for position in normalized.values()
        ),
        "all_source_heights_positive": all(position[2] > 0.0 for position in normalized.values()),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "pairwise_planar_center_separation_m": separations,
        "minimum_planar_center_separation_m": minimum,
        "cube_bowl_abs_longitudinal_separation_m": longitudinal,
        "cube_bowl_abs_lateral_separation_m": lateral,
        "neutrality_margin_m": neutrality_margin,
        "claim_boundary": "Analytic center checks do not establish collision-free settling, rendered visibility, reachability, or false live success predicates.",
    }


def _hashed_positions(
    source: Mapping[str, Any], layout_id: str, derivation_attempt: int
) -> dict[str, list[float]]:
    grid = source["generation_grid"]
    prefix = f"layout-v1:{layout_id}:attempt-{derivation_attempt:04d}"
    center = _hash_pick(NAMESPACE, f"{prefix}:shared-y", _grid_values(grid["shared_lateral_center"], "shared lateral center"))
    cube_y_offset = _hash_pick(NAMESPACE, f"{prefix}:cube-y", _grid_values(grid["cube_lateral_offset_from_center"], "cube y"))
    bowl_y_offset = _hash_pick(NAMESPACE, f"{prefix}:bowl-y", _grid_values(grid["bowl_lateral_offset_from_center"], "bowl y"))
    cube_x = _hash_pick(NAMESPACE, f"{prefix}:cube-x", _grid_values(grid["cube_x"], "cube x"))
    bowl_dx = _hash_pick(
        NAMESPACE,
        f"{prefix}:bowl-dx",
        _grid_values(grid["bowl_longitudinal_offset_from_cube"], "bowl longitudinal offset"),
    )
    banana_x = _hash_pick(NAMESPACE, f"{prefix}:banana-x", _grid_values(grid["banana_x"], "banana x"))
    banana_abs_dy = _hash_pick(
        NAMESPACE,
        f"{prefix}:banana-abs-y",
        _grid_values(grid["banana_abs_lateral_offset_from_center"], "banana absolute y offset"),
    )
    banana_sign = _hash_pick(NAMESPACE, f"{prefix}:banana-y-sign", (-1.0, 1.0))
    objects = source["objects"]
    return {
        "rubiks_cube": [cube_x, center + cube_y_offset, float(objects["rubiks_cube"]["z_m"])],
        "bowl": [cube_x + bowl_dx, center + bowl_y_offset, float(objects["bowl"]["z_m"])],
        "banana": [banana_x, center + banana_sign * banana_abs_dy, float(objects["banana"]["z_m"])],
    }


def _build_candidate(
    source: Mapping[str, Any], source_sha256: str, layout_id: str, rank: int,
    derivation_attempt: int, original_positions: Mapping[str, Sequence[Any]],
) -> dict[str, Any]:
    original = {name: _finite_vector(list(original_positions[name]), 3, f"{name} position") for name in MOVABLE_OBJECTS}
    reflected = {name: reflect_position_y(original[name]) for name in MOVABLE_OBJECTS}
    quaternions = {
        name: _finite_vector(source["objects"][name]["quaternion_wxyz"], 4, f"{name} quaternion")
        for name in MOVABLE_OBJECTS
    }
    precheck = _analytic_precheck(original, source["generation_grid"])
    require(precheck["passed"], f"internal generator emitted an analytically invalid candidate for {layout_id}")
    candidate: dict[str, Any] = {
        "schema_version": "wmf-forecast-layout-candidate-v1",
        "study_namespace": NAMESPACE,
        "layout_pair_id": layout_id,
        "candidate_id": f"{layout_id}__candidate_{rank:02d}",
        "candidate_rank": rank,
        "derivation_attempt": derivation_attempt,
        "derivation_key_sha256": sha256_bytes(
            f"{NAMESPACE}:layout-v1:{layout_id}:attempt-{derivation_attempt:04d}".encode("ascii")
        ),
        "status": CANDIDATE_STATUS,
        "released": False,
        "live_gate_passed": False,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "source_contract_sha256": source_sha256,
        "object_asset_provenance": {
            name: {
                "asset_path": source["objects"][name]["asset"]["path"],
                "asset_lfs_oid_sha256": source["objects"][name]["asset"]["lfs_oid_sha256"],
                "scene_metadata_sha256": source["scene"]["scene_metadata"]["sha256"],
                "quaternion_source": source["objects"][name]["quaternion_source"],
            }
            for name in MOVABLE_OBJECTS
        },
        "layouts": {
            "original": {
                "positions_robot_base_m": original,
                "quaternions_wxyz": quaternions,
            },
            "reflected": {
                "positions_robot_base_m": reflected,
                "quaternions_wxyz": quaternions,
            },
        },
        "factor": {
            "transform": "for rubiks_cube, bowl, and banana only: (x,y,z) -> (x,-y,z)",
            "positions_reflected_exactly": True,
            "quaternion_sources_unchanged": True,
            "nonmovable_scene_geometry_unchanged": True,
            "full_improper_orientation_reflection_claimed": False,
        },
        "analytic_precheck": precheck,
        "release_boundary": "Numeric candidate only. Live model-blind settle, visibility, camera, collision, predicate, and matched-reset checks must pass before this pose can enter a frozen manifest; that manifest still does not release model inference.",
    }
    candidate["candidate_payload_sha256"] = _candidate_payload_sha(candidate)
    return candidate


def validate_candidate(candidate: Any, source_sha256: str | None = None) -> dict[str, Any]:
    require(isinstance(candidate, dict), "candidate must be an object")
    require(candidate.get("schema_version") == "wmf-forecast-layout-candidate-v1", "candidate schema changed")
    require(candidate.get("study_namespace") == NAMESPACE, "candidate namespace changed")
    require(candidate.get("layout_pair_id") in PLANNED_LAYOUT_IDS, "candidate layout ID is not planned")
    rank = candidate.get("candidate_rank")
    require(type(rank) is int and rank >= 0, "candidate rank is invalid")
    require(candidate.get("candidate_id") == f"{candidate['layout_pair_id']}__candidate_{rank:02d}", "candidate ID changed")
    require(candidate.get("status") == CANDIDATE_STATUS, "candidate is not explicitly unreleased")
    require(candidate.get("released") is False and candidate.get("live_gate_passed") is False, "candidate bypassed live gate")
    require(candidate.get("model_request_count") == 0 and candidate.get("behavioral_episode_count") == 0, "candidate contains model evidence")
    _validate_sha(candidate.get("source_contract_sha256"), "candidate source digest")
    if source_sha256 is not None:
        require(candidate["source_contract_sha256"] == source_sha256, "candidate source binding changed")
    layouts = candidate.get("layouts")
    require(isinstance(layouts, dict) and set(layouts) == set(LAYOUT_ARMS), "candidate layout arms changed")
    for arm in LAYOUT_ARMS:
        row = layouts[arm]
        require(isinstance(row, dict), f"candidate {arm} layout is invalid")
        require(set(row.get("positions_robot_base_m", {})) == set(MOVABLE_OBJECTS), f"{arm} positions changed")
        require(set(row.get("quaternions_wxyz", {})) == set(MOVABLE_OBJECTS), f"{arm} quaternions changed")
    for name in MOVABLE_OBJECTS:
        original = _finite_vector(layouts["original"]["positions_robot_base_m"][name], 3, f"original {name}")
        reflected = _finite_vector(layouts["reflected"]["positions_robot_base_m"][name], 3, f"reflected {name}")
        require(reflected == reflect_position_y(original), f"{name} is not an exact y reflection")
        original_q = _finite_vector(layouts["original"]["quaternions_wxyz"][name], 4, f"original {name} quaternion")
        reflected_q = _finite_vector(layouts["reflected"]["quaternions_wxyz"][name], 4, f"reflected {name} quaternion")
        require(original_q == reflected_q, f"{name} quaternion changed across position reflection")
    expected_payload_sha = _candidate_payload_sha(candidate)
    require(candidate.get("candidate_payload_sha256") == expected_payload_sha, "candidate payload digest mismatch")
    return candidate


def build_candidate_pool(
    source: Mapping[str, Any], source_sha256: str,
    candidates_per_layout: int = DEFAULT_CANDIDATES_PER_LAYOUT,
) -> dict[str, Any]:
    validate_source_contract(source)
    _validate_sha(source_sha256, "source contract digest")
    require(candidates_per_layout >= 2, "candidate pool must retain at least one spare per layout")
    require(candidates_per_layout <= 32, "candidate pool is unexpectedly large")
    candidates: list[dict[str, Any]] = []
    seen_geometries: set[bytes] = set()
    for layout_id in PLANNED_LAYOUT_IDS:
        rank = 0
        derivation_attempt = 0
        while rank < candidates_per_layout:
            if layout_id == "P00" and rank == 0:
                positions = source["p00_original_positions_robot_base_m"]
                attempt = -1
            else:
                attempt = derivation_attempt
                derivation_attempt += 1
                positions = _hashed_positions(source, layout_id, attempt)
            precheck = _analytic_precheck(positions, source["generation_grid"])
            if not precheck["passed"]:
                continue
            geometry_key = canonical_json_bytes({
                name: _finite_vector(list(positions[name]), 3, name) for name in MOVABLE_OBJECTS
            })
            if geometry_key in seen_geometries:
                continue
            seen_geometries.add(geometry_key)
            candidates.append(
                _build_candidate(source, source_sha256, layout_id, rank, attempt, positions)
            )
            rank += 1
            require(derivation_attempt < 100000, f"candidate generation exhausted for {layout_id}")
    pool = {
        "schema_version": POOL_SCHEMA,
        "study_namespace": NAMESPACE,
        "status": POOL_STATUS,
        "released": False,
        "launch_ready": False,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "generation_uses_model_outcomes": False,
        "candidate_generation_inputs": [
            "frozen study namespace",
            "planned layout pair ID",
            "candidate derivation attempt",
            "pinned scene asset and source quaternions",
            "frozen quantized geometry grid",
        ],
        "inputs_not_consumed": [
            "model identity",
            "model outputs or forecasts",
            "policy actions",
            "episode success or failure",
            "annotation labels",
            "scientific endpoint values",
        ],
        "algorithm": {
            "name": "sha256_domain_separated_quantized_geometry_v1",
            "choice_rule": "Interpret full SHA-256 digest as an unsigned big-endian integer and reduce modulo the ordered grid length.",
            "duplicate_rule": "Skip exact duplicate original geometries in deterministic derivation-attempt order.",
            "analytic_rejection_rule": "Skip only frozen geometry-bound, pairwise-center, and neutral-cone failures; never inspect model evidence.",
            "reflection_rule": "Exact per-object y negation; source quaternion bytes are unchanged.",
            "p00_primary_rule": "candidate rank 0 is the pinned historical neutral layout; P00 spares use the same hash generator as new layouts.",
        },
        "source_contract_sha256": source_sha256,
        "planned_layout_ids": list(PLANNED_LAYOUT_IDS),
        "planned_layout_count": len(PLANNED_LAYOUT_IDS),
        "candidates_per_layout": candidates_per_layout,
        "spares_per_layout": candidates_per_layout - 1,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "release_boundary": "Every numeric row is unreleased. Only an accepted append-only live-gate record may select one candidate into a frozen pose manifest; physical qualification alone does not release a model job.",
    }
    validate_candidate_pool(pool, source_sha256=source_sha256)
    return pool


def validate_candidate_pool(pool: Any, source_sha256: str | None = None) -> dict[str, Any]:
    require(isinstance(pool, dict), "candidate pool must be an object")
    require(pool.get("schema_version") == POOL_SCHEMA, "candidate-pool schema changed")
    require(pool.get("study_namespace") == NAMESPACE, "candidate-pool namespace changed")
    require(pool.get("status") == POOL_STATUS, "candidate pool is not explicitly unreleased")
    require(pool.get("released") is False and pool.get("launch_ready") is False, "candidate pool bypassed live gates")
    require(pool.get("generation_uses_model_outcomes") is False, "candidate pool is not model-blind")
    require(pool.get("model_request_count") == 0 and pool.get("behavioral_episode_count") == 0, "candidate pool contains model evidence")
    pool_source_sha = _validate_sha(pool.get("source_contract_sha256"), "pool source digest")
    if source_sha256 is not None:
        require(pool_source_sha == source_sha256, "candidate-pool source digest mismatch")
    require(pool.get("planned_layout_ids") == list(PLANNED_LAYOUT_IDS), "candidate-pool layout inventory changed")
    count = pool.get("candidates_per_layout")
    require(type(count) is int and count >= 2, "candidate pool has no deterministic spares")
    rows = pool.get("candidates")
    require(isinstance(rows, list), "candidate-pool rows are missing")
    require(pool.get("candidate_count") == len(PLANNED_LAYOUT_IDS) * count == len(rows), "candidate-pool count mismatch")
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    geometries: set[bytes] = set()
    for row in rows:
        candidate = validate_candidate(row, source_sha256=pool_source_sha)
        key = (candidate["layout_pair_id"], candidate["candidate_rank"])
        require(key not in indexed, f"duplicate candidate: {key}")
        indexed[key] = candidate
        geometry = canonical_json_bytes(candidate["layouts"]["original"]["positions_robot_base_m"])
        require(geometry not in geometries, "duplicate geometry appears under multiple candidate IDs")
        geometries.add(geometry)
    expected = {(layout_id, rank) for layout_id in PLANNED_LAYOUT_IDS for rank in range(count)}
    require(set(indexed) == expected, "candidate ranks are incomplete")
    return pool


def candidate_index(pool: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    validate_candidate_pool(pool)
    return {row["candidate_id"]: row for row in pool["candidates"]}


def validate_frozen_pose_manifest(manifest: Any) -> dict[str, Any]:
    require(isinstance(manifest, dict), "pose manifest must be an object")
    require(manifest.get("schema_version") == POSE_MANIFEST_SCHEMA, "pose-manifest schema changed")
    require(manifest.get("study_namespace") == NAMESPACE, "pose-manifest namespace changed")
    require(manifest.get("status") == POSE_MANIFEST_STATUS, "pose manifest is not live-qualified and frozen")
    require(manifest.get("physical_layout_gate_passed") is True, "pose manifest lacks a passed physical gate")
    require(manifest.get("released_for_model_inference") is False, "pose manifest must not itself release inference")
    _validate_sha(manifest.get("candidate_pool_sha256"), "pose-manifest candidate-pool digest")
    _validate_sha(manifest.get("gate_ledger_sha256"), "pose-manifest gate-ledger digest")
    rows = manifest.get("layout_pairs")
    require(isinstance(rows, dict) and rows, "pose manifest has no qualified layouts")
    require(set(rows).issubset(set(PLANNED_LAYOUT_IDS)), "pose manifest contains an unplanned layout")
    for layout_id, row in rows.items():
        require(isinstance(row, dict) and row.get("layout_pair_id") == layout_id, f"pose row ID mismatch: {layout_id}")
        _validate_sha(row.get("candidate_payload_sha256"), f"{layout_id} candidate digest")
        _validate_sha(row.get("accepted_gate_record_sha256"), f"{layout_id} gate-record digest")
        layouts = row.get("layouts")
        require(isinstance(layouts, dict) and set(layouts) == set(LAYOUT_ARMS), f"{layout_id} arms changed")
        for name in MOVABLE_OBJECTS:
            original = _finite_vector(layouts["original"]["positions_robot_base_m"][name], 3, f"{layout_id} original {name}")
            reflected = _finite_vector(layouts["reflected"]["positions_robot_base_m"][name], 3, f"{layout_id} reflected {name}")
            require(reflected == reflect_position_y(original), f"{layout_id}/{name} pose is not an exact y reflection")
            original_q = _finite_vector(layouts["original"]["quaternions_wxyz"][name], 4, f"{layout_id} original {name} quaternion")
            reflected_q = _finite_vector(layouts["reflected"]["quaternions_wxyz"][name], 4, f"{layout_id} reflected {name} quaternion")
            require(original_q == reflected_q, f"{layout_id}/{name} source quaternion changed")
    require(manifest.get("qualified_layout_count") == len(rows), "qualified-layout count mismatch")
    return manifest


def load_frozen_pose_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    value, _ = load_json_file(path, expected_sha256)
    return validate_frozen_pose_manifest(value)


def write_immutable(path: Path, payload: bytes) -> None:
    path = Path(path)
    if path.exists():
        require(path.read_bytes() == payload, f"refusing to replace different evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _default_source_path() -> Path:
    return Path(__file__).with_name("layout_source_contract.json")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="write the immutable unreleased candidate pool")
    generate.add_argument("--source-contract", type=Path, default=_default_source_path())
    generate.add_argument("--candidates-per-layout", type=int, default=DEFAULT_CANDIDATES_PER_LAYOUT)
    generate.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate", help="validate an existing candidate pool")
    validate.add_argument("--pool", type=Path, required=True)
    validate.add_argument("--pool-sha256")
    args = parser.parse_args(argv)
    if args.command == "generate":
        source, source_payload = load_json_file(args.source_contract)
        source = validate_source_contract(source)
        pool = build_candidate_pool(
            source,
            sha256_bytes(source_payload),
            candidates_per_layout=args.candidates_per_layout,
        )
        payload = canonical_json_bytes(pool)
        write_immutable(args.output, payload)
        print(json.dumps({
            "path": str(args.output.resolve()),
            "sha256": sha256_bytes(payload),
            "candidate_count": pool["candidate_count"],
            "released": False,
        }, sort_keys=True))
        return 0
    pool, _ = load_json_file(args.pool, args.pool_sha256)
    validate_candidate_pool(pool)
    print(json.dumps({"path": str(args.pool.resolve()), "sha256": sha256_file(args.pool), "valid": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
