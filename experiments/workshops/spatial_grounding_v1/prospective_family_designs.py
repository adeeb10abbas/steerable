"""Plan and author bounded prospective HEIGHT/DIST perturbation overlays.

This module deliberately operates between the four baseline zero-action
captures and measured-layout materialization.  A plan row is a prospective
design, never a measured layout or a qualified fixture.  Every accepted row
must receive a new zero-model capture of its own overlay before downstream
measurement materialization can consume it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .fixtures import Pose, pose_error
from .lat_candidate_generator import workspace_digest
from .prospective_family_capture import _sha256

PLAN_SCHEMA = "sgw-01-prospective-family-design-plan-v1"
CANDIDATE_OVERLAY_SCHEMA = "sgw-01-prospective-family-candidate-overlay-v1"
CAPTURE_STATUS = "prospective_native_capture_not_candidate_qualified"
BASELINE_STATUS = "prospective_scene_design_not_measured_or_qualified"
CANDIDATE_STATUS = "prospective_candidate_design_requires_zero_model_capture"
MAX_DESIGNS = 100
TRANSLATION_LIMIT_M = 0.04


def build_design_plan(
    *,
    family: str,
    seed: int,
    baseline_capture_paths: Mapping[str, Path],
    baseline_manifest_paths: Mapping[str, Path],
    count: int = MAX_DESIGNS,
) -> dict[str, Any]:
    """Build a finite, no-refill plan using the two measured side baselines."""

    if family not in {"HEIGHT", "DIST"}:
        raise ValueError("family must be HEIGHT or DIST")
    if not 1 <= count <= MAX_DESIGNS:
        raise ValueError("design count must be within the inclusive 1..100 cap")
    sides = ("left", "right")
    if set(baseline_capture_paths) != set(sides) or set(baseline_manifest_paths) != set(sides):
        raise ValueError("both left and right measured baseline captures and manifests are required")
    baselines = {
        side: _baseline_binding(
            family=family,
            side=side,
            capture_path=Path(baseline_capture_paths[side]),
            manifest_path=Path(baseline_manifest_paths[side]),
        )
        for side in sides
    }
    rows = []
    accepted = []
    fingerprints: list[dict[str, Any]] = []
    for index in range(count):
        side = sides[index % len(sides)]
        translation = _translation(seed, family, index)
        binding = baselines[side]
        roots = _translated_roots(binding["capture"]["objects"], family, translation)
        row = {
            "design_id": f"SGW-{family}-DESIGN-{index:03d}",
            "family": family,
            "seed": seed,
            "ordinal": index,
            "side": side,
            "status": "prospective_design_rejected_geometrically",
            "model_request_count": 0,
            "behavioral_episode_count": 0,
            "translation_xy_m": translation,
            "baseline": binding["binding"],
            "authored_scored_object_roots": roots,
            "required_next_step": "author immutable overlay then run this design's zero-model native capture",
        }
        rejection = _geometric_rejection(binding["capture"], family, side, roots, translation, fingerprints)
        if rejection is None:
            row["status"] = "prospective_design_requires_zero_model_capture"
            row["candidate_overlay_status"] = CANDIDATE_STATUS
            accepted.append(row["design_id"])
            fingerprints.append({"roots": roots})
        else:
            row["geometric_rejection"] = rejection
        rows.append(row)
    value = {
        "schema_version": PLAN_SCHEMA,
        "family": family,
        "seed": seed,
        "design_slot_count": count,
        "accepted_design_count": len(accepted),
        "geometric_rejection_count": count - len(accepted),
        "accepted_design_ids": accepted,
        "status": "prospective_design_plan_not_measured_or_qualified",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "baselines": {side: baselines[side]["binding"] for side in sides},
        "designs": rows,
    }
    value["plan_sha256"] = _digest(value, "plan_sha256")
    return value


def author_candidate_overlay(*, plan: Mapping[str, Any], design_id: str, output: Path, manifest_output: Path) -> dict[str, Any]:
    """Author one immutable perturbation overlay from a hash-verified plan row."""

    _validate_plan(plan)
    if output.exists() or manifest_output.exists():
        raise FileExistsError("refusing to overwrite a prospective candidate overlay or manifest")
    row = next((item for item in plan["designs"] if item["design_id"] == design_id), None)
    if not isinstance(row, Mapping):
        raise ValueError("design ID is not present in the bound plan")
    if row.get("status") != "prospective_design_requires_zero_model_capture":
        raise ValueError("rejected designs do not have a candidate overlay")
    binding = row["baseline"]
    capture_path = Path(binding["capture"]["path"])
    manifest_path = Path(binding["overlay_manifest"]["path"])
    capture = _baseline_capture(capture_path, expected_sha256=binding["capture"]["sha256"])
    manifest = _prospective_manifest(manifest_path, expected_sha256=binding["overlay_manifest"]["sha256"])
    if capture["overlay_manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("baseline capture does not bind its baseline overlay manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_candidate_usda(manifest, capture, row), encoding="utf-8")
    value = {
        "schema_version": CANDIDATE_OVERLAY_SCHEMA,
        "status": CANDIDATE_STATUS,
        "family": plan["family"],
        "design_id": design_id,
        "plan_sha256": plan["plan_sha256"],
        "source_baseline": binding,
        "base_scene": binding["base_scene"],
        "base_workspace_receipt": binding["base_workspace_receipt"],
        "native_import_contract": binding["native_import_contract"],
        "inherited_overlay_dependencies": binding["used_layer_dependencies"],
        "design_sha256": _digest(row),
        "overlay_usda": {"path": str(output.resolve()), "sha256": _sha256(output), "bytes": output.stat().st_size},
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "required_next_step": "run a fresh zero-model native capture using this exact overlay manifest",
    }
    value["manifest_sha256"] = _digest(value, "manifest_sha256")
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return value


def require_design_capture(*, candidate_manifest_path: Path, capture_path: Path) -> dict[str, Any]:
    """Fail closed until a fresh capture proves this candidate overlay was loaded."""

    manifest = _candidate_manifest(candidate_manifest_path)
    capture = _baseline_capture(capture_path)
    if capture["family"] != manifest["family"]:
        raise ValueError("candidate capture family differs from candidate overlay")
    if capture.get("overlay_manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("candidate capture does not bind the candidate overlay manifest")
    _capture_dependencies_include(capture, manifest["overlay_usda"], manifest["inherited_overlay_dependencies"])
    return capture


def _baseline_binding(*, family: str, side: str, capture_path: Path, manifest_path: Path) -> dict[str, Any]:
    capture = _baseline_capture(capture_path)
    manifest = _prospective_manifest(manifest_path)
    expected = {"upper_support_side": side} if family == "HEIGHT" else {"bowl_side": side}
    if capture["family"] != family or manifest["family"] != family or manifest.get("counterbalance") != expected:
        raise ValueError(f"{family} {side} baseline does not match its required side")
    if capture.get("overlay_manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError(f"{family} {side} capture does not bind the supplied overlay manifest")
    if capture.get("asset_manifest_sha256") != manifest["base_workspace_receipt"]["asset_manifest_sha256"]:
        raise ValueError(f"{family} {side} capture asset identity differs from its overlay manifest")
    if capture.get("robolab_commit") != manifest["base_workspace_receipt"]["robolab_commit"]:
        raise ValueError(f"{family} {side} capture RoboLab identity differs from its overlay manifest")
    dependencies = _validated_capture_dependencies(capture, manifest)
    binding = {
        "capture": {"path": str(capture_path.resolve()), "sha256": _sha256(capture_path), "receipt_sha256": capture["receipt_sha256"]},
        "overlay_manifest": {"path": str(manifest_path.resolve()), "sha256": _sha256(manifest_path), "manifest_sha256": manifest["manifest_sha256"]},
        "base_scene": manifest["base_scene"],
        "base_workspace_receipt": manifest["base_workspace_receipt"],
        "native_import_contract": manifest["native_import_contract"],
        "asset_manifest_sha256": capture["asset_manifest_sha256"],
        "robolab_commit": capture["robolab_commit"],
        "used_layer_dependencies": dependencies,
    }
    return {"capture": capture, "manifest": manifest, "binding": binding}


def _baseline_capture(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"native baseline capture is missing: {path}")
    if expected_sha256 is not None and _sha256(path) != expected_sha256:
        raise ValueError("native baseline capture bytes differ from plan binding")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "sgw-01-prospective-family-native-capture-v1" or value.get("status") != CAPTURE_STATUS:
        raise ValueError("baseline must be a prospective native capture receipt")
    if value.get("model_request_count") != 0 or value.get("behavioral_episode_count") != 0:
        raise ValueError("baseline capture must remain model blind")
    if value.get("receipt_sha256") != workspace_digest(value):
        raise ValueError("baseline capture receipt digest differs")
    required = {"rubiks_cube", "bowl"} | ({"plate"} if value.get("family") == "DIST" else set())
    objects = value.get("objects")
    if not isinstance(objects, Mapping) or not required.issubset(objects):
        raise ValueError("baseline capture lacks required scored objects")
    for name in required:
        _object_row(objects[name], name)
    return value


def _validated_capture_dependencies(capture: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Recheck the layer hashes recorded during baseline native capture."""

    rows = capture.get("usd_dependency_inventory")
    if not isinstance(rows, list) or not rows:
        raise ValueError("baseline capture lacks a hash-bound USD dependency inventory")
    expected = {
        str(Path(manifest["overlay_usda"]["path"]).resolve()): manifest["overlay_usda"]["sha256"],
        str(Path(manifest["base_scene"]["path"]).resolve()): manifest["base_scene"]["sha256"],
    }
    actual = {}
    validated = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("baseline capture has malformed USD dependency inventory")
        path = Path(str(row.get("real_path", "")))
        if not path.is_file() or row.get("sha256") != _sha256(path):
            raise ValueError("captured USD dependency bytes differ from the baseline receipt")
        actual[str(path.resolve())] = row["sha256"]
        validated.append({"real_path": str(path.resolve()), "sha256": row["sha256"], "bytes": path.stat().st_size})
    if any(actual.get(path) != digest for path, digest in expected.items()):
        raise ValueError("baseline capture dependency inventory does not bind its overlay and base scene")
    return validated


def _capture_dependencies_include(
    capture: Mapping[str, Any], overlay: Mapping[str, Any], inherited: list[Mapping[str, Any]],
) -> None:
    rows = capture.get("usd_dependency_inventory")
    if not isinstance(rows, list):
        raise ValueError("candidate capture lacks a USD dependency inventory")
    observed = {str(Path(row.get("real_path", "")).resolve()): row.get("sha256") for row in rows if isinstance(row, Mapping)}
    expected = {str(Path(overlay["path"]).resolve()): overlay["sha256"]}
    expected.update({str(Path(row["real_path"]).resolve()): row["sha256"] for row in inherited})
    if any(observed.get(path) != digest for path, digest in expected.items()):
        raise ValueError("candidate capture does not bind its overlay and inherited USD dependencies")


def _prospective_manifest(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"prospective baseline manifest is missing: {path}")
    if expected_sha256 is not None and _sha256(path) != expected_sha256:
        raise ValueError("prospective baseline manifest bytes differ from plan binding")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != BASELINE_STATUS or value.get("manifest_sha256") != _digest(value, "manifest_sha256"):
        raise ValueError("baseline overlay manifest is malformed")
    overlay = value.get("overlay_usda", {})
    if not isinstance(overlay, Mapping) or not Path(overlay.get("path", "")).is_file():
        raise ValueError("baseline overlay USD is missing")
    if _sha256(Path(overlay["path"])) != overlay.get("sha256"):
        raise ValueError("baseline overlay USD bytes differ from manifest")
    return value


def _candidate_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"candidate overlay manifest is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != CANDIDATE_STATUS or value.get("manifest_sha256") != _digest(value, "manifest_sha256"):
        raise ValueError("candidate overlay manifest is malformed")
    if not Path(value.get("overlay_usda", {}).get("path", "")).is_file():
        raise ValueError("candidate overlay USD is missing")
    if _sha256(Path(value["overlay_usda"]["path"])) != value["overlay_usda"].get("sha256"):
        raise ValueError("candidate overlay USD bytes differ from manifest")
    return value


def _translation(seed: int, family: str, index: int) -> list[float]:
    digest = hashlib.sha256(f"{seed}|{family}|{index}".encode()).digest()
    return [round(((digest[axis] / 255.0) * 2 - 1) * TRANSLATION_LIMIT_M, 6) for axis in (0, 1)]


def _translated_roots(objects: Mapping[str, Any], family: str, translation: list[float]) -> dict[str, dict[str, list[float]]]:
    names = ("rubiks_cube", "bowl") + (("plate",) if family == "DIST" else ())
    result = {}
    for name in names:
        row = _object_row(objects[name], name)
        root = row["root_position_env_local_xyz_m"]
        result[name] = {
            "position_m": [root[0] + translation[0], root[1] + translation[1], root[2]],
            "quaternion_wxyz": row["root_quaternion_world_wxyz"],
        }
    return result


def _geometric_rejection(
    capture: Mapping[str, Any], family: str, side: str, roots: Mapping[str, Any], translation: list[float], fingerprints: list[Mapping[str, Any]],
) -> str | None:
    if not all(math.isfinite(value) and abs(value) <= TRANSLATION_LIMIT_M for value in translation):
        return "translation_not_finite_or_outside_bounded_xy_range"
    table = capture["objects"].get("table")
    if not isinstance(table, Mapping):
        return "missing_measured_table_bounds"
    minimum = table.get("bbox_env_local_min_xyz_m")
    maximum = table.get("bbox_env_local_max_xyz_m")
    if not _finite_vector(minimum, 3) or not _finite_vector(maximum, 3):
        return "missing_finite_measured_table_bounds"
    for name, pose in roots.items():
        point = pose["position_m"]
        if not minimum[0] <= point[0] <= maximum[0] or not minimum[1] <= point[1] <= maximum[1]:
            return f"{name}_root_outside_measured_table_xy_bounds"
    for prior in fingerprints:
        if all(
            pose_error(
                Pose.from_json(roots[name]),
                Pose.from_json(prior["roots"][name]),
            )[0] <= 0.003
            and pose_error(
                Pose.from_json(roots[name]),
                Pose.from_json(prior["roots"][name]),
            )[1] <= 2
            for name in roots
        ):
            return "duplicate_layout_within_3mm_2deg"
    return None


def _object_row(row: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(row, Mapping):
        raise ValueError(f"baseline capture {name} object row is missing")
    for key, length in (
        ("root_position_env_local_xyz_m", 3),
        ("root_quaternion_world_wxyz", 4),
        ("geometric_center_offset_root_local_xyz_m", 3),
    ):
        if not _finite_vector(row.get(key), length):
            raise ValueError(f"baseline capture {name} lacks finite {key}")
    Pose.from_json({
        "position_m": row["root_position_env_local_xyz_m"],
        "quaternion_wxyz": row["root_quaternion_world_wxyz"],
    })
    return row


def _finite_vector(value: Any, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(isinstance(item, (int, float)) and math.isfinite(item) for item in value)


def _candidate_usda(manifest: Mapping[str, Any], capture: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    translation = row["translation_xy_m"]
    roots = row["authored_scored_object_roots"]
    specs = manifest["prospective_design"]["dimensions_and_poses"]
    lines = []
    for name, pose in roots.items():
        position, quaternion = pose["position_m"], pose["quaternion_wxyz"]
        lines.extend((
            f'    over "{name}" {{',
            f"        double3 xformOp:translate = ({position[0]}, {position[1]}, {position[2]})",
            f"        quatf xformOp:orient = ({quaternion[0]}, {quaternion[1]}, {quaternion[2]}, {quaternion[3]})",
            '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]',
            "    }",
        ))
    for spec in specs:
        if spec["name"] in roots:
            continue
        center = spec["center_m"]
        lines.extend((
            f'    over "{spec["name"]}" {{',
            f"        double3 xformOp:translate = ({center[0] + translation[0]}, {center[1] + translation[1]}, {center[2]})",
            "    }",
        ))
    source = manifest["overlay_usda"]["path"].replace("\\", "\\\\")
    return "#usda 1.0\n(\n    defaultPrim = \"World\"\n    subLayers = [@" + source + "@]\n)\n\nover \"World\" {\n" + "\n".join(lines) + "\n}\n"


def _validate_plan(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != PLAN_SCHEMA or value.get("plan_sha256") != _digest(value, "plan_sha256"):
        raise ValueError("prospective design plan is malformed")
    if value.get("status") != "prospective_design_plan_not_measured_or_qualified":
        raise ValueError("prospective design plan has an invalid status")
    if value.get("design_slot_count") != len(value.get("designs", ())) or not 1 <= value["design_slot_count"] <= MAX_DESIGNS:
        raise ValueError("prospective design plan has an invalid fixed slot count")


def _digest(value: Mapping[str, Any], field: str | None = None) -> str:
    material = dict(value)
    if field:
        material.pop(field, None)
    return hashlib.sha256((json.dumps(material, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--family", choices=("HEIGHT", "DIST"), required=True)
    parser.add_argument("--left-baseline-capture", type=Path, required=True)
    parser.add_argument("--right-baseline-capture", type=Path, required=True)
    parser.add_argument("--left-baseline-manifest", type=Path, required=True)
    parser.add_argument("--right-baseline-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--count", type=int, default=MAX_DESIGNS)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite prospective design plan: {args.output}")
    plan = build_design_plan(
        family=args.family, seed=args.seed, count=args.count,
        baseline_capture_paths={"left": args.left_baseline_capture, "right": args.right_baseline_capture},
        baseline_manifest_paths={"left": args.left_baseline_manifest, "right": args.right_baseline_manifest},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
