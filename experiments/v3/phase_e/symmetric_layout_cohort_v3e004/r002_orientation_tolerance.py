"""Prospective V3-E004-R002 live-orientation and control-asset binding.

R002 is an instrumentation correction, not a scene or outcome change.  The
loader is deliberately pure so registration, queue, simulator, and compiler
paths all enforce the same hash-bound amendment before an affected request.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from .runtime_contract import RuntimeContractError, sha256_file


AMENDMENT_ID = "V3-E004-R002"
AMENDMENT_SCHEMA = "vla-wam-shared-v3e004-live-orientation-realisation-tolerance-amendment-v1"
ATTESTATION_SCHEMA = "vla-wam-shared-v3e004-r002-runtime-attestation-v1"
AMENDMENT_SHA256 = "d2540c5c5ad673a150de5f8d3a97e7adef977ebaa1a8f04d1a8ebab128c0c013"
ORIGINAL_TOLERANCE_RAD = 0.03490658503988659
CORRECTED_TOLERANCE_RAD = 0.04
ORIGINAL_CONTROL_ASSET_SHA256 = "22b95d601defb9252a448a1b37d0548e1f2dc3bfb0c3d86a04c2ef5c4cc817a0"
PAIRED_ASSET_SHA256 = "73ac533ccf505f9d71124b3da9aee328d3ebd32fffbd75bc88f32f0d15b20eec"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeContractError(message)


def _artifact(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    _require(resolved.is_file() and resolved.stat().st_size > 0, f"missing R002 artifact: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def load_amendment(
    path: Path,
    expected_sha256: str,
    *,
    registration_sha256: str,
    queue_sha256: str,
    candidate_sha256: str,
) -> dict[str, Any]:
    """Load and validate the committed R002 registration amendment."""

    artifact = _artifact(path)
    _require(expected_sha256 == AMENDMENT_SHA256, "R002 caller did not bind the registered amendment digest")
    _require(artifact["sha256"] == expected_sha256, "R002 amendment SHA-256 mismatch")
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeContractError(f"R002 amendment is not finite UTF-8 JSON: {exc}") from exc
    _require(isinstance(value, dict), "R002 amendment must be an object")
    _require(value.get("schema_version") == AMENDMENT_SCHEMA, "R002 amendment schema changed")
    _require(value.get("amendment_id") == AMENDMENT_ID, "R002 amendment id changed")
    _require(value.get("registered_before_new_s0_request") is True, "R002 was not registered prospectively")
    for key, expected in (
        ("registration_sha256", registration_sha256),
        ("queue_sha256", queue_sha256),
        ("candidate_sha256", candidate_sha256),
    ):
        _require(value.get(key) == expected, f"R002 differs for {key}")
    change = value.get("frozen_change")
    _require(isinstance(change, Mapping), "R002 frozen change is missing")
    original = change.get("original_live_orientation_realisation_tolerance_rad")
    corrected = change.get("corrected_live_orientation_realisation_tolerance_rad")
    _require(
        type(original) in (int, float)
        and math.isclose(float(original), ORIGINAL_TOLERANCE_RAD, abs_tol=1e-15),
        "R002 original orientation tolerance changed",
    )
    _require(
        type(corrected) in (int, float)
        and math.isclose(float(corrected), CORRECTED_TOLERANCE_RAD, abs_tol=1e-15),
        "R002 corrected orientation tolerance changed",
    )
    binding = value.get("control_asset_binding")
    _require(isinstance(binding, Mapping), "R002 control-asset binding is missing")
    for name in ("required_original_control_asset", "incorrect_substituted_asset"):
        record = binding.get(name)
        _require(isinstance(record, Mapping), f"R002 {name} binding is missing")
        digest = record.get("sha256")
        _require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            f"R002 {name} SHA-256 is invalid",
        )
        _require(type(record.get("bytes")) is int and record["bytes"] > 0, f"R002 {name} byte count is invalid")
    return value


def build_runtime_attestation(
    *,
    amendment: Mapping[str, Any],
    amendment_path: Path,
    amendment_sha256: str,
    control_scene_asset: Path,
    paired_scene_asset: Path,
    symmetry_level_s: float,
) -> dict[str, Any]:
    """Bind the effective tolerance and exact scene source bytes.

    Both assets are verified for every DROID layout.  This prevents the paired
    clutter USDA from being silently substituted as ``--control-scene-asset``
    even when an importer later hides its extra prim.
    """

    level = float(symmetry_level_s)
    _require(level in {0.0, 0.25, 0.5, 0.75, 1.0}, "R002 received an unregistered symmetry level")
    binding = amendment["control_asset_binding"]
    wanted_control = binding["required_original_control_asset"]
    wanted_paired = binding["incorrect_substituted_asset"]
    control = _artifact(control_scene_asset)
    paired = _artifact(paired_scene_asset)
    _require(control["sha256"] == wanted_control["sha256"], "R002 control scene is not the original control USDA")
    _require(control["bytes"] == wanted_control["bytes"], "R002 control scene byte count changed")
    _require(paired["sha256"] == wanted_paired["sha256"], "R002 paired scene USDA changed")
    _require(paired["bytes"] == wanted_paired["bytes"], "R002 paired scene byte count changed")
    _require(control["sha256"] != paired["sha256"], "R002 control and paired scene assets alias")
    amendment_record = _artifact(amendment_path)
    _require(amendment_record["sha256"] == amendment_sha256, "R002 amendment changed before runtime binding")
    return {
        "schema_version": ATTESTATION_SCHEMA,
        "amendment_id": AMENDMENT_ID,
        "amendment": amendment_record,
        "effective_live_orientation_realisation_tolerance_rad": CORRECTED_TOLERANCE_RAD,
        "symmetry_level_s": level,
        "control_scene_asset": control,
        "paired_scene_asset": paired,
        "s0_original_control_asset_binding_passed": True if math.isclose(level, 0.0, abs_tol=1e-12) else None,
        "asset_binding_passed": True,
        "applied_uniformly_to_droid_layouts": True,
        "selection_used_model_requests": 0,
        "pre_r002_rows_reclassified": False,
    }


def validate_runtime_attestation(
    record: Any,
    *,
    amendment_sha256: str,
    symmetry_level_s: float,
) -> dict[str, Any]:
    """Validate an R002 attestation retained by a live gate or episode."""

    _require(isinstance(record, Mapping), "R002 runtime attestation is missing")
    _require(record.get("schema_version") == ATTESTATION_SCHEMA, "R002 runtime attestation schema changed")
    _require(record.get("amendment_id") == AMENDMENT_ID, "R002 runtime attestation id changed")
    amendment = record.get("amendment")
    _require(isinstance(amendment, Mapping), "R002 runtime amendment artifact is missing")
    path = Path(str(amendment.get("path")))
    _require(path.is_file(), "R002 runtime amendment artifact is unavailable")
    _require(amendment.get("sha256") == amendment_sha256 == sha256_file(path), "R002 runtime amendment digest changed")
    _require(amendment_sha256 == AMENDMENT_SHA256, "R002 runtime amendment is not the registered artifact")
    _require(amendment.get("bytes") == path.stat().st_size, "R002 runtime amendment byte count changed")
    _require(
        math.isclose(
            float(record.get("effective_live_orientation_realisation_tolerance_rad")),
            CORRECTED_TOLERANCE_RAD,
            abs_tol=1e-15,
        ),
        "R002 effective orientation tolerance changed",
    )
    _require(math.isclose(float(record.get("symmetry_level_s")), float(symmetry_level_s), abs_tol=1e-12), "R002 symmetry level changed")
    _require(record.get("asset_binding_passed") is True, "R002 scene asset binding did not pass")
    if math.isclose(float(symmetry_level_s), 0.0, abs_tol=1e-12):
        _require(record.get("s0_original_control_asset_binding_passed") is True, "R002 s=0 control asset binding did not pass")
    for name in ("control_scene_asset", "paired_scene_asset"):
        artifact = record.get(name)
        _require(isinstance(artifact, Mapping), f"R002 {name} artifact is missing")
        asset = Path(str(artifact.get("path")))
        _require(asset.is_file() and artifact.get("sha256") == sha256_file(asset), f"R002 {name} changed")
        _require(artifact.get("bytes") == asset.stat().st_size, f"R002 {name} byte count changed")
    _require(
        record["control_scene_asset"]["sha256"] == ORIGINAL_CONTROL_ASSET_SHA256,
        "R002 runtime control scene is not the frozen original asset",
    )
    _require(
        record["paired_scene_asset"]["sha256"] == PAIRED_ASSET_SHA256,
        "R002 runtime paired scene is not the frozen companion asset",
    )
    return dict(record)
