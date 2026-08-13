#!/usr/bin/env python3
"""Validate untouched R012 raw evidence after an origin-argument validator fix."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FROZEN_PATH = ROOT / "tools/validate_v3e006_r012.py"
SPEC = importlib.util.spec_from_file_location("v3e006_r012_frozen_validator", FROZEN_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load frozen validator: {FROZEN_PATH}")
frozen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frozen)

AMENDMENT = (
    ROOT
    / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r012"
    / "postexecution_validator_amendment_v1.json"
)
SCHEMA = "vla-wam-shared-v3e006-r012-postexecution-validator-amendment-v1"
STATUS = "validator_only_origin_argument_correction_after_execution_before_r012_closure"
FROZEN_SOURCE_GATE_SHA256 = "149358af6c730ac1b1fc87c52b4540aad8cb011787331a695195b18b521168e5"
RAW_RESULT_SHA256 = "a556be5fb67e95687ba390b689c3f7f676b20d21475fd2357ae1429c7b424c56"
FROZEN_HELPER = frozen._validate_live_tensor_geometry


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_binding(root: Path, row: Mapping[str, Any], label: str) -> Path:
    path = Path(str(row.get("path", "")))
    if not path.is_absolute():
        path = root / path
    frozen.require(
        path.is_file()
        and path.stat().st_size == row.get("bytes")
        and sha256(path) == row.get("sha256"),
        f"{label} binding differs: {path}",
    )
    return path


def validate_live_tensor_geometry(
    value: Mapping[str, Any],
    static: Mapping[str, Any],
    *,
    label: str,
    origin: np.ndarray | None = None,
) -> dict[str, Any]:
    """Cross-check the call-site origin and delegate the frozen geometry math."""

    pose = value.get("live_tensor_pose")
    frozen.require(isinstance(pose, Mapping), f"{label} tensor pose absent")
    embedded = frozen._finite_vector(
        pose.get("scene_env_origin_world_m"), 3, f"{label} env origin"
    )
    if origin is not None:
        supplied = np.asarray(origin, dtype=np.float64)
        frozen.require(
            supplied.shape == (3,)
            and np.isfinite(supplied).all()
            and np.array_equal(embedded, supplied),
            f"{label} call-site/embedded origin differs",
        )
    return FROZEN_HELPER(value, static, label=label)


def validate_amendment(root: Path) -> dict[str, Any]:
    value = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    frozen.require(value.get("schema_version") == SCHEMA, "validator amendment schema differs")
    frozen.require(value.get("status") == STATUS, "validator amendment status differs")
    frozen.require(value.get("model_request_count") == 0, "validator amendment model count differs")
    frozen.require(value.get("behavioral_episode_count") == 0, "validator amendment behavior count differs")
    frozen.require(value.get("accepted_state_candidate_count") == 0, "validator amendment accepted count differs")
    frozen.require(value.get("raw_result", {}).get("sha256") == RAW_RESULT_SHA256, "raw result digest differs")
    frozen.require(
        value.get("correction") == {
            "frozen_helper_signature": "_validate_live_tensor_geometry(value, static, *, label)",
            "frozen_preflight_call": "_validate_live_tensor_geometry(value, static, origin=origin, label=label)",
            "corrected_semantics": "require supplied origin bitwise equals retained live_tensor_pose.scene_env_origin_world_m, then delegate frozen geometry checks",
            "scientific_execution_change": False,
            "raw_evidence_change": False,
        },
        "validator correction contract differs",
    )
    source_gate = verify_binding(root, value["frozen_source_push_gate"], "frozen source gate")
    frozen.require(sha256(source_gate) == FROZEN_SOURCE_GATE_SHA256, "frozen source gate digest differs")
    for label in ("raw_result", "raw_harness", "raw_launch", "raw_runtime_log", "failed_target_validation"):
        verify_binding(root, value[label], label)
    return {"passed": True, "amendment": binding(AMENDMENT)}


def validate_candidate_root(root: Path, candidate_root: Path) -> dict[str, Any]:
    original = frozen._validate_live_tensor_geometry
    frozen._validate_live_tensor_geometry = validate_live_tensor_geometry
    try:
        return frozen.validate_candidate_root(root, candidate_root)
    finally:
        frozen._validate_live_tensor_geometry = original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, default=ROOT)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--verify-raw", action="store_true", required=True)
    args = parser.parse_args()
    root = args.study_root.resolve()
    result = frozen.validate_static(root, source_gate_required=True)
    result["postexecution_validator_amendment"] = validate_amendment(root)
    result["candidate_evidence"] = validate_candidate_root(root, args.candidate_root.resolve())
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
