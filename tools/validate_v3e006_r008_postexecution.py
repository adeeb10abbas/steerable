#!/usr/bin/env python3
"""Validate retained R008 evidence after a quaternion-representative fix.

The frozen R008 validator recomputes the registered SE(3) servo command and
compares its quaternion components directly with the action dispatched by the
runtime.  The runtime's frozen ``_command_base_link`` path first applies the
frozen OOD helper's deterministic quaternion sign canonicalization.  Therefore
some mathematically identical commands are stored as the exact antipode of the
validator's expected quaternion.

This additive wrapper requires the runtime action to equal the exact frozen
sign-canonicalized command at float32 precision.  It then replaces only that
quaternion representative in an in-memory copy and delegates every other
scientific, selection, provenance, and raw-evidence check to the frozen R008
validator.  It never modifies retained evidence.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v3.phase_e.canonical_stage_localization_v3e006.ood_reference import (  # noqa: E402
    _quat_normalize_wxyz,
)

_FROZEN_PATH = ROOT / "tools/validate_v3e006_r008.py"
_FROZEN_SPEC = importlib.util.spec_from_file_location(
    "v3e006_r008_frozen_validator", _FROZEN_PATH
)
if _FROZEN_SPEC is None or _FROZEN_SPEC.loader is None:
    raise RuntimeError(f"cannot load frozen validator: {_FROZEN_PATH}")
frozen = importlib.util.module_from_spec(_FROZEN_SPEC)
_FROZEN_SPEC.loader.exec_module(frozen)

AMENDMENT = (
    ROOT
    / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r008"
    / "postexecution_validator_amendment_v1.json"
)
SCHEMA = "vla-wam-shared-v3e006-r008-postexecution-validator-amendment-v1"
STATUS = "validator_only_quaternion_representative_correction_after_execution_before_r008_closure_or_behavior"
FROZEN_SOURCE_GATE_SHA256 = (
    "946f8f5eb7cdf0a2308f5925a41532e89267e222ee32629b586b1398e7d1c79b"
)
RAW_RESULT_SHA256 = "58f8e54994348634d8793f0448c49144b35ab3b601065ab4be298cbd9ebc6548"
EXPECTED_ANTIPODE_COUNTS = {
    (1, "canonical_grasp"): 0,
    (1, "canonical_carry"): 301,
    (2, "canonical_grasp"): 295,
    (2, "canonical_carry"): 263,
    (3, "canonical_grasp"): 0,
    (3, "canonical_carry"): 263,
    (4, "canonical_grasp"): 295,
    (4, "canonical_carry"): 301,
}
_FROZEN_VALIDATE_CANDIDATE_STATE = frozen.validate_candidate_state


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


def mathematical_action_float32(evidence: Mapping[str, Any]) -> list[float]:
    """Return the frozen validator's noncanonical quaternion representative."""

    position = np.asarray(evidence.get("command_base_position_world_m"), dtype=np.float64)
    quaternion = np.asarray(
        evidence.get("command_base_quaternion_world_wxyz"), dtype=np.float64
    )
    frozen.require(
        position.shape == (3,)
        and quaternion.shape == (4,)
        and np.isfinite(position).all()
        and np.isfinite(quaternion).all(),
        "servo mathematical command is malformed",
    )
    return np.concatenate((position, quaternion, [1.0])).astype(np.float32).tolist()


def runtime_action_float32(evidence: Mapping[str, Any]) -> list[float]:
    """Apply the exact frozen runtime sign canonicalizer before float32 cast."""

    position = np.asarray(evidence.get("command_base_position_world_m"), dtype=np.float64)
    quaternion = _quat_normalize_wxyz(
        evidence.get("command_base_quaternion_world_wxyz")
    )
    frozen.require(
        position.shape == (3,) and np.isfinite(position).all(),
        "servo runtime command position is malformed",
    )
    return np.concatenate((position, quaternion, [1.0])).astype(np.float32).tolist()


def same_float32(left: Sequence[float], right: Sequence[float]) -> bool:
    return bool(
        np.array_equal(
            np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)
        )
    )


def normalize_servo_actions_for_frozen_validator(
    state: Mapping[str, Any], *, label: str, candidate_rank: int
) -> dict[str, Any]:
    """Verify exact runtime actions and normalize only their SO(3) representative."""

    normalized = deepcopy(state)
    construction = normalized.get("construction")
    frozen.require(isinstance(construction, Mapping), f"{label} construction absent")
    trace = construction.get("construction_action_trace")
    servo = construction.get("object_space_servo_trace")
    frozen.require(
        isinstance(trace, list)
        and len(trace) == 1290
        and isinstance(servo, list)
        and len(servo) == 360
        and servo == trace[330:690],
        f"{label} original servo trace differs",
    )
    antipode_count = 0
    for index, servo_row in enumerate(servo):
        trace_row = trace[330 + index]
        evidence = servo_row.get("pre_action_object_space_servo")
        frozen.require(
            isinstance(evidence, Mapping),
            f"{label} servo {index + 1} evidence absent",
        )
        mathematical = mathematical_action_float32(evidence)
        runtime = runtime_action_float32(evidence)
        actual = servo_row.get("command_action_8d", [])
        frozen.require(
            same_float32(actual, runtime),
            f"{label} servo {index + 1} action differs from exact runtime canonicalization",
        )
        if not same_float32(mathematical, runtime):
            antipode_count += 1
            frozen.require(
                same_float32(mathematical[:3], runtime[:3])
                and same_float32(mathematical[7:], runtime[7:])
                and np.array_equal(
                    np.asarray(mathematical[3:7], dtype=np.float32),
                    -np.asarray(runtime[3:7], dtype=np.float32),
                ),
                f"{label} servo {index + 1} mismatch is not an exact quaternion antipode",
            )
        servo_row["command_action_8d"] = mathematical
        trace_row["command_action_8d"] = deepcopy(mathematical)
    frozen.require(
        antipode_count == EXPECTED_ANTIPODE_COUNTS[(candidate_rank, label)],
        f"{label} antipode count differs for rank {candidate_rank}",
    )
    return normalized


def validate_candidate_state(
    state: Mapping[str, Any],
    expected_stage: Mapping[str, Any],
    rank: int,
    schedule: Mapping[str, Any],
) -> None:
    label = str(expected_stage.get("stage", ""))
    frozen.require(
        label in ("canonical_grasp", "canonical_carry"),
        "candidate stage label differs",
    )
    normalized = normalize_servo_actions_for_frozen_validator(
        state, label=label, candidate_rank=rank
    )
    _FROZEN_VALIDATE_CANDIDATE_STATE(normalized, expected_stage, rank, schedule)


def validate_amendment(root: Path) -> dict[str, Any]:
    value = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    frozen.require(value.get("schema_version") == SCHEMA, "validator amendment schema differs")
    frozen.require(value.get("status") == STATUS, "validator amendment status differs")
    for key in (
        "model_request_count",
        "behavioral_episode_count",
        "accepted_state_candidate_count",
        "raw_evidence_mutation_count",
        "scientific_execution_change_count",
    ):
        frozen.require(value.get(key) == 0, f"validator amendment count differs: {key}")
    frozen.require(
        value.get("raw_result", {}).get("sha256") == RAW_RESULT_SHA256,
        "raw result digest differs",
    )
    frozen.require(
        value.get("correction")
        == {
            "runtime_path": "_command_base_link applies frozen _quat_normalize_wxyz before float32 action dispatch",
            "frozen_validator_defect": "compared noncanonical recomputed quaternion components directly with canonical runtime action components",
            "corrected_expectation": "canonicalize the recomputed expected quaternion with the exact frozen runtime helper before float32 action comparison",
            "so3_command_changed": False,
            "scientific_execution_change": False,
            "raw_evidence_change": False,
        },
        "validator correction contract differs",
    )
    frozen.require(
        value.get("observed_antipode_mismatch_counts")
        == [0, 301, 295, 263, 0, 263, 295, 301],
        "observed antipode profile differs",
    )
    frozen.require(value.get("observed_other_mismatch_count") == 0, "non-antipode mismatch exists")
    frozen.require(
        value.get("scientific_outcome")
        == {
            "status": "r008_candidate_budget_exhausted_no_valid_state_pair",
            "diagnostic_count": 4,
            "candidate_pair_count": 4,
            "accepted_candidate_rank": None,
        },
        "scientific outcome differs",
    )
    source_gate_path = verify_binding(
        root, value["frozen_source_push_gate"], "frozen source gate"
    )
    frozen.require(
        sha256(source_gate_path) == FROZEN_SOURCE_GATE_SHA256,
        "frozen source gate digest differs",
    )
    gate = json.loads(source_gate_path.read_text(encoding="utf-8"))
    inventory = {row["path"]: row for row in gate["implementation_files"]}
    for relative in (
        "experiments/v3/phase_e/canonical_stage_localization_v3e006_r008/object_servo.py",
        "experiments/v3/phase_e/canonical_stage_localization_v3e006_r008/state_repair_gate.py",
        "tools/validate_v3e006_r008.py",
        "tests/test_v3e006_r008.py",
    ):
        frozen.require(
            value["frozen_execution_inventory"][relative] == inventory[relative],
            f"frozen inventory differs: {relative}",
        )
        verify_binding(root, inventory[relative], f"frozen execution inventory {relative}")
    canonicalizer_path = verify_binding(
        root,
        value["runtime_quaternion_canonicalizer_source"],
        "runtime quaternion canonicalizer source",
    )
    frozen.require(
        canonicalizer_path
        == root
        / "experiments/v3/phase_e/canonical_stage_localization_v3e006/ood_reference.py"
        and sha256(canonicalizer_path)
        == "4df1ebf0061096a74b5eccd10b2a144e840f52fd50469b8bdae9369d1696fd04",
        "runtime quaternion canonicalizer identity differs",
    )
    for label, row in value["corrected_validator_inventory"].items():
        verify_binding(root, row, f"corrected validator inventory {label}")
    for label in (
        "raw_result",
        "raw_harness",
        "raw_launch",
        "raw_runtime_log",
        "failed_frozen_target_validation_receipt",
    ):
        verify_binding(root, value[label], label)
    implementation = str(value.get("correction_implementation_commit", ""))
    frozen.require(
        subprocess.run(
            ["git", "-C", str(root), "cat-file", "-e", f"{implementation}^{{commit}}"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0,
        "validator correction implementation commit absent",
    )
    frozen.require(
        subprocess.run(
            ["git", "-C", str(root), "merge-base", "--is-ancestor", implementation, "HEAD"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0,
        "validator correction implementation is not an ancestor",
    )
    return {"passed": True, "amendment": binding(AMENDMENT)}


def validate_candidate_root(root: Path, candidate_root: Path) -> dict[str, Any]:
    original = frozen.validate_candidate_state
    frozen.validate_candidate_state = validate_candidate_state
    try:
        return frozen.validate_candidate_root(root, candidate_root)
    finally:
        frozen.validate_candidate_state = original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, default=ROOT)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--verify-raw", action="store_true", required=True)
    args = parser.parse_args()
    root = args.study_root.resolve()
    result = frozen.validate_static(root, source_gate_required=True)
    result["postexecution_validator_amendment"] = validate_amendment(root)
    result["candidate_evidence"] = validate_candidate_root(
        root, args.candidate_root.resolve()
    )
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
