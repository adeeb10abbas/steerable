#!/usr/bin/env python3
"""Fail-closed artifact and runtime contract for Nano V3-B005."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


STUDY_ID = "vla_wam_language_steerability_v3"
AMENDMENT_ID = "V3-B005"
PHASE = "B_confound_ablation"
MODEL_ID = "cosmos3_nano_policy_droid"
MODEL_REPOSITORY = "nvidia/Cosmos3-Nano-Policy-DROID"
CHECKPOINT_REVISION = "6706d7680581c255ff61e0f3bb49d90eac55c79e"
COSMOS_REPOSITORY_COMMIT = "411d25b2e35bc441126f48c44a4b93e1c0564274"
ROBOLAB_REPOSITORY_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
PHASE = "B_confound_ablation"

SEEDS = tuple(range(9500, 9515))
LEVELS = (
    0.03658219039440155,
    0.06658219039440155,
    0.09658219039440155,
    0.12658219039440155,
    0.15658219039440155,
    0.18658219039440156,
    0.21658219039440155,
)
PROBE_LEVELS = (0, 3, 6)
RELATIONS = ("left", "right")
PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
ACTION_CHUNK_STEPS = 32
ACTION_DIM = 8
ACTION_CAP = 450
ACTION_SPACE = "joint_position_8d"
SETTLE_STEPS = 60
STABILITY_WINDOW_STEPS = 15
LINEAR_SPEED_TOLERANCE_M_S = 0.02
ANGULAR_SPEED_TOLERANCE_RAD_S = 0.20
SETTLE_OBJECTS = ("rubiks_cube", "bowl", "banana")
SETTLE_STEPS = 60
STABILITY_WINDOW_STEPS = 15
LINEAR_SPEED_TOLERANCE_M_S = 0.02
ANGULAR_SPEED_TOLERANCE_RAD_S = 0.20
SETTLE_OBJECTS = ("rubiks_cube", "bowl", "banana")
SUCCESS_PREDICATE_ID = "v2_frozen_droid_robolab_release_inside_45deg_requested_relation"

AMENDMENT_SCHEMA = "vla-wam-shared-v3b-nano-lateral-sweep-amendment-v2"
CELL_SCHEMA = "vla-wam-shared-v3b-nano-lateral-cell-v1"
MANIFEST_SCHEMA = "vla-wam-shared-v3b-nano-lateral-manifest-v1"
PHYSICAL_GATE_SCHEMA = "vla-wam-shared-v3b-nano-lateral-model-blind-calibration-v1"
SAFE_FIXTURE_SCHEMA = "vla-wam-shared-v3b-nano-lateral-safe-distractor-fixture-v1"
RUNTIME_SCHEMA = "vla-wam-shared-v3b005-nano-runtime-identity-v1"
FIXED_OBSERVATION_SCHEMA = "vla-wam-shared-v3b005-nano-fixed-observation-v1"
RELEASE_GATE_SCHEMA = "vla-wam-shared-v3b005-nano-behavioral-release-v1"
RESET_SCHEMA = "vla-wam-shared-v3b005-nano-reset-attestation-v1"
SETTLE_EVIDENCE_SCHEMA = "vla-wam-shared-v3b005-nano-settle-stability-v1"
SETTLE_EVIDENCE_SCHEMA = "vla-wam-shared-v3b005-nano-settle-stability-v1"

EXPECTED_SHA256 = {
    "amendment": "ff23475b53791c42715938d51a303e0ab82de88b1b8a7a30758c008c9919a47b",
    "safe_fixture": "87ff070be25b61538ead16ddbe06d2e9c155698ec2ea8acecbc30bd20b0197a5",
    "physical_gate": "60a065f24f76b0fe007a2455bf674dcde33204beb2f00dac1d930edd8f6542bf",
    "cells": "a770ae94274eaa85591a3ecd1f0f919b85dadc1c0ac3197c363b31659cb6b132",
    "manifest": "47c426f13146591d1a0bde60136e124eb5818cd8d44ef312f0f8fa82ad1623a1",
}
EXPECTED_FILENAMES = {
    "amendment": "post_result_nano_lateral_sweep_v3b005_amendment.json",
    "safe_fixture": "prospective_safe_distractor_fixture.json",
    "physical_gate": "model_blind_lateral_calibration_report.json",
    "cells": "nano_lateral_v3b005_cells.jsonl",
}
CONTRACT_FILES = (
    "experiments/v3/cosmos_nano_lateral_sweep/runtime_adapter.py",
    "experiments/v3/cosmos_nano_lateral_sweep/live_support.py",
    "experiments/v3/cosmos_nano_lateral_sweep/serve_nano.py",
    "experiments/v3/cosmos_nano_lateral_sweep/capture_fixed_observation.py",
    "experiments/v3/cosmos_nano_lateral_sweep/fixed_observation_gate.py",
    "experiments/v3/cosmos_nano_lateral_sweep/build_release_gate.py",
    "experiments/v3/cosmos_nano_lateral_sweep/fixture_tasks.py",
    "experiments/v3/cosmos_nano_lateral_sweep/task_files/left.py",
    "experiments/v3/cosmos_nano_lateral_sweep/task_files/right.py",
    "experiments/v3/cosmos_nano_lateral_sweep/live_client.py",
    "experiments/v3/cosmos_nano_lateral_sweep/robolab_bridge.py",
    "experiments/v3/cosmos_nano_lateral_sweep/compile_cell.py",
    "experiments/v3/cosmos_nano_lateral_sweep/compile_pair.py",
    "experiments/v3/cosmos_nano_lateral_sweep/queue.py",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RuntimeContractError(ValueError):
    """Raised before inference when a V3-B005 binding differs."""


def _fail(message: str) -> None:
    raise RuntimeContractError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeContractError(f"value is not finite canonical JSON: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    _fail(f"non-finite JSON constant is prohibited: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key is prohibited: {key}")
        result[key] = value
    return result


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeContractError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _exact(value: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            _fail(f"{label} mismatch for {key}")


def compute_contract_sha256(study_root: Path) -> str:
    root = Path(study_root).resolve()
    inventory = []
    for relative in CONTRACT_FILES:
        path = root / relative
        if not path.is_file():
            _fail(f"missing V3-B005 contract source: {relative}")
        inventory.append(
            {"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size}
        )
    return sha256_bytes(canonical_json_bytes(inventory))


# Compatibility spelling used by the earlier Nano Phase-B bridge.
compute_adapter_contract_sha256 = compute_contract_sha256


@dataclass(frozen=True)
class AuthorizedCell:
    row: dict[str, Any]
    cell_sha256: str

    @property
    def cell_id(self) -> str:
        return self.row["cell_id"]

    @property
    def seed(self) -> int:
        return self.row["environment_seed"]

    @property
    def level_index(self) -> int:
        return self.row["level_index"]

    @property
    def relation(self) -> str:
        return self.row["relation"]

    @property
    def arm(self) -> str:
        """Compatibility name used by the shared Phase-B bridge."""

        return f"level{self.level_index}"

    @property
    def level_y_m(self) -> float:
        return LEVELS[self.level_index]

    @property
    def fixture_id(self) -> str:
        return f"v3b005:nano:lateral_level{self.level_index}"

    @property
    def fixture_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes({
            "schema_version": "vla-wam-shared-v3b005-nano-level-fixture-fingerprint-v1",
            "safe_distractor_fixture_sha256": EXPECTED_SHA256["safe_fixture"],
            "physical_gate_sha256": EXPECTED_SHA256["physical_gate"],
            "level_index": self.level_index,
            "reference_object_initial_lateral_position_y_m": LEVELS[self.level_index],
        }))

@dataclass(frozen=True)
class ReleaseBundle:
    amendment: dict[str, Any]
    safe_fixture: dict[str, Any]
    physical_gate: dict[str, Any]
    manifest: dict[str, Any]
    cells: tuple[AuthorizedCell, ...]
    by_cell_id: dict[str, AuthorizedCell]
    hashes: dict[str, str]

    @property
    def manifest_sha256(self) -> str:
        return self.hashes["manifest"]

    @property
    def amendment_sha256(self) -> str:
        return self.hashes["amendment"]

    @property
    def cells_sha256(self) -> str:
        return self.hashes["cells"]

    @property
    def safe_fixture_sha256(self) -> str:
        return self.hashes["safe_fixture"]

    @property
    def physical_gate_sha256(self) -> str:
        return self.hashes["physical_gate"]

    @property
    def safe_fixture_sha256(self) -> str:
        return self.hashes["safe_fixture"]

    def cell(self, cell_id: str) -> AuthorizedCell:
        try:
            return self.by_cell_id[cell_id]
        except KeyError as exc:
            raise RuntimeContractError(
                f"cell is not in the exact V3-B005 210-cell registry: {cell_id}"
            ) from exc

    def release_fingerprint(self, cell: AuthorizedCell) -> str:
        return sha256_bytes(canonical_json_bytes({
            "schema_version": "vla-wam-shared-v3b005-nano-cell-fingerprint-v1",
            "study_id": STUDY_ID,
            "amendment_id": AMENDMENT_ID,
            "artifact_sha256": self.hashes,
            "cell_id": cell.cell_id,
            "cell_sha256": cell.cell_sha256,
            "fixture_id": cell.fixture_id,
            "fixture_sha256": cell.fixture_sha256,
            "prompt_sha256": cell.row["prompt_sha256"],
            "physical_gate_sha256": cell.row["physical_gate_sha256"],
            "safe_distractor_fixture_sha256": cell.row["safe_distractor_fixture_sha256"],
        }))


def _load_cells(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeContractError(f"cannot read V3-B005 cells: {exc}") from exc
    if len(lines) != 210 or any(not line for line in lines):
        _fail("V3-B005 registry must contain exactly 210 non-empty rows")
    rows = []
    for number, line in enumerate(lines, 1):
        try:
            value = json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except json.JSONDecodeError as exc:
            raise RuntimeContractError(f"invalid V3-B005 row {number}: {exc}") from exc
        if not isinstance(value, dict):
            _fail(f"V3-B005 row {number} must be an object")
        rows.append(value)
    return rows


def _validate_cell(row: dict[str, Any]) -> AuthorizedCell:
    seed = row.get("environment_seed")
    level = row.get("level_index")
    relation = row.get("relation")
    if type(seed) is not int or seed not in SEEDS:
        _fail("V3-B005 environment seeds must be exactly 9500..9514")
    if type(level) is not int or level not in range(7) or relation not in RELATIONS:
        _fail("V3-B005 conditions must be exactly seven levels by LEFT/RIGHT")
    cell_id = f"v3b005:nano:seed{seed}:level{level}:{relation}"
    _exact(row, {
        "schema_version": CELL_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "phase": "B_confound_ablation",
        "arena": "droid_robolab",
        "model_id": MODEL_ID,
        "cell_id": cell_id,
        "matched_block_id": f"v3b005:nano:seed{seed}:level{level}",
        "seed_block_id": f"v3b005:nano:seed{seed}",
        "environment_seed": seed,
        "sampling_seed": seed,
        "level_index": level,
        "reference_object_initial_lateral_position_y_m": LEVELS[level],
        "relation": relation,
        "prompt": PROMPTS[relation],
        "prompt_sha256": sha256_bytes(PROMPTS[relation].encode("utf-8")),
        "safe_distractor_fixture_sha256": EXPECTED_SHA256["safe_fixture"],
        "physical_gate_sha256": EXPECTED_SHA256["physical_gate"],
        "success_predicate_id": SUCCESS_PREDICATE_ID,
        "execution_status": "registered_after_physical_gate_runtime_and_fixed_observation_release_required",
        "valid_failure_policy": "retain every valid behavioral failure",
        "technical_invalidity_policy": "separate stream; repair only the identical registered cell",
    }, cell_id)
    order = row.get("execution_order_index_within_seed")
    if type(order) is not int or order not in range(1, 15):
        _fail(f"{cell_id} has invalid within-seed order")
    expected_randomization = (
        "14-seed cyclic Latin rotation"
        if seed < 9514
        else "prospectively specified SHA-256 order"
    )
    if row.get("randomization") != expected_randomization:
        _fail(f"{cell_id} randomization changed")
    runtime = row.get("runtime_identity_requirement")
    if not isinstance(runtime, Mapping):
        _fail(f"{cell_id} lacks runtime identity requirements")
    _exact(runtime, {
        "checkpoint_revision": CHECKPOINT_REVISION,
        "server_repository_commit": COSMOS_REPOSITORY_COMMIT,
        "robolab_commit": ROBOLAB_REPOSITORY_COMMIT,
        "fresh_runtime_and_fixed_observation_gate_required": True,
        "fixed_observation_gate_levels": list(PROBE_LEVELS),
    }, f"{cell_id}.runtime_identity_requirement")
    return AuthorizedCell(dict(row), sha256_bytes(canonical_json_bytes(row)))


def load_release_bundle(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str = EXPECTED_SHA256["manifest"],
) -> ReleaseBundle:
    """Load the exact committed, still-unreleased V3-B005 prospective queue."""

    manifest_path = Path(manifest_path).resolve()
    _sha(expected_manifest_sha256, "expected manifest SHA-256")
    if sha256_file(manifest_path) != expected_manifest_sha256:
        _fail("V3-B005 manifest does not match its committed SHA-256")
    if expected_manifest_sha256 != EXPECTED_SHA256["manifest"]:
        _fail("V3-B005 manifest must use the frozen committed SHA-256")
    base = manifest_path.parent
    paths = {key: base / name for key, name in EXPECTED_FILENAMES.items()}
    hashes = {"manifest": sha256_file(manifest_path)}
    for key, path in paths.items():
        if not path.is_file():
            _fail(f"missing exact V3-B005 {key} artifact: {path}")
        hashes[key] = sha256_file(path)
        if hashes[key] != EXPECTED_SHA256[key]:
            _fail(f"V3-B005 {key} artifact hash changed")

    manifest = load_json(manifest_path, "V3-B005 manifest")
    _exact(manifest, {
        "schema_version": MANIFEST_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "status": "hash_bound_after_passed_physical_gate_runtime_release_required",
        "amendment_sha256": EXPECTED_SHA256["amendment"],
        "safe_fixture_sha256": EXPECTED_SHA256["safe_fixture"],
        "behavioral_release": False,
    }, "V3-B005 manifest")
    if manifest.get("counts") != {
        "matched_seeds": 15, "levels": 7, "relations": 2, "registered_cells": 210
    }:
        _fail("V3-B005 manifest counts changed")
    _exact(manifest.get("physical_gate", {}), {
        "path": EXPECTED_FILENAMES["physical_gate"],
        "sha256": EXPECTED_SHA256["physical_gate"],
        "model_request_count": 0,
        "behavioral_episode_count": 0,
    }, "V3-B005 manifest physical gate")
    _exact(manifest.get("cells", {}), {
        "path": EXPECTED_FILENAMES["cells"],
        "sha256": EXPECTED_SHA256["cells"],
        "row_count": 210,
    }, "V3-B005 manifest cells")

    amendment = load_json(paths["amendment"], "V3-B005 amendment")
    _exact(amendment, {
        "schema_version": AMENDMENT_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "status": "design_frozen_before_v3b005_model_blind_gate_and_before_any_v3b005_model_request",
    }, "V3-B005 amendment")
    design = amendment.get("design", {})
    _exact(design, {
        "model_id": MODEL_ID,
        "arena": "droid_robolab",
        "factor": "reference_object_initial_lateral_position_y_m",
        "matched_seeds": list(SEEDS),
        "relations": list(RELATIONS),
        "exact_prompts": PROMPTS,
        "ordered_bowl_y_levels_m": list(LEVELS),
        "registered_behavioral_episode_ceiling_after_release": 210,
    }, "V3-B005 amendment design")
    _exact(amendment.get("runtime_identity_requirement", {}), {
        "checkpoint_revision": CHECKPOINT_REVISION,
        "server_repository_commit": COSMOS_REPOSITORY_COMMIT,
        "robolab_commit": ROBOLAB_REPOSITORY_COMMIT,
        "fresh_runtime_and_fixed_observation_gate_required": True,
        "fixed_observation_gate_levels": list(PROBE_LEVELS),
    }, "V3-B005 amendment runtime requirement")

    safe_fixture = load_json(paths["safe_fixture"], "V3-B005 safe fixture")
    _exact(safe_fixture, {
        "schema_version": SAFE_FIXTURE_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "status": "frozen_before_v3b005_model_blind_gate_and_before_any_v3b005_model_request",
        "model_request_count_before_registration": 0,
        "behavioral_episode_count_before_registration": 0,
        "ordered_bowl_y_levels_m": list(LEVELS),
    }, "V3-B005 safe fixture")

    physical = load_json(paths["physical_gate"], "V3-B005 physical gate")
    _exact(physical, {
        "schema_version": PHYSICAL_GATE_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "passed": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
    }, "V3-B005 physical gate")
    if (
        physical.get("dense_scan", {}).get("row_count") != 42
        or physical.get("dense_scan", {}).get("passing_candidate_y_m") != list(LEVELS)
        or physical.get("selection", {}).get("ordered_seven_levels_y_m") != list(LEVELS)
        or physical.get("selection", {}).get("banana_y_override_m") != -0.2755556747317314
    ):
        _fail("V3-B005 physical gate differs from the exact passed design")

    cells = tuple(_validate_cell(row) for row in _load_cells(paths["cells"]))
    if len({cell.cell_id for cell in cells}) != 210:
        _fail("V3-B005 cell IDs are not unique")
    conditions = {(level, relation) for level in range(7) for relation in RELATIONS}
    for seed in SEEDS:
        block = [cell for cell in cells if cell.seed == seed]
        if (
            len(block) != 14
            or {(cell.level_index, cell.relation) for cell in block} != conditions
            or {cell.row["execution_order_index_within_seed"] for cell in block}
            != set(range(1, 15))
        ):
            _fail(f"V3-B005 seed {seed} is not a complete ordered 14-cell block")
    return ReleaseBundle(
        amendment=amendment,
        safe_fixture=safe_fixture,
        physical_gate=physical,
        manifest=manifest,
        cells=cells,
        by_cell_id={cell.cell_id: cell for cell in cells},
        hashes=hashes,
    )


def verify_runtime_identity(
    runtime_manifest: Path,
    *,
    study_root: Path,
    release: ReleaseBundle,
) -> dict[str, Any]:
    runtime = load_json(runtime_manifest, "V3-B005 runtime identity")
    _exact(runtime, {
        "schema_version": RUNTIME_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "checkpoint_identifier": MODEL_REPOSITORY,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_hash_gate_passed": True,
        "external_repository_commit": COSMOS_REPOSITORY_COMMIT,
        "external_repository_diff_hash": EMPTY_SHA256,
        "simulator_repository_commit": ROBOLAB_REPOSITORY_COMMIT,
        "simulator_repository_diff_hash": EMPTY_SHA256,
        "prospective_artifact_sha256": release.hashes,
        "action_space": ACTION_SPACE,
        "action_chunk_shape": [ACTION_CHUNK_STEPS, ACTION_DIM],
        "open_loop_horizon": ACTION_CHUNK_STEPS,
        "action_cap": ACTION_CAP,
        "instruction_controller": "static",
        "fixed_observation_gate_levels": list(PROBE_LEVELS),
    }, "V3-B005 runtime identity")
    for key in (
        "checkpoint_sha256",
        "environment_lock_sha256",
        "v3b005_contract_sha256",
        "phase_a_runtime_identity_sha256",
        "phase_a_runtime_manifest_sha256",
        "runtime_identity_sha256",
    ):
        _sha(runtime.get(key), key)
    if runtime["v3b005_contract_sha256"] != compute_contract_sha256(study_root):
        _fail("V3-B005 runtime does not bind the checked-in contract stack")
    payload = {key: value for key, value in runtime.items() if key != "runtime_identity_sha256"}
    if runtime["runtime_identity_sha256"] != sha256_bytes(canonical_json_bytes(payload)):
        _fail("runtime_identity_sha256 does not bind the exact runtime fields")
    return runtime


def validate_reset_attestation(
    reset_attestation_path: Path,
    *,
    cell: AuthorizedCell,
    release: ReleaseBundle,
    runtime: Mapping[str, Any],
    behavioral_release_gate_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Validate a cell reset before the first behavioral model request."""

    reset = load_json(reset_attestation_path, "V3-B005 reset attestation")
    expected = {
        "schema_version": RESET_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "registered_cell_id": cell.cell_id,
        "matched_block_id": cell.row["matched_block_id"],
        "model_id": MODEL_ID,
        "level_index": cell.level_index,
        "relation": cell.relation,
        "environment_seed": cell.seed,
        "sampling_seed": cell.seed,
        "fixture_id": cell.fixture_id,
        "released_fixture_sha256": cell.fixture_sha256,
        "prompt_sha256": cell.row["prompt_sha256"],
        "release_fingerprint_sha256": release.release_fingerprint(cell),
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "model_request_count_before_attestation": 0,
        "neutral_reset_passed": True,
        "released_fixture_match_passed": True,
        "viewport_writer_preflight_passed": True,
        "raw_output_preflight_passed": True,
        "model_blind_settle_gate_passed": True,
        "settle_steps": SETTLE_STEPS,
        "stable_window_steps": STABILITY_WINDOW_STEPS,
        "linear_speed_tolerance_m_s": LINEAR_SPEED_TOLERANCE_M_S,
        "angular_speed_tolerance_rad_s": ANGULAR_SPEED_TOLERANCE_RAD_S,
        "episode_length_buf_reset_passed": True,
    }
    if behavioral_release_gate_sha256 is not None:
        expected["behavioral_release_gate_sha256"] = _sha(
            behavioral_release_gate_sha256,
            "behavioral release gate SHA-256",
        )
    _exact(reset, expected, "V3-B005 reset attestation")
    for key in (
        "physical_reset_sha256",
        "initial_state_sha256",
        "fixture_match_evidence_sha256",
    ):
        _sha(reset.get(key), f"V3-B005 reset attestation.{key}")
    settle_value = reset.get("settle_stability_evidence_path")
    if not isinstance(settle_value, str) or not settle_value:
        _fail("V3-B005 reset attestation lacks settle evidence path")
    settle_path = Path(settle_value).resolve()
    settle_sha256 = _sha(
        reset.get("settle_stability_evidence_sha256"),
        "V3-B005 reset attestation.settle_stability_evidence_sha256",
    )
    if not settle_path.is_file() or sha256_file(settle_path) != settle_sha256:
        _fail("V3-B005 settle evidence file binding changed")
    validate_settle_stability_evidence(
        load_json(settle_path, "V3-B005 settle evidence"),
        cell=cell,
    )
    fixture_value = reset.get("fixture_match_evidence_path")
    if not isinstance(fixture_value, str) or not fixture_value:
        _fail("V3-B005 reset attestation lacks fixture evidence path")
    fixture_path = Path(fixture_value).resolve()
    if (
        not fixture_path.is_file()
        or sha256_file(fixture_path) != reset["fixture_match_evidence_sha256"]
    ):
        _fail("V3-B005 fixture evidence file binding changed")
    return reset, sha256_bytes(canonical_json_bytes(reset))


def validate_settle_stability_evidence(
    evidence: Mapping[str, Any],
    *,
    cell: AuthorizedCell,
    runner_reset_contract_complete: bool = True,
) -> dict[str, Any]:
    """Verify the model-blind 60+15-step stability evidence."""

    if not isinstance(evidence, Mapping):
        _fail("settle/stability evidence must be an object")
    _exact(evidence, {
        "schema_version": SETTLE_EVIDENCE_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "registered_cell_id": cell.cell_id,
        "settle_steps": SETTLE_STEPS,
        "stable_window_steps": STABILITY_WINDOW_STEPS,
        "linear_speed_tolerance_m_s": LINEAR_SPEED_TOLERANCE_M_S,
        "angular_speed_tolerance_rad_s": ANGULAR_SPEED_TOLERANCE_RAD_S,
        "hold_action_shape": [1, ACTION_DIM],
        "terminated_or_truncated_during_gate": False,
        "neutral_after_settle": True,
        "episode_length_buf_reset_passed": True,
        "episode_length_buf_before_reset": [SETTLE_STEPS + STABILITY_WINDOW_STEPS],
        "episode_length_buf_after_reset": [0],
        "model_request_count_during_gate": 0,
        "runner_pre_action_reset_calls": 2 if runner_reset_contract_complete else 1,
        "physical_reset_calls": 1,
        "settle_gate_runs": 1,
        "duplicate_second_reset_idempotent": runner_reset_contract_complete,
    }, "V3-B005 settle/stability evidence")
    maxima = evidence.get("stability_window_component_maxima")
    velocities = evidence.get("post_settle_velocities")
    positions = evidence.get("post_settle_positions_world_xyz_m")
    quaternions = evidence.get("post_settle_quaternions_world_wxyz")
    for label, value in (
        ("stability_window_component_maxima", maxima),
        ("post_settle_velocities", velocities),
        ("post_settle_positions_world_xyz_m", positions),
        ("post_settle_quaternions_world_wxyz", quaternions),
    ):
        if not isinstance(value, Mapping) or set(value) != set(SETTLE_OBJECTS):
            _fail(f"V3-B005 settle evidence {label} must cover all three objects")
    for name in SETTLE_OBJECTS:
        maximum = maxima[name]
        if not isinstance(maximum, Mapping):
            _fail(f"V3-B005 settle maximum for {name} must be an object")
        linear = maximum.get("max_linear_component_speed_m_s")
        angular = maximum.get("max_angular_component_speed_rad_s")
        if (
            isinstance(linear, bool)
            or not isinstance(linear, (int, float))
            or not np.isfinite(linear)
            or not 0 <= linear <= LINEAR_SPEED_TOLERANCE_M_S
        ):
            _fail(f"{name} exceeded the V3-B005 linear stability threshold")
        if (
            isinstance(angular, bool)
            or not isinstance(angular, (int, float))
            or not np.isfinite(angular)
            or not 0 <= angular <= ANGULAR_SPEED_TOLERANCE_RAD_S
        ):
            _fail(f"{name} exceeded the V3-B005 angular stability threshold")
        for label, value, length in (
            ("velocity", velocities[name], 6),
            ("position", positions[name], 3),
            ("quaternion", quaternions[name], 4),
        ):
            array = np.asarray(value, dtype=np.float64)
            if array.shape != (length,) or not np.isfinite(array).all():
                _fail(f"{name} post-settle {label} is malformed")
    return dict(evidence)


def validate_settle_stability_evidence(
    evidence: Mapping[str, Any],
    *,
    cell: AuthorizedCell,
    runner_reset_contract_complete: bool = True,
) -> dict[str, Any]:
    """Verify the model-blind settle gate retained before inference."""

    _exact(evidence, {
        "schema_version": SETTLE_EVIDENCE_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "registered_cell_id": cell.cell_id,
        "settle_steps": SETTLE_STEPS,
        "stable_window_steps": STABILITY_WINDOW_STEPS,
        "linear_speed_tolerance_m_s": LINEAR_SPEED_TOLERANCE_M_S,
        "angular_speed_tolerance_rad_s": ANGULAR_SPEED_TOLERANCE_RAD_S,
        "hold_action_shape": [1, ACTION_DIM],
        "terminated_or_truncated_during_gate": False,
        "neutral_after_settle": True,
        "episode_length_buf_reset_passed": True,
        "episode_length_buf_before_reset": [SETTLE_STEPS + STABILITY_WINDOW_STEPS],
        "episode_length_buf_after_reset": [0],
        "model_request_count_during_gate": 0,
        "runner_pre_action_reset_calls": 2 if runner_reset_contract_complete else 1,
        "physical_reset_calls": 1,
        "settle_gate_runs": 1,
        "duplicate_second_reset_idempotent": runner_reset_contract_complete,
    }, "V3-B005 settle/stability evidence")
    for label in (
        "stability_window_component_maxima",
        "post_settle_velocities",
        "post_settle_positions_world_xyz_m",
        "post_settle_quaternions_world_wxyz",
    ):
        value = evidence.get(label)
        if not isinstance(value, Mapping) or set(value) != set(SETTLE_OBJECTS):
            _fail(f"settle/stability evidence {label} must cover all released objects")
    maxima = evidence["stability_window_component_maxima"]
    velocities = evidence["post_settle_velocities"]
    positions = evidence["post_settle_positions_world_xyz_m"]
    quaternions = evidence["post_settle_quaternions_world_wxyz"]
    for name in SETTLE_OBJECTS:
        row = maxima[name]
        if not isinstance(row, Mapping):
            _fail(f"settle/stability maximum for {name} must be an object")
        linear = row.get("max_linear_component_speed_m_s")
        angular = row.get("max_angular_component_speed_rad_s")
        if (
            not isinstance(linear, (int, float))
            or isinstance(linear, bool)
            or not np.isfinite(linear)
            or not 0 <= linear <= LINEAR_SPEED_TOLERANCE_M_S
        ):
            _fail(f"{name} exceeded the released linear-speed threshold")
        if (
            not isinstance(angular, (int, float))
            or isinstance(angular, bool)
            or not np.isfinite(angular)
            or not 0 <= angular <= ANGULAR_SPEED_TOLERANCE_RAD_S
        ):
            _fail(f"{name} exceeded the released angular-speed threshold")
        for label, values, length in (
            ("velocity", velocities[name], 6),
            ("position", positions[name], 3),
            ("quaternion", quaternions[name], 4),
        ):
            array = np.asarray(values, dtype=np.float64)
            if array.shape != (length,) or not np.isfinite(array).all():
                _fail(f"{name} post-settle {label} is malformed")
    return dict(evidence)
