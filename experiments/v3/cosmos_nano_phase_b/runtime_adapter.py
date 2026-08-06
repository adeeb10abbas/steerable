#!/usr/bin/env python3
"""Fail-closed runtime authorization for the V3-B001 Cosmos3 Nano mirror.

This module does not import or contact a model server.  It verifies the
prospective release, the exact runtime identity, and a cell-specific live-reset
attestation before returning an adapter that is allowed to call an injected
transport.  The transport boundary makes the important ordering testable: a
release or reset mismatch fails before any policy request can be issued.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


STUDY_ID = "vla_wam_language_steerability_v3"
AMENDMENT_ID = "V3-B001"
PHASE = "B_confound_ablation"
MODEL_ID = "cosmos3_nano_policy_droid"
MODEL_REPOSITORY = "nvidia/Cosmos3-Nano-Policy-DROID"
CHECKPOINT_REVISION = "6706d7680581c255ff61e0f3bb49d90eac55c79e"
COSMOS_REPOSITORY_COMMIT = "411d25b2e35bc441126f48c44a4b93e1c0564274"
ROBOLAB_REPOSITORY_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

SEEDS = tuple(range(9400, 9427))
ARMS = ("control", "position_mirrored")
RELATIONS = ("left", "right")
PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
ACTION_CHUNK_STEPS = 32
ACTION_DIM = 8
ACTION_CAP = 450
ACTION_SPACE = "joint_position_8d"
MIRROR_FACTOR = "movable_object_center_position_reflection_about_robot_sagittal_plane"
SETTLE_STEPS = 60
STABILITY_WINDOW_STEPS = 15
LINEAR_SPEED_TOLERANCE_M_S = 0.02
ANGULAR_SPEED_TOLERANCE_RAD_S = 0.20
SETTLE_OBJECTS = ("rubiks_cube", "bowl", "banana")
SUCCESS_PREDICATE_ID = (
    "v2_frozen_droid_robolab_release_inside_45deg_requested_relation"
)

AMENDMENT_SCHEMA = "vla-wam-shared-v3b-nano-mirror-amendment-v1"
CELL_SCHEMA = "vla-wam-shared-v3b-nano-mirror-cell-v1"
MANIFEST_SCHEMA = "vla-wam-shared-v3b-nano-mirror-manifest-v1"
RUNTIME_SCHEMA = "vla-wam-shared-v3b-nano-runtime-identity-v1"
RESET_SCHEMA = "vla-wam-shared-v3b-nano-reset-attestation-v1"
SETTLE_EVIDENCE_SCHEMA = "vla-wam-shared-v3b-nano-settle-stability-v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

ADAPTER_CONTRACT_FILES = (
    "experiments/v3/cosmos_nano_phase_b/runtime_adapter.py",
    "experiments/v3/cosmos_nano_phase_b/compile_cell.py",
)


class RuntimeContractError(ValueError):
    """Raised before a model request when any prospective identity differs."""


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


def release_json_bytes(value: Any) -> bytes:
    """Match the release builder's pretty canonical object serialization."""

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
    except (TypeError, ValueError) as exc:
        raise RuntimeContractError(f"release value is not finite canonical JSON: {exc}") from exc


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


def _file_record(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        _fail(f"required release file is missing or empty: {path}")
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def compute_adapter_contract_sha256(study_root: Path) -> str:
    root = Path(study_root).resolve()
    inventory: list[dict[str, str]] = []
    for relative in ADAPTER_CONTRACT_FILES:
        path = root / relative
        if not path.is_file():
            _fail(f"missing Phase-B adapter source: {relative}")
        inventory.append({"path": relative, "sha256": sha256_file(path)})
    return sha256_bytes(canonical_json_bytes(inventory))


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
    def arm(self) -> str:
        return self.row["arm"]

    @property
    def relation(self) -> str:
        return self.row["relation"]


@dataclass(frozen=True)
class ReleaseBundle:
    amendment: dict[str, Any]
    manifest: dict[str, Any]
    cells: tuple[AuthorizedCell, ...]
    by_cell_id: dict[str, AuthorizedCell]
    amendment_sha256: str
    cells_sha256: str
    manifest_sha256: str

    def cell(self, cell_id: str) -> AuthorizedCell:
        try:
            return self.by_cell_id[cell_id]
        except KeyError as exc:
            raise RuntimeContractError(f"cell is not in the released 108-cell registry: {cell_id}") from exc

    def release_fingerprint(self, cell: AuthorizedCell) -> str:
        payload = {
            "schema_version": "vla-wam-shared-v3b-nano-cell-release-fingerprint-v1",
            "study_id": STUDY_ID,
            "amendment_id": AMENDMENT_ID,
            "release_manifest_sha256": self.manifest_sha256,
            "amendment_sha256": self.amendment_sha256,
            "cells_sha256": self.cells_sha256,
            "cell_id": cell.cell_id,
            "cell_sha256": cell.cell_sha256,
            "fixture_sha256": cell.row["fixture_sha256"],
            "prompt_sha256": cell.row["prompt_sha256"],
        }
        return sha256_bytes(canonical_json_bytes(payload))


def _resolve_sibling(manifest_path: Path, file_value: Any, label: str) -> Path:
    if not isinstance(file_value, str) or not file_value or Path(file_value).name != file_value:
        _fail(f"{label}.path must name a sibling release file")
    path = (manifest_path.parent / file_value).resolve()
    if path.parent != manifest_path.parent.resolve():
        _fail(f"{label}.path escapes the release directory")
    return path


def _validate_release_row(row: dict[str, Any], amendment_sha256: str) -> AuthorizedCell:
    if not isinstance(row, dict):
        _fail("every release row must be a JSON object")
    seed = row.get("environment_seed")
    arm = row.get("arm")
    relation = row.get("relation")
    if type(seed) is not int or seed not in SEEDS:
        _fail("Phase-B Nano seeds must be exactly 9400..9426")
    if arm not in ARMS or relation not in RELATIONS:
        _fail("Phase-B Nano conditions are exactly control/position_mirrored x left/right")
    expected_cell_id = f"v3b001:nano:seed{seed}:{arm}:{relation}"
    expected_block = f"v3b001:nano:seed{seed}"
    _exact(
        row,
        {
            "schema_version": CELL_SCHEMA,
            "study_id": STUDY_ID,
            "amendment_id": AMENDMENT_ID,
            "amendment_sha256": amendment_sha256,
            "phase": PHASE,
            "arena": "droid_robolab",
            "model_id": MODEL_ID,
            "cell_id": expected_cell_id,
            "matched_block_id": expected_block,
            "arm": arm,
            "relation": relation,
            "environment_seed": seed,
            "sampling_seed": seed,
            "factor": MIRROR_FACTOR,
            "prompt_family": "direct_command",
            "prompt": PROMPTS[relation],
            "prompt_sha256": sha256_bytes(PROMPTS[relation].encode("utf-8")),
            "success_predicate_id": SUCCESS_PREDICATE_ID,
            "execution_status": (
                "authorized_after_v3b001_calibration_with_live_identity_and_output_gate_recheck"
            ),
        },
        expected_cell_id,
    )
    if type(row.get("execution_order_index_within_seed")) is not int or not 1 <= row["execution_order_index_within_seed"] <= 4:
        _fail(f"{expected_cell_id} has invalid within-seed order")
    for key in ("fixture_id", "fixture_sha256", "randomization_key_sha256"):
        if key.endswith("sha256"):
            _sha(row.get(key), f"{expected_cell_id}.{key}")
        elif not isinstance(row.get(key), str) or not row[key]:
            _fail(f"{expected_cell_id}.{key} is required")
    runtime = row.get("runtime_identity_requirement")
    if not isinstance(runtime, dict):
        _fail(f"{expected_cell_id} lacks runtime identity requirements")
    _exact(
        runtime,
        {
            "model_repository": MODEL_REPOSITORY,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "server_repository_commit": COSMOS_REPOSITORY_COMMIT,
            "robolab_repository_commit": ROBOLAB_REPOSITORY_COMMIT,
            "clean_external_repositories_required": True,
        },
        f"{expected_cell_id}.runtime_identity_requirement",
    )
    outputs = row.get("required_raw_outputs")
    if not isinstance(outputs, list) or set(outputs) != {
        "viewport_video",
        "executed_action_trace",
        "raw_result_jsonl",
        "every_exposed_decoded_future",
    }:
        _fail(f"{expected_cell_id} raw-output contract changed")
    fields = row.get("required_episode_fields")
    if not isinstance(fields, dict) or not {
        "signed_final_lateral_offset_m",
        "final_requested_signed_margin_m",
        "requested_success",
        "failure_class",
    }.issubset(fields):
        _fail(f"{expected_cell_id} measurement contract is incomplete")
    return AuthorizedCell(row=dict(row), cell_sha256=sha256_bytes(canonical_json_bytes(row)))


def load_release_bundle(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
) -> ReleaseBundle:
    """Load only the externally hash-pinned, exact 108-cell V3-B001 release."""

    manifest_path = Path(manifest_path).resolve()
    expected_manifest_sha256 = _sha(expected_manifest_sha256, "expected release manifest hash")
    if not manifest_path.is_file() or sha256_file(manifest_path) != expected_manifest_sha256:
        _fail("release manifest does not match the externally pinned SHA-256")
    manifest = load_json(manifest_path, "release manifest")
    _exact(
        manifest,
        {
            "schema_version": MANIFEST_SCHEMA,
            "study_id": STUDY_ID,
            "amendment_id": AMENDMENT_ID,
            "status": "hash_bound_release_ready",
        },
        "release manifest",
    )
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {"amendment", "cells"}:
        _fail("release manifest must bind exactly amendment and cells")
    amendment_path = _resolve_sibling(manifest_path, files["amendment"].get("path"), "amendment")
    cells_path = _resolve_sibling(manifest_path, files["cells"].get("path"), "cells")
    for label, path in (("amendment", amendment_path), ("cells", cells_path)):
        record = files[label]
        if not isinstance(record, dict):
            _fail(f"release manifest {label} record must be an object")
        observed = _file_record(path)
        if any(record.get(key) != observed[key] for key in ("path", "sha256", "bytes")):
            _fail(f"release manifest {label} hash/size binding changed")
    if files["cells"].get("row_count") != 108:
        _fail("release manifest must bind exactly 108 cells")

    amendment = load_json(amendment_path, "release amendment")
    amendment_hash = sha256_file(amendment_path)
    _exact(
        amendment,
        {
            "schema_version": AMENDMENT_SCHEMA,
            "study_id": STUDY_ID,
            "amendment_id": AMENDMENT_ID,
            "phase": PHASE,
            "status": "released_after_model_blind_calibration_before_any_phase_b_model_request",
            "exact_prompts": PROMPTS,
        },
        "release amendment",
    )
    model_identity = amendment.get("model_identity")
    if not isinstance(model_identity, dict):
        _fail("release amendment lacks model identity")
    _exact(
        model_identity,
        {
            "model_id": MODEL_ID,
            "model_repository": MODEL_REPOSITORY,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "server_repository_commit": COSMOS_REPOSITORY_COMMIT,
            "robolab_repository_commit": ROBOLAB_REPOSITORY_COMMIT,
        },
        "release model identity",
    )
    design = amendment.get("design")
    if not isinstance(design, dict):
        _fail("release amendment lacks design")
    _exact(
        design,
        {
            "arms": list(ARMS),
            "directions": list(RELATIONS),
            "seeds": list(SEEDS),
            "matched_seed_count": 27,
            "cells_per_seed": 4,
            "behavioral_cell_count": 108,
        },
        "release design",
    )
    fixtures = amendment.get("fixtures")
    if not isinstance(fixtures, dict) or set(fixtures) != set(ARMS):
        _fail("release amendment must bind exactly control and position_mirrored fixtures")
    fixture_bindings: dict[str, tuple[str, str]] = {}
    for arm, fixture in fixtures.items():
        if not isinstance(fixture, dict) or not isinstance(fixture.get("fixture_id"), str):
            _fail(f"release fixture {arm} is incomplete")
        fixture_bindings[arm] = (
            fixture["fixture_id"],
            sha256_bytes(release_json_bytes(fixture)),
        )

    cells: list[AuthorizedCell] = []
    try:
        lines = cells_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeContractError(f"cannot read released cells: {exc}") from exc
    if len(lines) != 108 or any(not line.strip() for line in lines):
        _fail("released cell registry must contain exactly 108 non-empty rows")
    for line_number, line in enumerate(lines, 1):
        try:
            row = json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except json.JSONDecodeError as exc:
            raise RuntimeContractError(f"invalid release row {line_number}: {exc}") from exc
        cells.append(_validate_release_row(row, amendment_hash))
    if len({cell.cell_id for cell in cells}) != 108:
        _fail("released cell IDs are not unique")
    expected_conditions = {(arm, relation) for arm in ARMS for relation in RELATIONS}
    for seed in SEEDS:
        seed_cells = [cell for cell in cells if cell.seed == seed]
        if (
            len(seed_cells) != 4
            or {(cell.arm, cell.relation) for cell in seed_cells} != expected_conditions
            or {cell.row["execution_order_index_within_seed"] for cell in seed_cells} != {1, 2, 3, 4}
        ):
            _fail(f"seed {seed} is not a complete matched four-cell block")
    for cell in cells:
        expected_fixture_id, expected_fixture_hash = fixture_bindings[cell.arm]
        if (
            cell.row["fixture_id"] != expected_fixture_id
            or cell.row["fixture_sha256"] != expected_fixture_hash
        ):
            _fail(f"{cell.cell_id} does not bind its released fixture")
    return ReleaseBundle(
        amendment=amendment,
        manifest=manifest,
        cells=tuple(cells),
        by_cell_id={cell.cell_id: cell for cell in cells},
        amendment_sha256=amendment_hash,
        cells_sha256=sha256_file(cells_path),
        manifest_sha256=expected_manifest_sha256,
    )


def verify_runtime_identity(
    runtime_manifest_path: Path,
    *,
    study_root: Path,
    release: ReleaseBundle,
) -> dict[str, Any]:
    runtime = load_json(runtime_manifest_path, "Phase-B runtime identity")
    _exact(
        runtime,
        {
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
            "release_manifest_sha256": release.manifest_sha256,
            "action_space": ACTION_SPACE,
            "action_chunk_shape": [ACTION_CHUNK_STEPS, ACTION_DIM],
            "open_loop_horizon": ACTION_CHUNK_STEPS,
            "action_cap": ACTION_CAP,
            "instruction_controller": "static",
        },
        "Phase-B runtime identity",
    )
    for key in (
        "checkpoint_sha256",
        "environment_lock_sha256",
        "phase_b_adapter_contract_sha256",
        "runtime_identity_sha256",
    ):
        _sha(runtime.get(key), key)
    if runtime["phase_b_adapter_contract_sha256"] != compute_adapter_contract_sha256(study_root):
        _fail("Phase-B adapter contract hash does not match checked-in runtime/compiler sources")
    payload = {key: value for key, value in runtime.items() if key != "runtime_identity_sha256"}
    if runtime["runtime_identity_sha256"] != sha256_bytes(canonical_json_bytes(payload)):
        _fail("runtime_identity_sha256 does not bind the exact runtime fields")
    return runtime


def validate_settle_stability_evidence(
    evidence: Mapping[str, Any],
    *,
    cell: AuthorizedCell,
    runner_reset_contract_complete: bool = True,
) -> dict[str, Any]:
    """Verify the released model-blind settling gate without simulator imports."""

    if not isinstance(evidence, Mapping):
        _fail("settle/stability evidence must be an object")
    _exact(
        evidence,
        {
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
        },
        "settle/stability evidence",
    )
    maxima = evidence.get("stability_window_component_maxima")
    final_velocities = evidence.get("post_settle_velocities")
    positions = evidence.get("post_settle_positions_world_xyz_m")
    quaternions = evidence.get("post_settle_quaternions_world_wxyz")
    for label, value in (
        ("stability_window_component_maxima", maxima),
        ("post_settle_velocities", final_velocities),
        ("post_settle_positions_world_xyz_m", positions),
        ("post_settle_quaternions_world_wxyz", quaternions),
    ):
        if not isinstance(value, Mapping) or set(value) != set(SETTLE_OBJECTS):
            _fail(f"settle/stability evidence {label} must cover the three released objects")
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
            or linear < 0
            or linear > LINEAR_SPEED_TOLERANCE_M_S
        ):
            _fail(f"{name} exceeded the released linear-speed stability threshold")
        if (
            not isinstance(angular, (int, float))
            or isinstance(angular, bool)
            or not np.isfinite(angular)
            or angular < 0
            or angular > ANGULAR_SPEED_TOLERANCE_RAD_S
        ):
            _fail(f"{name} exceeded the released angular-speed stability threshold")
        for label, values, length in (
            ("post-settle velocity", final_velocities[name], 6),
            ("post-settle position", positions[name], 3),
            ("post-settle quaternion", quaternions[name], 4),
        ):
            array = np.asarray(values, dtype=np.float64)
            if array.shape != (length,) or not np.isfinite(array).all():
                _fail(f"{name} {label} is malformed")
    return dict(evidence)


def validate_reset_attestation(
    reset_attestation_path: Path,
    *,
    cell: AuthorizedCell,
    release: ReleaseBundle,
    runtime: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    reset = load_json(reset_attestation_path, "live reset attestation")
    expected_release = release.release_fingerprint(cell)
    _exact(
        reset,
        {
            "schema_version": RESET_SCHEMA,
            "study_id": STUDY_ID,
            "amendment_id": AMENDMENT_ID,
            "registered_cell_id": cell.cell_id,
            "matched_block_id": cell.row["matched_block_id"],
            "model_id": MODEL_ID,
            "arm": cell.arm,
            "relation": cell.relation,
            "environment_seed": cell.seed,
            "sampling_seed": cell.seed,
            "fixture_id": cell.row["fixture_id"],
            "released_fixture_sha256": cell.row["fixture_sha256"],
            "prompt_sha256": cell.row["prompt_sha256"],
            "release_fingerprint_sha256": expected_release,
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
        },
        "live reset attestation",
    )
    for key in ("physical_reset_sha256", "initial_state_sha256", "fixture_match_evidence_sha256"):
        _sha(reset.get(key), f"live reset attestation.{key}")
    settle_path_value = reset.get("settle_stability_evidence_path")
    if not isinstance(settle_path_value, str) or not settle_path_value:
        _fail("live reset attestation lacks settle/stability evidence path")
    settle_path = Path(settle_path_value).resolve()
    settle_evidence = load_json(settle_path, "settle/stability evidence")
    settle_hash = _sha(
        reset.get("settle_stability_evidence_sha256"),
        "live reset attestation.settle_stability_evidence_sha256",
    )
    if not settle_path.is_file() or sha256_file(settle_path) != settle_hash:
        _fail("settle/stability evidence file hash changed")
    validate_settle_stability_evidence(settle_evidence, cell=cell)
    fingerprint = sha256_bytes(canonical_json_bytes(reset))
    return reset, fingerprint


class PhaseBNanoRequestAdapter:
    """Request gate for one released cell using an injected model transport."""

    def __init__(
        self,
        *,
        cell: AuthorizedCell,
        release: ReleaseBundle,
        runtime: Mapping[str, Any],
        reset_attestation: Mapping[str, Any],
        reset_fingerprint_sha256: str,
        transport: Callable[[dict[str, Any]], Mapping[str, Any]],
    ) -> None:
        _sha(reset_fingerprint_sha256, "reset fingerprint")
        if not callable(transport):
            _fail("transport must be callable")
        # The attestation was already file-validated.  Re-bind the two
        # fingerprints here so an in-memory substitution cannot issue a call.
        if reset_attestation.get("registered_cell_id") != cell.cell_id:
            _fail("reset attestation is for a different cell")
        if reset_attestation.get("release_fingerprint_sha256") != release.release_fingerprint(cell):
            _fail("per-cell release fingerprint changed before request")
        if sha256_bytes(canonical_json_bytes(reset_attestation)) != reset_fingerprint_sha256:
            _fail("per-cell reset fingerprint changed before request")
        self.cell = cell
        self.release = release
        self.runtime = dict(runtime)
        self.reset_attestation = dict(reset_attestation)
        self.reset_fingerprint_sha256 = reset_fingerprint_sha256
        self.transport = transport
        self.request_count = 0

    def request(self, observation: Any, instruction: str, *, action_step_start: int) -> dict[str, Any]:
        if instruction != self.cell.row["prompt"]:
            _fail("the episode prompt must remain byte-identical and static")
        expected_start = self.request_count * ACTION_CHUNK_STEPS
        if type(action_step_start) is not int or action_step_start != expected_start:
            _fail("policy requests must begin on contiguous 32-action chunk boundaries")
        if action_step_start >= ACTION_CAP:
            _fail("no policy request is allowed at or beyond the 450-action cap")
        request = {
            "observation": observation,
            "instruction": instruction,
            "sampling_seed": self.cell.seed,
            "registered_cell_id": self.cell.cell_id,
            "request_index": self.request_count,
            "action_step_start": action_step_start,
            "release_fingerprint_sha256": self.release.release_fingerprint(self.cell),
            "reset_fingerprint_sha256": self.reset_fingerprint_sha256,
            "runtime_identity_sha256": self.runtime["runtime_identity_sha256"],
        }
        response = self.transport(request)
        if not isinstance(response, Mapping):
            _fail("policy transport response must be an object")
        action = np.asarray(response.get("action"), dtype=np.float32)
        future = np.asarray(response.get("video"), dtype=np.uint8)
        if action.shape != (ACTION_CHUNK_STEPS, ACTION_DIM) or not np.isfinite(action).all():
            _fail("Cosmos3 Nano must return one finite [32,8] action chunk")
        if future.ndim != 4 or future.shape[0] != 33 or future.shape[-1] != 3:
            _fail("every exposed Cosmos3 Nano future must be a 33-frame RGB array")
        if response.get("sampling_seed") != self.cell.seed:
            _fail("Cosmos3 Nano did not echo the released sampling seed")
        self.request_count += 1
        return {"action": action, "video": future, "sampling_seed": self.cell.seed}
