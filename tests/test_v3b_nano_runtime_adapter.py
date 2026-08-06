#!/usr/bin/env python3
"""Fail-closed tests for the V3-B001 Nano runtime and raw compiler."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.v3.cosmos_nano_phase_b.compile_cell import (
    EXPORT_SCHEMA,
    compile_cell,
)
from experiments.v3.cosmos_nano_phase_b.live_support import compute_live_stack_sha256
from experiments.v3.cosmos_nano_phase_b.runtime_adapter import (
    ACTION_CAP,
    ACTION_CHUNK_STEPS,
    ACTION_DIM,
    ACTION_SPACE,
    ANGULAR_SPEED_TOLERANCE_RAD_S,
    AMENDMENT_ID,
    AMENDMENT_SCHEMA,
    ARMS,
    CELL_SCHEMA,
    CHECKPOINT_REVISION,
    COSMOS_REPOSITORY_COMMIT,
    EMPTY_SHA256,
    MANIFEST_SCHEMA,
    MIRROR_FACTOR,
    MODEL_ID,
    MODEL_REPOSITORY,
    PHASE,
    PROMPTS,
    RELATIONS,
    RESET_SCHEMA,
    SETTLE_EVIDENCE_SCHEMA,
    SETTLE_OBJECTS,
    SETTLE_STEPS,
    STABILITY_WINDOW_STEPS,
    LINEAR_SPEED_TOLERANCE_M_S,
    ROBOLAB_REPOSITORY_COMMIT,
    RUNTIME_SCHEMA,
    SEEDS,
    STUDY_ID,
    PhaseBNanoRequestAdapter,
    RuntimeContractError,
    canonical_json_bytes,
    compute_adapter_contract_sha256,
    load_release_bundle,
    release_json_bytes,
    sha256_bytes,
    sha256_file,
    validate_reset_attestation,
    validate_settle_stability_evidence,
    verify_runtime_identity,
)
from tools.vla_wam_v3_episode_schema import (
    MEASUREMENT_FRAME_ID,
    derive_initial_state_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
REAL_RELEASE_MANIFEST = (
    ROOT
    / "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/"
    "nano_mirror_v3b001_manifest.json"
)
REAL_RELEASE_MANIFEST_SHA256 = "5c82268739feb41281435a51dcd848b575218cd9fbe5839d9ad130d1a7888830"


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n")


def _release(directory: Path) -> tuple[Path, str]:
    fixtures = {
        "control": {
            "fixture_id": "v3b001_nano_control",
            "positions_world_xyz": {"rubiks_cube": [0.3, 0.12, 0.08], "bowl": [0.44, 0.13, 0.08]},
            "factor": MIRROR_FACTOR,
        },
        "position_mirrored": {
            "fixture_id": "v3b001_nano_position_mirrored",
            "positions_world_xyz": {"rubiks_cube": [0.3, -0.12, 0.08], "bowl": [0.44, -0.13, 0.08]},
            "factor": MIRROR_FACTOR,
        },
    }
    amendment = {
        "schema_version": AMENDMENT_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "phase": PHASE,
        "status": "released_after_model_blind_calibration_before_any_phase_b_model_request",
        "exact_prompts": PROMPTS,
        "model_identity": {
            "model_id": MODEL_ID,
            "model_repository": MODEL_REPOSITORY,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "server_repository_commit": COSMOS_REPOSITORY_COMMIT,
            "robolab_repository_commit": ROBOLAB_REPOSITORY_COMMIT,
        },
        "design": {
            "factor": MIRROR_FACTOR,
            "arms": list(ARMS),
            "directions": list(RELATIONS),
            "seeds": list(SEEDS),
            "matched_seed_count": 27,
            "cells_per_seed": 4,
            "behavioral_cell_count": 108,
        },
        "fixtures": fixtures,
    }
    amendment_path = directory / "post_result_nano_mirror_v3b001_amendment.json"
    _write_json(amendment_path, amendment)
    amendment_hash = sha256_file(amendment_path)

    rows = []
    for seed in SEEDS:
        for order, (arm, relation) in enumerate(
            (("control", "left"), ("control", "right"),
             ("position_mirrored", "left"), ("position_mirrored", "right")),
            1,
        ):
            fixture = fixtures[arm]
            row = {
                "schema_version": CELL_SCHEMA,
                "study_id": STUDY_ID,
                "amendment_id": AMENDMENT_ID,
                "amendment_sha256": amendment_hash,
                "phase": PHASE,
                "arena": "droid_robolab",
                "model_id": MODEL_ID,
                "cell_id": f"v3b001:nano:seed{seed}:{arm}:{relation}",
                "matched_block_id": f"v3b001:nano:seed{seed}",
                "arm": arm,
                "relation": relation,
                "environment_seed": seed,
                "sampling_seed": seed,
                "execution_order_index_within_seed": order,
                "randomization_key_sha256": hashlib.sha256(
                    f"{seed}:{arm}:{relation}:{order}".encode()
                ).hexdigest(),
                "factor": MIRROR_FACTOR,
                "fixture_id": fixture["fixture_id"],
                "fixture_sha256": sha256_bytes(release_json_bytes(fixture)),
                "prompt_family": "direct_command",
                "prompt": PROMPTS[relation],
                "prompt_sha256": hashlib.sha256(PROMPTS[relation].encode()).hexdigest(),
                "success_predicate_id": "v2_frozen_droid_robolab_release_inside_45deg_requested_relation",
                "runtime_identity_requirement": {
                    "model_repository": MODEL_REPOSITORY,
                    "checkpoint_revision": CHECKPOINT_REVISION,
                    "server_repository_commit": COSMOS_REPOSITORY_COMMIT,
                    "robolab_repository_commit": ROBOLAB_REPOSITORY_COMMIT,
                    "clean_external_repositories_required": True,
                },
                "required_raw_outputs": [
                    "viewport_video",
                    "executed_action_trace",
                    "raw_result_jsonl",
                    "every_exposed_decoded_future",
                ],
                "required_episode_fields": {
                    "signed_final_lateral_offset_m": "required",
                    "final_requested_signed_margin_m": "required",
                    "requested_success": "required",
                    "failure_class": "required",
                },
                "execution_status": "authorized_after_v3b001_calibration_with_live_identity_and_output_gate_recheck",
            }
            rows.append(row)
    cells_path = directory / "nano_mirror_v3b001_cells.jsonl"
    cells_path.write_text(
        "".join(json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "status": "hash_bound_release_ready",
        "files": {
            "amendment": {
                "path": amendment_path.name,
                "sha256": sha256_file(amendment_path),
                "bytes": amendment_path.stat().st_size,
            },
            "cells": {
                "path": cells_path.name,
                "sha256": sha256_file(cells_path),
                "bytes": cells_path.stat().st_size,
                "row_count": 108,
            },
        },
    }
    manifest_path = directory / "nano_mirror_v3b001_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path, sha256_file(manifest_path)


def _runtime(directory: Path, release_hash: str) -> tuple[Path, dict]:
    runtime = {
        "schema_version": RUNTIME_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "checkpoint_identifier": MODEL_REPOSITORY,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_sha256": "1" * 64,
        "checkpoint_hash_gate_passed": True,
        "external_repository_commit": COSMOS_REPOSITORY_COMMIT,
        "external_repository_diff_hash": EMPTY_SHA256,
        "simulator_repository_commit": ROBOLAB_REPOSITORY_COMMIT,
        "simulator_repository_diff_hash": EMPTY_SHA256,
        "environment_lock_sha256": "2" * 64,
        "phase_b_adapter_contract_sha256": compute_adapter_contract_sha256(ROOT),
        "release_manifest_sha256": release_hash,
        "action_space": ACTION_SPACE,
        "action_chunk_shape": [ACTION_CHUNK_STEPS, ACTION_DIM],
        "open_loop_horizon": ACTION_CHUNK_STEPS,
        "action_cap": ACTION_CAP,
        "instruction_controller": "static",
        "phase_b_live_stack_sha256": compute_live_stack_sha256(ROOT),
    }
    runtime["runtime_identity_sha256"] = sha256_bytes(canonical_json_bytes(runtime))
    path = directory / "runtime.json"
    _write_json(path, runtime)
    return path, runtime


def _steps(relation: str, actions: int) -> list[dict]:
    sign = 1.0 if relation == "left" else -1.0
    rows = []
    for index in range(actions + 1):
        final = index >= max(1, actions - 2)
        rows.append(
            {
                "action_step": index,
                "object_xyz": [0.05, sign * 0.20 if final else 0.0, 0.04 if index > 0 else 0.0],
                "reference_xyz": [0.0, 0.0, 0.0],
                "grippers_open": index == actions,
                "contact_detected": False,
            }
        )
    return rows


def _reset(
    directory: Path,
    *,
    release,
    cell,
    runtime: dict,
    steps: list[dict],
) -> tuple[Path, dict]:
    initial_hash = derive_initial_state_sha256(
        {"measurement_frame": MEASUREMENT_FRAME_ID, "steps": steps}
    )
    settle = {
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
        "stability_window_component_maxima": {
            name: {
                "max_linear_component_speed_m_s": 0.001,
                "max_angular_component_speed_rad_s": 0.01,
            }
            for name in SETTLE_OBJECTS
        },
        "post_settle_velocities": {name: [0.0] * 6 for name in SETTLE_OBJECTS},
        "post_settle_positions_world_xyz_m": {
            name: [0.1, 0.0, 0.1] for name in SETTLE_OBJECTS
        },
        "post_settle_quaternions_world_wxyz": {
            name: [1.0, 0.0, 0.0, 0.0] for name in SETTLE_OBJECTS
        },
        "neutral_after_settle": True,
        "episode_length_buf_before_reset": [SETTLE_STEPS + STABILITY_WINDOW_STEPS],
        "episode_length_buf_reset_passed": True,
        "episode_length_buf_after_reset": [0],
        "model_request_count_during_gate": 0,
    }
    settle_path = directory / "settle.json"
    _write_json(settle_path, settle)
    reset = {
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
        "settle_stability_evidence_path": str(settle_path),
        "settle_stability_evidence_sha256": sha256_file(settle_path),
        "physical_reset_sha256": "3" * 64,
        "initial_state_sha256": initial_hash,
        "fixture_match_evidence_sha256": "4" * 64,
    }
    path = directory / "reset.json"
    _write_json(path, reset)
    return path, reset


def _export(
    directory: Path,
    *,
    release,
    cell,
    runtime: dict,
    reset_path: Path,
    reset_hash: str,
    steps: list[dict],
    requested_success: bool = True,
) -> Path:
    actions = len(steps) - 1
    executed = directory / "executed.npy"
    np.save(executed, np.zeros((actions, ACTION_DIM), dtype=np.float32), allow_pickle=False)
    video = directory / "viewport.mp4"
    video.write_bytes(b"bounded-test-viewport")
    requests = []
    for index in range(math.ceil(actions / ACTION_CHUNK_STEPS)):
        returned = directory / f"returned_{index}.npy"
        future = directory / f"future_{index}.npy"
        np.save(returned, np.zeros((ACTION_CHUNK_STEPS, ACTION_DIM), dtype=np.float32), allow_pickle=False)
        np.save(future, np.zeros((33, 2, 2, 3), dtype=np.uint8), allow_pickle=False)
        requests.append(
            {
                "request_index": index,
                "action_step_start": index * ACTION_CHUNK_STEPS,
                "sampling_seed": cell.seed,
                "prompt": cell.row["prompt"],
                "release_fingerprint_sha256": release.release_fingerprint(cell),
                "reset_fingerprint_sha256": reset_hash,
                "returned_action_path": str(returned),
                "decoded_future_path": str(future),
            }
        )
    export = {
        "schema_version": EXPORT_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "phase": PHASE,
        "registered_cell_id": cell.cell_id,
        "matched_block_id": cell.row["matched_block_id"],
        "model_id": MODEL_ID,
        "arm": cell.arm,
        "requested_relation": cell.relation,
        "prompt": cell.row["prompt"],
        "prompt_sha256": cell.row["prompt_sha256"],
        "environment_seed": cell.seed,
        "sampling_seed": cell.seed,
        "fixture_id": cell.row["fixture_id"],
        "fixture_sha256": cell.row["fixture_sha256"],
        "release_fingerprint_sha256": release.release_fingerprint(cell),
        "reset_fingerprint_sha256": reset_hash,
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "action_space": ACTION_SPACE,
        "open_loop_horizon": ACTION_CHUNK_STEPS,
        "action_cap": ACTION_CAP,
        "instruction_controller": "static",
        "attempt_id": "test-attempt-01",
        "initial_state_sha256": derive_initial_state_sha256(
            {"measurement_frame": MEASUREMENT_FRAME_ID, "steps": steps}
        ),
        "steps": steps,
        "actions_executed": actions,
        "executed_action_trace_path": str(executed),
        "viewport_video_path": str(video),
        "policy_requests": requests,
        "reset_attestation_path": str(reset_path),
        "source_artifacts": {},
        "requested_success": requested_success,
        "right_censored": not requested_success,
        "final_detached_release": requested_success,
        "wall_time_s": 1.0,
        "operational_wall_time_valid": True,
        "first_contact_step": None,
        "first_contact_unavailable_reason": None,
    }
    path = directory / "export.json"
    _write_json(path, export)
    return path


class NanoPhaseBRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.tmp = Path(self.temporary.name)
        self.release_path, self.release_hash = _release(self.tmp)
        self.release = load_release_bundle(
            self.release_path, expected_manifest_sha256=self.release_hash
        )
        self.cell = self.release.cell("v3b001:nano:seed9400:control:left")
        self.runtime_path, self.runtime = _runtime(self.tmp, self.release_hash)
        verify_runtime_identity(self.runtime_path, study_root=ROOT, release=self.release)

    def test_release_is_exactly_108_hash_pinned_cells(self) -> None:
        self.assertEqual(len(self.release.cells), 108)
        self.assertEqual({cell.seed for cell in self.release.cells}, set(SEEDS))
        with self.assertRaisesRegex(RuntimeContractError, "externally pinned"):
            load_release_bundle(self.release_path, expected_manifest_sha256="f" * 64)
        cells_path = self.tmp / "nano_mirror_v3b001_cells.jsonl"
        cells_path.write_text(cells_path.read_text().replace("position_mirrored", "mirror", 1))
        with self.assertRaisesRegex(RuntimeContractError, "hash/size"):
            load_release_bundle(self.release_path, expected_manifest_sha256=self.release_hash)

    def test_real_committed_release_loads_with_exact_factor_and_hash(self) -> None:
        release = load_release_bundle(
            REAL_RELEASE_MANIFEST,
            expected_manifest_sha256=REAL_RELEASE_MANIFEST_SHA256,
        )
        self.assertEqual(len(release.cells), 108)
        self.assertEqual({cell.row["factor"] for cell in release.cells}, {MIRROR_FACTOR})
        self.assertEqual(
            release.cells[0].cell_id,
            "v3b001:nano:seed9400:position_mirrored:right",
        )

    def test_no_transport_call_before_per_cell_fingerprints(self) -> None:
        steps = _steps("left", 4)
        reset_path, _ = _reset(
            self.tmp, release=self.release, cell=self.cell, runtime=self.runtime, steps=steps
        )
        calls: list[dict] = []

        def transport(request: dict) -> dict:
            calls.append(request)
            return {
                "action": np.zeros((ACTION_CHUNK_STEPS, ACTION_DIM), dtype=np.float32),
                "video": np.zeros((33, 2, 2, 3), dtype=np.uint8),
                "sampling_seed": self.cell.seed,
            }

        reset = json.loads(reset_path.read_text())
        reset["release_fingerprint_sha256"] = "f" * 64
        _write_json(reset_path, reset)
        with self.assertRaisesRegex(RuntimeContractError, "release_fingerprint"):
            validate_reset_attestation(
                reset_path, cell=self.cell, release=self.release, runtime=self.runtime
            )
        self.assertEqual(calls, [])

        reset_path, _ = _reset(
            self.tmp, release=self.release, cell=self.cell, runtime=self.runtime, steps=steps
        )
        reset, reset_hash = validate_reset_attestation(
            reset_path, cell=self.cell, release=self.release, runtime=self.runtime
        )
        adapter = PhaseBNanoRequestAdapter(
            cell=self.cell,
            release=self.release,
            runtime=self.runtime,
            reset_attestation=reset,
            reset_fingerprint_sha256=reset_hash,
            transport=transport,
        )
        with self.assertRaisesRegex(RuntimeContractError, "byte-identical"):
            adapter.request({}, PROMPTS["right"], action_step_start=0)
        self.assertEqual(calls, [])
        response = adapter.request({}, PROMPTS["left"], action_step_start=0)
        self.assertEqual(response["action"].shape, (32, 8))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["release_fingerprint_sha256"], self.release.release_fingerprint(self.cell))
        self.assertEqual(calls[0]["reset_fingerprint_sha256"], reset_hash)

    def test_settle_gate_rejects_linear_or_angular_instability(self) -> None:
        steps = _steps("left", 4)
        reset_path, _ = _reset(
            self.tmp, release=self.release, cell=self.cell, runtime=self.runtime, steps=steps
        )
        reset = json.loads(reset_path.read_text())
        settle_path = Path(reset["settle_stability_evidence_path"])
        stable = json.loads(settle_path.read_text())
        validate_settle_stability_evidence(stable, cell=self.cell)
        for field, value, pattern in (
            ("max_linear_component_speed_m_s", 0.020001, "linear-speed"),
            ("max_angular_component_speed_rad_s", 0.200001, "angular-speed"),
        ):
            unstable = json.loads(json.dumps(stable))
            unstable["stability_window_component_maxima"]["rubiks_cube"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeContractError, pattern):
                validate_settle_stability_evidence(unstable, cell=self.cell)

    def test_unsettled_reset_cannot_reach_request_adapter(self) -> None:
        steps = _steps("left", 4)
        reset_path, reset = _reset(
            self.tmp, release=self.release, cell=self.cell, runtime=self.runtime, steps=steps
        )
        settle_path = Path(reset["settle_stability_evidence_path"])
        settle = json.loads(settle_path.read_text())
        settle["stability_window_component_maxima"]["bowl"][
            "max_angular_component_speed_rad_s"
        ] = 0.21
        _write_json(settle_path, settle)
        reset["settle_stability_evidence_sha256"] = sha256_file(settle_path)
        _write_json(reset_path, reset)
        transport_calls: list[dict] = []
        with self.assertRaisesRegex(RuntimeContractError, "angular-speed"):
            validate_reset_attestation(
                reset_path, cell=self.cell, release=self.release, runtime=self.runtime
            )
        self.assertEqual(transport_calls, [])

    def test_runtime_identity_pins_exact_revisions_and_action_contract(self) -> None:
        payload = json.loads(self.runtime_path.read_text())
        payload["checkpoint_revision"] = "0" * 40
        payload["runtime_identity_sha256"] = sha256_bytes(
            canonical_json_bytes({k: v for k, v in payload.items() if k != "runtime_identity_sha256"})
        )
        _write_json(self.runtime_path, payload)
        with self.assertRaisesRegex(RuntimeContractError, "checkpoint_revision"):
            verify_runtime_identity(self.runtime_path, study_root=ROOT, release=self.release)

    def test_compiler_emits_full_sample_offset_and_margin(self) -> None:
        steps = _steps("left", 4)
        reset_path, _ = _reset(
            self.tmp, release=self.release, cell=self.cell, runtime=self.runtime, steps=steps
        )
        reset_hash = sha256_bytes(canonical_json_bytes(json.loads(reset_path.read_text())))
        export = _export(
            self.tmp,
            release=self.release,
            cell=self.cell,
            runtime=self.runtime,
            reset_path=reset_path,
            reset_hash=reset_hash,
            steps=steps,
        )
        output = self.tmp / "raw.jsonl"
        manifest = compile_cell(
            study_root=ROOT,
            release_manifest=self.release_path,
            release_manifest_sha256=self.release_hash,
            runtime_manifest=self.runtime_path,
            reset_attestation=reset_path,
            cell_id=self.cell.cell_id,
            export=export,
            output_jsonl=output,
        )
        row = json.loads(output.read_text())
        self.assertEqual(manifest["row_count"], 1)
        self.assertAlmostEqual(row["measurements"]["signed_final_lateral_offset_m"], 0.20)
        self.assertAlmostEqual(row["measurements"]["final_requested_signed_margin_m"], 0.20)
        self.assertEqual(row["action_chunk_shape"], [32, 8])
        self.assertEqual(row["release_fingerprint_sha256"], self.release.release_fingerprint(self.cell))
        self.assertEqual(row["reset_fingerprint_sha256"], reset_hash)

    def test_short_failure_is_excluded_before_jsonl_write(self) -> None:
        steps = _steps("left", 4)
        reset_path, _ = _reset(
            self.tmp, release=self.release, cell=self.cell, runtime=self.runtime, steps=steps
        )
        reset_hash = sha256_bytes(canonical_json_bytes(json.loads(reset_path.read_text())))
        export = _export(
            self.tmp,
            release=self.release,
            cell=self.cell,
            runtime=self.runtime,
            reset_path=reset_path,
            reset_hash=reset_hash,
            steps=steps,
            requested_success=False,
        )
        output = self.tmp / "must_not_exist.jsonl"
        with self.assertRaisesRegex(RuntimeContractError, "450-action cap"):
            compile_cell(
                study_root=ROOT,
                release_manifest=self.release_path,
                release_manifest_sha256=self.release_hash,
                runtime_manifest=self.runtime_path,
                reset_attestation=reset_path,
                cell_id=self.cell.cell_id,
                export=export,
                output_jsonl=output,
            )
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
