from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

from experiments.v3.phase_c_semantic_equivalence_v3c002.compiler import compile_episode, compile_results, decision_memo, manuscript_insert
from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import (
    ContractError,
    REPO_ROOT as CONTRACT_REPO_ROOT,
    deterministic_condition_order,
    file_binding,
    load_cells,
    registered_prompts,
    require_released_gate,
)
from experiments.v3.phase_c_semantic_equivalence_v3c002.wording_gate import build_blinded_sheet, validate_attestations
from experiments.v3.phase_c_semantic_equivalence_v3c002.runtime import bind_runtime
from tools.validate_v3c002_v1_historical import validate as validate_v1_historical
from tools.validate_v3c002_v2_historical import validate as validate_v2_historical
from tools.validate_v3c002_v3_historical import validate as validate_v3_historical
from tools.validate_v3c002_v4_historical import validate as validate_v4_historical
from tools.validate_v3c002_v5_historical import validate as validate_v5_historical
from tools.validate_v3c002_v6_historical import validate as validate_v6_historical
from tools.validate_v3c002_active import validate as validate_active


ROOT = Path(__file__).resolve().parents[1]


class V3C002SemanticEquivalenceTests(unittest.TestCase):
    def test_builds_exactly_one_complete_four_condition_block_per_registered_seed(self) -> None:
        with __import__("tempfile").TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            subprocess.run([sys.executable, "tools/build_v3c002_registration.py", "--output-root", str(tmp_path)], cwd=ROOT, check=True)
            registration, cells = load_cells(registration_path=tmp_path / "registration.json", queue_path=tmp_path / "queue.jsonl")
            self.assertEqual(len(cells), 1364)
            self.assertFalse(registration["model_requests_authorized"])
            self.assertEqual(deterministic_condition_order(12000), tuple(cells[index].row["execution_order"][index] for index in range(4)))
            self.assertTrue(all(cell.row["layout_candidate_sha256"] == registration["e004_s1_layout"]["candidate_sha256"] for cell in cells))

    def test_blinded_sheet_does_not_disclose_expected_goal_or_condition(self) -> None:
        sheet = build_blinded_sheet()
        encoded = json.dumps(sheet, sort_keys=True)
        self.assertNotIn("canonical_left", encoded)
        self.assertNotIn("inverse_reference_left", encoded)
        self.assertTrue(sheet["blinding"]["withheld_from_readers"])

    def test_two_real_distinct_attestations_are_required(self) -> None:
        with __import__("tempfile").TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            sheet = tmp_path / "sheet.json"
            sheet.write_text(json.dumps(build_blinded_sheet()), encoding="utf-8")
            payload = {
                "schema_version": "vla-wam-shared-v3c002-human-prompt-attestation-v1",
                "sheet_sha256": __import__("hashlib").sha256(sheet.read_bytes()).hexdigest(),
                "reader_id": "reader-a",
                "authorization_reference": "authorized-review-log:A",
                "attested_at_utc": "2026-08-12T00:00:00Z",
                "signature_or_record_reference": "record:A",
                "responses": [{"reader_pair_id": "pair_kestrel", "decision": "same_physical_endpoint"}, {"reader_pair_id": "pair_orchid", "decision": "same_physical_endpoint"}],
            }
            first = tmp_path / "a.json"; second = tmp_path / "b.json"
            first.write_text(json.dumps(payload), encoding="utf-8")
            second.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "not independent"):
                validate_attestations(sheet_path=sheet, attestation_paths=[first, second])

    def test_finalizer_records_date_only_attestation_receipt_order_without_inventing_times(self) -> None:
        with __import__("tempfile").TemporaryDirectory(dir=ROOT) as directory:
            tmp_path = Path(directory); draft = tmp_path / "draft"; active = tmp_path / "active"
            subprocess.run([sys.executable, "tools/build_v3c002_registration.py", "--output-root", str(draft)], cwd=ROOT, check=True)
            sheet_sha = __import__("hashlib").sha256((draft / "prompt_comprehension_sheet.json").read_bytes()).hexdigest()
            attestations = []
            for reader in ("reader-a", "reader-b"):
                path = tmp_path / f"{reader}.json"
                _write_json(path, {
                    "schema_version": "vla-wam-shared-v3c002-human-prompt-attestation-v1", "sheet_sha256": sheet_sha,
                    "reader_id": reader, "authorization_reference": f"auth:{reader}", "attested_at_utc": "2026-08-12",
                    "signature_or_record_reference": f"record:{reader}",
                    "responses": [{"reader_pair_id": "pair_kestrel", "decision": "same_physical_endpoint"}, {"reader_pair_id": "pair_orchid", "decision": "same_physical_endpoint"}],
                }); attestations.append(path)
            subprocess.run([
                sys.executable, "tools/finalize_v3c002_registration.py", "--draft-root", str(draft),
                "--reader-attestation", str(attestations[0]), "--reader-attestation", str(attestations[1]),
                "--registered-at-utc", "2026-08-12T16:00:00Z", "--activation-root", str(active),
            ], cwd=ROOT, check=True)
            receipt = json.loads((active / "attestation_receipt_order.json").read_text(encoding="utf-8"))
            self.assertTrue(receipt["received_before_registration"])
            self.assertEqual(receipt["receipt_source"], "Codex task response")
            self.assertIn("no time-of-day", receipt["timestamp_limitation"])
            self.assertEqual(json.loads(attestations[0].read_text())["attested_at_utc"], "2026-08-12")
            self.assertEqual(validate_active(active)["status"], "valid_registered_pending_preflight_release")
            (active / "results").mkdir()
            _write_json(active / "results/results.json", {"partial": True})
            with self.assertRaisesRegex(ContractError, "partial C002 final result bundle"):
                validate_active(active)

    def test_primary_analysis_uses_registered_physical_goal_and_requires_both_directions(self) -> None:
        pairs, result = compile_results(_episodes(), registration_sha256="d" * 64, queue_sha256="e" * 64)
        self.assertEqual(len(pairs), 682)
        self.assertTrue(result["primary_requested_side_depth_equivalence"]["left"]["depth_inverse_minus_canonical_m"]["equivalent"])
        self.assertTrue(result["primary_requested_side_depth_equivalence"]["right"]["depth_inverse_minus_canonical_m"]["equivalent"])
        self.assertTrue(result["positive_controls"]["inverse_reference"]["positive_with_ci_excluding_zero"])
        self.assertTrue(result["model_level_semantic_depth_equivalence_claim_authorized"])

    def test_semantic_claim_is_withheld_when_depth_tost_passes_but_inverse_positive_control_fails(self) -> None:
        episodes = _episodes()
        for row in episodes:
            if row["prompt_condition"] == "inverse_reference_right":
                row["endpoint_value"] = 0.102
        _, result = compile_results(episodes, registration_sha256="d" * 64, queue_sha256="e" * 64)
        self.assertTrue(result["descriptive_directional_depth_form_equivalence"])
        self.assertFalse(result["positive_controls"]["inverse_reference"]["positive_with_ci_excluding_zero"])
        self.assertFalse(result["model_level_semantic_depth_equivalence_claim_authorized"])
        self.assertTrue(result["model_level_semantic_depth_equivalence_claim_withheld"])
        self.assertIn("claim authorized: False", decision_memo(result))
        self.assertIn("| withheld |", manuscript_insert(result))

    def test_complete_claim_output_and_partial_block_rejection(self) -> None:
        _, result = compile_results(_episodes(), registration_sha256="d" * 64, queue_sha256="e" * 64)
        self.assertIn("claim authorized: True", decision_memo(result))
        self.assertIn("| authorized |", manuscript_insert(result))
        with self.assertRaisesRegex(ContractError, "lacks a complete block"):
            compile_results(_episodes()[:-1], registration_sha256="d" * 64, queue_sha256="e" * 64)

    def test_contract_repo_root_resolves_the_active_worktree(self) -> None:
        self.assertEqual(CONTRACT_REPO_ROOT, ROOT)

    def test_historical_v1_draft_remains_independently_verifiable_and_unexecuted(self) -> None:
        result = validate_v1_historical()
        self.assertEqual(result["status"], "valid_immutable_unexecuted_superseded_v1_draft")
        self.assertEqual(result["queue_rows"], 1364)

    def test_historical_v2_draft_remains_independently_verifiable_and_unexecuted(self) -> None:
        result = validate_v2_historical()
        self.assertEqual(result["status"], "valid_immutable_unexecuted_superseded_v2_draft")
        self.assertEqual(result["queue_rows"], 1364)

    def test_historical_v3_draft_remains_independently_verifiable_and_unexecuted(self) -> None:
        result = validate_v3_historical()
        self.assertEqual(result["status"], "valid_immutable_unexecuted_superseded_v3_draft")
        self.assertEqual(result["queue_rows"], 1364)

    def test_historical_v4_draft_remains_independently_verifiable_and_unexecuted(self) -> None:
        result = validate_v4_historical()
        self.assertEqual(result["status"], "valid_immutable_unexecuted_superseded_v4_draft")
        self.assertEqual(result["queue_rows"], 1364)

    def test_historical_v5_draft_remains_independently_verifiable_and_unexecuted(self) -> None:
        result = validate_v5_historical()
        self.assertEqual(result["status"], "valid_immutable_unexecuted_superseded_v5_draft")
        self.assertEqual(result["queue_rows"], 1364)

    def test_historical_v6_draft_remains_independently_verifiable_and_unexecuted(self) -> None:
        result = validate_v6_historical()
        self.assertEqual(result["status"], "valid_immutable_unexecuted_superseded_v6_draft")
        self.assertEqual(result["queue_rows"], 1364)

    def test_release_gate_requires_and_hash_checks_all_pre_request_evidence(self) -> None:
        with __import__("tempfile").TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            registration, queue, gate = _release_fixture(tmp_path)
            with mock.patch("experiments.v3.phase_c_semantic_equivalence_v3c002.contract._verify_pushed_source_commit"):
                require_released_gate(registration_path=registration, queue_path=queue, release_gate_path=gate)
            release = json.loads(gate.read_text(encoding="utf-8"))
            physical = Path(release["physical_gate"]["path"])
            physical.write_text(physical.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with mock.patch("experiments.v3.phase_c_semantic_equivalence_v3c002.contract._verify_pushed_source_commit"):
                with self.assertRaisesRegex(ContractError, "physical gate artifact byte count changed"):
                    require_released_gate(registration_path=registration, queue_path=queue, release_gate_path=gate)

    def test_hand_authored_passed_release_cannot_bypass_missing_isolation_artifact(self) -> None:
        with __import__("tempfile").TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            registration, queue, gate = _release_fixture(tmp_path)
            release = json.loads(gate.read_text(encoding="utf-8"))
            del release["two_lane_isolation_gate"]
            gate.write_text(json.dumps(release), encoding="utf-8")
            with mock.patch("experiments.v3.phase_c_semantic_equivalence_v3c002.contract._verify_pushed_source_commit"):
                with self.assertRaisesRegex(ContractError, "two-lane isolation gate binding is missing"):
                    require_released_gate(registration_path=registration, queue_path=queue, release_gate_path=gate)

    def test_runtime_rejects_shape_correct_but_unequal_e004_component_digest(self) -> None:
        with __import__("tempfile").TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            subprocess.run([sys.executable, "tools/build_v3c002_registration.py", "--output-root", str(tmp_path / "draft")], cwd=ROOT, check=True)
            registration = tmp_path / "draft/registration.json"
            queue = tmp_path / "draft/queue.jsonl"
            exact = json.loads(registration.read_text(encoding="utf-8"))["exact_e004_pi05_runtime"]
            observed = _observed_runtime(exact)
            observed["component_digests"]["controller"] = "0" * 64
            observed_path = tmp_path / "observed.json"
            _write_json(observed_path, observed)
            with self.assertRaisesRegex(ContractError, "component digests differ"):
                bind_runtime(
                    registration_path=registration, queue_path=queue, observed_runtime_path=observed_path,
                    observed_runtime_sha256=__import__("hashlib").sha256(observed_path.read_bytes()).hexdigest(),
                    lane_pod_uid="sim-pod-a", lane_gpu_uuid="GPU-sim-a", policy_server_pod_uid="policy-pod-a",
                    policy_server_gpu_uuid="GPU-policy-a", server_port=18001, raw_root="/raw/lane-a",
                    container_identity="container-a", runtime_identity="runtime-a", lane_id="lane-a",
                    server_process_identity="pid-a", server_lock_identity="lock-a",
                )

    def test_model_level_primary_claim_is_a_two_goal_conjunction(self) -> None:
        episodes = _episodes()
        for row in episodes:
            if row["prompt_condition"] == "inverse_reference_right":
                row["requested_side_depth"] = 0.20
        _, result = compile_results(episodes, registration_sha256="d" * 64, queue_sha256="e" * 64)
        self.assertTrue(result["primary_requested_side_depth_equivalence"]["left"]["depth_inverse_minus_canonical_m"]["equivalent"])
        self.assertFalse(result["primary_requested_side_depth_equivalence"]["right"]["depth_inverse_minus_canonical_m"]["equivalent"])
        self.assertFalse(result["model_level_semantic_depth_equivalence_claim_authorized"])

    def test_compiler_uses_goal_metadata_not_surface_word(self) -> None:
        with __import__("tempfile").TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            subprocess.run([sys.executable, "tools/build_v3c002_registration.py", "--output-root", str(tmp_path / "draft")], cwd=ROOT, check=True)
            registration_path = tmp_path / "draft" / "registration.json"
            queue_path = tmp_path / "draft" / "queue.jsonl"
            _, cells = load_cells(registration_path=registration_path, queue_path=queue_path)
            registration = json.loads(registration_path.read_text(encoding="utf-8"))
            exact = registration["exact_e004_pi05_runtime"]
            cell = next(value for value in cells if value.cell_id == "v3c002:seed12000:inverse_reference_left")
            artifact = tmp_path / "raw.bin"; artifact.write_bytes(b"evidence")
            binding = {"path": str(artifact), "bytes": artifact.stat().st_size, "sha256": __import__("hashlib").sha256(artifact.read_bytes()).hexdigest()}
            raw = {
                "schema_version": "vla-wam-shared-v3c002-raw-episode-v1",
                "study_id": "vla_wam_language_steerability_v3",
                "amendment_id": "V3-C002",
                "cell_id": cell.cell_id,
                "cell_sha256": cell.row_sha256,
                "registration_sha256": __import__("hashlib").sha256(registration_path.read_bytes()).hexdigest(),
                "queue_sha256": __import__("hashlib").sha256(queue_path.read_bytes()).hexdigest(),
                "model_id": "pi05_current_stack_droid",
                "physical_goal": "left",
                "surface_direction_word": "right",
                "prompt": cell.row["prompt"],
                "prompt_utf8_hex": cell.row["prompt_utf8_hex"],
                "prompt_sha256": cell.row["prompt_sha256"],
                "final_detached_release": True,
                "reported_frozen_task_success": True,
                "reported_failure_category": "correct",
                "signed_final_lateral_offset": 0.10,
                "requested_side_depth": 0.10,
                "initial_state_sha256": "a" * 64,
                "request_events": [{"replan_index": 0, "request_seed": 12000000}],
                "runtime_identity": {
                    **exact["identity_values"],
                    "exact_runtime_contract_sha256": exact["contract_sha256"],
                    "component_digests": exact["component_digests"],
                    "dependency_bindings": exact["dependency_bindings"],
                    "lane_id": "lane-a",
                    "simulator_pod_uid": "sim-pod-a",
                    "simulator_gpu_uuid": "GPU-sim-a",
                    "policy_server_pod_uid": "policy-pod-a",
                    "policy_server_gpu_uuid": "GPU-policy-a",
                    "server_port": 18001,
                    "raw_root": "/retained/v3c002/lane-a",
                    "container_identity": "sha256:container-a",
                    "runtime_identity": "runtime-a",
                    "server_process_identity": "pid:100:boot:a",
                    "server_lock_identity": "lock:a",
                    "full_reset": True,
                    "stage_identifier": "full_reset",
                    "policy_camera_image_artifact_hashes": {name: binding["sha256"] for name in exact["identity_values"]["policy_cameras"]},
                },
                "state_trace": [
                    {"action_step": 0, "object_xyz": [0.0, 0.0, 0.0], "reference_xyz": [0.0, 0.0, 0.0], "grippers_open": False, "object_grabbed": False},
                    {"action_step": 1, "object_xyz": [0.1, 0.1, 0.0], "reference_xyz": [0.0, 0.0, 0.0], "grippers_open": True, "object_grabbed": True},
                ],
                "raw_artifacts": {
                    **{name: binding for name in ("simulator_video", "executed_action_trace", "raw_episode_jsonl", "final_state", "state_trace")},
                    "policy_camera_images": {name: binding for name in exact["identity_values"]["policy_cameras"]},
                },
            }
            episode = compile_episode(raw, cell=cell, registration_sha256=raw["registration_sha256"], queue_sha256=raw["queue_sha256"], exact_runtime_contract=exact)
            self.assertEqual(episode["physical_goal"], "left")
            self.assertEqual(episode["requested_side_depth"], 0.10)
            malformed = json.loads(json.dumps(raw))
            malformed["state_trace"][1]["action_step"] = 3
            with self.assertRaisesRegex(ContractError, "exact frozen E004 normalizer"):
                compile_episode(malformed, cell=cell, registration_sha256=raw["registration_sha256"], queue_sha256=raw["queue_sha256"], exact_runtime_contract=exact)
            nonfinite = json.loads(json.dumps(raw))
            nonfinite["state_trace"][1]["object_xyz"][0] = float("nan")
            with self.assertRaisesRegex(ContractError, "exact frozen E004 normalizer"):
                compile_episode(nonfinite, cell=cell, registration_sha256=raw["registration_sha256"], queue_sha256=raw["queue_sha256"], exact_runtime_contract=exact)


def _episodes() -> list[dict]:
    prompts = registered_prompts()
    rows = []
    for seed in range(12000, 12341):
        lane_index = seed % 2
        for condition, signed, depth in (
            ("canonical_left", 0.10, 0.10),
            ("inverse_reference_left", 0.102, 0.102),
            ("canonical_right", -0.10, 0.10),
            ("inverse_reference_right", -0.102, 0.102),
        ):
            prompt = prompts[condition]
            rows.append({
                "episode_seed": seed,
                "seed_block_id": f"v3c002:seed{seed}",
                "prompt_condition": condition,
                "cell_id": f"v3c002:seed{seed}:{condition}",
                "physical_goal": prompt["physical_goal"],
                "initial_state_sha256": "a" * 64,
                "requested_side_depth": depth,
                "success": True,
                "endpoint_value": signed,
                "action_trace_sha256": ("b" * 64 if condition.startswith("canonical") else "c" * 64),
                "lane_id": f"lane-{lane_index}",
                "server_port": 18001 + lane_index,
                "raw_root": f"/retained/v3c002/lane-{lane_index}",
                "simulator_pod_uid": f"sim-pod-{lane_index}",
                "simulator_gpu_uuid": f"GPU-sim-{lane_index}",
                "policy_server_pod_uid": f"policy-pod-{lane_index}",
                "policy_server_gpu_uuid": f"GPU-policy-{lane_index}",
                "container_identity": f"container-{lane_index}",
                "runtime_identity_label": f"runtime-{lane_index}",
                "source_commit": "f" * 40,
                "checkpoint_digest": "d" * 64,
            })
    return rows


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _observed_runtime(exact: dict) -> dict:
    values = exact["identity_values"]
    return {
        **values,
        "exact_runtime_contract_sha256": exact["contract_sha256"],
        "component_digests": dict(exact["component_digests"]),
        "dependency_bindings": exact["dependency_bindings"],
        "simulator_pod_uid": "sim-pod-a", "simulator_gpu_uuid": "GPU-sim-a",
        "policy_server_pod_uid": "policy-pod-a", "policy_server_gpu_uuid": "GPU-policy-a",
        "server_port": 18001, "raw_root": "/raw/lane-a", "container_identity": "container-a",
        "runtime_identity": "runtime-a", "lane_id": "lane-a", "server_process_identity": "pid-a",
        "server_lock_identity": "lock-a", "full_reset": True, "stage_identifier": "full_reset",
        "policy_camera_image_artifact_hashes": {name: "a" * 64 for name in values["policy_cameras"]},
    }


def _release_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    draft = tmp_path / "draft"
    subprocess.run([sys.executable, "tools/build_v3c002_registration.py", "--output-root", str(draft)], cwd=ROOT, check=True)
    queue = draft / "queue.jsonl"
    draft_value = json.loads((draft / "registration.json").read_text(encoding="utf-8"))
    active = tmp_path / "registration.json"
    draft_value["registration_status"] = "registered_after_two_human_wording_agreements"
    _write_json(active, draft_value)
    registration_sha = __import__("hashlib").sha256(active.read_bytes()).hexdigest()
    queue_sha = __import__("hashlib").sha256(queue.read_bytes()).hexdigest()
    exact_sha = draft_value["exact_e004_pi05_runtime"]["contract_sha256"]
    source_commit = draft_value["exact_e004_pi05_runtime"]["identity_values"]["source_commit"]

    wording = tmp_path / "wording.json"
    _write_json(wording, {
        "schema_version": "vla-wam-shared-v3c002-wording-gate-v1", "status": "passed_two_authorized_independent_human_readers_agree_same_endpoint",
        "passed": True, "reader_attestations": [{"reader_id": "a"}, {"reader_id": "b"}],
    })
    source = tmp_path / "source.json"
    _write_json(source, {
        "schema_version": "vla-wam-shared-v3c002-source-push-gate-v1", "status": "passed_source_commit_pushed",
        "passed": True, "pushed": True, "source_commit": source_commit, "branch": "test", "remote": "origin",
        "registration_sha256": registration_sha,
    })
    physical = tmp_path / "physical.json"
    _write_json(physical, {
        "schema_version": "vla-wam-shared-v3c002-model-blind-physical-gate-v1", "status": "passed_exact_e004_model_blind_physical_preflight",
        "passed": True, "physical_scene": True, "full_reset": True, "policy_cameras": True, "raw_writer": True, "renderer": True,
        "model_requests": 0, "behavioral_episodes": 0, "exact_runtime_contract_sha256": exact_sha,
    })
    smoke = tmp_path / "smoke.json"
    _write_json(smoke, {
        "schema_version": "vla-wam-shared-v3c002-excluded-smoke-gate-v1", "status": "passed_excluded_four_cell_smoke",
        "passed": True, "excluded_from_behavioral_denominators": True, "completed_cells": 4,
    })
    isolation = tmp_path / "isolation.json"
    _write_json(isolation, {
        "schema_version": "vla-wam-shared-v3c002-two-lane-isolation-gate-v1", "status": "passed_two_lane_fixed_observation_isolation",
        "passed": True, "fixed_observation_equal": True, "fixed_prompt_equal": True, "request_seed_equal": True,
        "outputs_match": True, "lane_state_isolated": True,
    })
    lane_bindings = []
    for index in range(2):
        lane = tmp_path / f"lane-{index}.json"
        _write_json(lane, {
            "schema_version": "vla-wam-shared-v3c002-lane-release-manifest-v1", "status": "passed_lane_release", "passed": True,
            "lane_id": f"lane-{index}", "simulator_pod_uid": f"sim-pod-{index}", "simulator_gpu_uuid": f"GPU-sim-{index}",
            "policy_server_pod_uid": f"policy-pod-{index}", "policy_server_gpu_uuid": f"GPU-policy-{index}",
            "server_port": 18001 + index, "raw_root": f"/raw/lane-{index}", "container_identity": f"container-{index}",
            "runtime_identity": f"runtime-{index}", "server_process_identity": f"pid-{index}", "server_lock_identity": f"lock-{index}",
            "source_commit": source_commit, "exact_runtime_contract_sha256": exact_sha,
            "registration_sha256": registration_sha, "queue_sha256": queue_sha,
        })
        lane_bindings.append(file_binding(lane))
    gate = tmp_path / "release.json"
    _write_json(gate, {
        "schema_version": "vla-wam-shared-v3c002-release-gate-v4", "status": "passed_pre_request_release", "passed": True,
        "registration": file_binding(active), "queue": file_binding(queue), "wording_gate": file_binding(wording),
        "source_push_gate": file_binding(source), "physical_gate": file_binding(physical), "excluded_smoke_gate": file_binding(smoke),
        "two_lane_isolation_gate": file_binding(isolation), "lane_manifests": lane_bindings,
    })
    return active, queue, gate
