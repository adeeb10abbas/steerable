from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest

from experiments.v3.phase_c_semantic_equivalence_v3c002.compiler import compile_episode, compile_results
from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import (
    ContractError,
    deterministic_condition_order,
    load_cells,
    registered_prompts,
)
from experiments.v3.phase_c_semantic_equivalence_v3c002.wording_gate import build_blinded_sheet, validate_attestations


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

    def test_primary_analysis_uses_registered_physical_goal_and_requires_both_directions(self) -> None:
        pairs, result = compile_results(_episodes(), registration_sha256="d" * 64, queue_sha256="e" * 64)
        self.assertEqual(len(pairs), 682)
        self.assertTrue(result["primary_requested_side_depth_equivalence"]["left"]["depth_inverse_minus_canonical_m"]["equivalent"])
        self.assertTrue(result["primary_requested_side_depth_equivalence"]["right"]["depth_inverse_minus_canonical_m"]["equivalent"])
        self.assertTrue(result["positive_controls"]["inverse_reference"]["positive_with_ci_excluding_zero"])
        self.assertTrue(result["model_level_semantic_depth_equivalence_claim_authorized"])

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
                "prompt": cell.row["prompt"],
                "prompt_sha256": cell.row["prompt_sha256"],
                "final_detached_release": True,
                "reported_frozen_task_success": True,
                "reported_failure_category": "correct",
                "signed_final_lateral_offset": 0.10,
                "requested_side_depth": 0.10,
                "initial_state_sha256": "a" * 64,
                "request_events": [{"replan_index": 0, "request_seed": 12000000}],
                "runtime_identity": {name: "b" * 64 for name in ("checkpoint_digest", "controller_digest", "action_interface_digest", "camera_configuration_digest", "horizon_digest", "scorer_digest")},
                "state_trace": [
                    {"action_step": 0, "object_xyz": [0.0, 0.0, 0.0], "reference_xyz": [0.0, 0.0, 0.0], "grippers_open": False, "object_grabbed": False},
                    {"action_step": 1, "object_xyz": [0.1, 0.1, 0.0], "reference_xyz": [0.0, 0.0, 0.0], "grippers_open": True, "object_grabbed": True},
                ],
                "raw_artifacts": {name: binding for name in ("simulator_video", "executed_action_trace", "raw_episode_jsonl", "final_state", "state_trace")},
            }
            episode = compile_episode(raw, cell=cell, registration_sha256=raw["registration_sha256"], queue_sha256=raw["queue_sha256"])
            self.assertEqual(episode["physical_goal"], "left")
            self.assertEqual(episode["requested_side_depth"], 0.10)


def _episodes() -> list[dict]:
    prompts = registered_prompts()
    rows = []
    for seed in range(12000, 12341):
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
            })
    return rows
