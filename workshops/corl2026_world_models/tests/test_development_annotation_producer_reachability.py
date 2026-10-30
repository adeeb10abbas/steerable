from __future__ import annotations

import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


WORKSHOP = Path(__file__).resolve().parents[1]
ROOT = WORKSHOP.parents[1]
MODULE = (
    WORKSHOP / "analysis/audit_development_annotation_producer_reachability.py"
)
REPORT = (
    WORKSHOP / "results/development_annotation_producer_reachability.json"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_development_annotation_producer_reachability", MODULE
)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class DevelopmentAnnotationProducerReachabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.blobs = audit.load_pinned_blobs()
        cls.report = audit.build_report(cls.blobs)
        cls.camera_contract = audit.load_json(
            cls.blobs[(audit.CAMERA_COMMIT, audit.CAMERA_CONTRACT_PATH)],
            "camera contract",
        )
        cls.alignment_contract = audit.load_json(
            cls.blobs[(audit.ALIGNMENT_COMMIT, audit.ALIGNMENT_CONTRACT_PATH)],
            "alignment contract",
        )

    def test_saved_report_is_exact_canonical_regeneration(self) -> None:
        saved = REPORT.read_bytes()
        self.assertEqual(saved, audit.canonical_bytes(self.report))
        self.assertEqual(audit.load_json(saved, "saved report"), self.report)
        self.assertEqual(audit.build_report(self.blobs), self.report)

    def test_exact_six_blob_authentication(self) -> None:
        sources = self.report["authenticated_source_blobs"]
        self.assertEqual(sources["count"], 6)
        self.assertEqual(sources["source_bytes"], 216167)
        self.assertEqual(len(sources["sources"]), 6)
        self.assertEqual(
            {(row["commit"], row["path"]) for row in sources["sources"]},
            set(audit.EXPECTED_SOURCES),
        )
        self.assertEqual(
            self.report["audit_activity"]["authenticated_git_blobs_read"], 6
        )
        self.assertEqual(
            self.report["audit_activity"]["authenticated_git_bytes_read"],
            216167,
        )

    def test_camera_attempt_is_diagnostic_and_publisher_is_unreachable(self) -> None:
        camera = self.report["camera_producer"]
        self.assertEqual(camera["retained_job_id"], "camera-crop-replay-witness-002")
        self.assertTrue(camera["diagnostic_only"])
        self.assertFalse(camera["publish_crop_contracts"])
        self.assertEqual(camera["publisher_definition_count"], 1)
        self.assertEqual(camera["publisher_call_site_count"], 0)
        self.assertEqual(camera["publisher_name_reference_count"], 0)
        self.assertFalse(
            camera["publisher_reachable_from_authorized_run_job_entrypoint"]
        )
        self.assertFalse(
            camera["success_receipt_reachable_from_authorized_run_job_entrypoint"]
        )
        self.assertEqual(
            camera["terminal_behavior"],
            "technical_invalid_after_successful_diagnostic_children_with_raw_outputs_"
            "retained_but_not_terminally_published",
        )
        self.assertEqual(
            camera["prospective_output_names"], audit.EXPECTED_CROP_NAMES
        )
        self.assertEqual(
            {row["name"] for row in camera["functions"]},
            {"build_wave", "_publish_contracts", "run_job"},
        )

    def test_alignment_requires_later_passed_original_pixel_attempt(self) -> None:
        alignment = self.report["physical_alignment_producer"]
        self.assertEqual(alignment["camera_terminal_minimum_attempt"], 3)
        self.assertEqual(
            alignment["forbidden_camera_job_ids"],
            ["camera-crop-replay-witness-001", "camera-crop-replay-witness-002"],
        )
        self.assertEqual(alignment["required_terminal_status"], "passed")
        self.assertEqual(
            alignment["required_terminal_decision"],
            "qualified_from_original_camera_pixels",
        )
        self.assertTrue(alignment["requires_original_camera_pixels"])
        self.assertTrue(alignment["requires_safe_for_physical_alignment_input"])
        self.assertEqual(
            alignment["prospective_output_names"], audit.EXPECTED_ALIGNMENT_NAMES
        )
        self.assertEqual(
            alignment["prerequisite_chain"],
            [
                "build_wave_from_local_files",
                "validate_local_prerequisites",
                "_validate_camera_job",
                "_camera_attempt",
            ],
        )

    def test_exact_two_chain_six_slot_dag_is_closed(self) -> None:
        chain = self.report["authorized_terminal_activation_chain"]
        self.assertEqual(chain["model_count"], 2)
        self.assertEqual(chain["required_slot_count"], 6)
        self.assertFalse(
            chain["authorized_queue_entrypoint_terminal_activation_reachable"]
        )
        self.assertEqual(chain["end_to_end_activation_reachable_slot_count"], 0)
        self.assertEqual(chain["diagnostic_only_unreachable_slot_count"], 2)
        self.assertEqual(chain["downstream_blocked_slot_count"], 4)
        self.assertEqual(chain["runtime_byte_verified_slot_count"], 0)
        self.assertEqual(chain["semantic_replay_complete_slot_count"], 0)
        rows = [node for item in chain["chains"] for node in item["nodes"]]
        self.assertTrue(all(
            not item["authorized_queue_entrypoint_terminal_activation_reachable"]
            for item in chain["chains"]
        ))
        self.assertEqual(len(rows), 6)
        self.assertEqual(
            {row["slot_id"] for row in rows},
            {
                "N3.camera_crop_contract", "N3.physical_alignment_receipt",
                "N3.alignment_contract", "D1.camera_crop_contract",
                "D1.physical_alignment_receipt", "D1.alignment_contract",
            },
        )
        self.assertTrue(all(not row["runtime_byte_verified"] for row in rows))
        self.assertTrue(all(not row["semantic_replay_complete"] for row in rows))

    def test_activation_counts_and_scientific_gates_remain_closed(self) -> None:
        snapshot = self.report["retained_activation_snapshot"]
        self.assertEqual(
            snapshot,
            {
                "required_runtime_slots": 14,
                "unauthenticated_runtime_slots": 6,
                "runtime_byte_verified_slots": 0,
                "semantic_replay_complete_slots": 0,
                "request_field_values_authenticated": 4,
                "request_field_values_missing": 26,
                "complete_request_rows": 0,
                "total_request_rows": 1152,
                "sampling_key_feasible_rows": 0,
                "eligibility_feasible_rows": 0,
                "selected_request_rows": 0,
            },
        )
        self.assertEqual(
            self.report["global_study_snapshot"], audit.EXPECTED_GLOBAL_SNAPSHOT
        )
        self.assertEqual(self.report["gate_state"]["core_episode_ceiling"], 232)
        self.assertFalse(
            self.report["gate_state"]["current_GM_PVC_worker_state_reconciled"]
        )
        self.assertFalse(self.report["gate_state"]["live_quiescence_inferred"])
        self.assertFalse(self.report["gate_state"]["human_annotation_ready"])
        self.assertFalse(self.report["gate_state"]["resource_freeze_ready"])
        self.assertEqual(
            self.report["gate_state"]["retained_snapshot_active_launched_cells"], 0
        )
        self.assertFalse(self.report["gate_state"]["safe_to_release_new_work"])
        self.assertEqual(self.report["new_activity"], audit.ZERO_ACTIVITY)

    def test_decision_does_not_invent_attempt003_or_authority(self) -> None:
        decision = self.report["decision"]
        self.assertEqual(
            decision["status"],
            "pinned_authorized_terminal_chain_cannot_authenticate_and_activate_"
            "six_missing_slots",
        )
        self.assertFalse(
            decision["authorized_queue_entrypoint_terminal_activation_reachable"]
        )
        self.assertFalse(
            decision[
                "camera_attempt_003_or_later_authenticated_by_pinned_activation_evidence"
            ]
        )
        self.assertFalse(decision["safe_to_dispatch_replacement"])
        self.assertFalse(decision["safe_to_materialize_annotation_package"])
        self.assertFalse(decision["safe_to_release"])
        boundaries = self.report["claim_boundaries"]
        self.assertTrue(boundaries["static_reachability_is_not_runtime_observation"])
        self.assertTrue(boundaries["prospective_output_names_are_not_existing_artifacts"])
        self.assertIn(
            "authority to dispatch camera attempt 003 or any replacement work",
            boundaries["not_established"],
        )

    def test_duplicate_and_nonfinite_json_fail_closed(self) -> None:
        with self.assertRaisesRegex(audit.AuditError, "Duplicate JSON key"):
            audit.load_json(b'{"a":1,"a":2}', "duplicate")
        for value in (b"NaN", b"Infinity", b"-Infinity"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(audit.AuditError, "Non-finite JSON"):
                    audit.load_json(b'{"a":' + value + b"}", "nonfinite")

    def test_any_pinned_source_mutation_is_rejected(self) -> None:
        key = next(iter(audit.EXPECTED_SOURCES))
        changed = dict(self.blobs)
        changed[key] += b"\n"
        with self.assertRaisesRegex(audit.AuditError, "Pinned source changed"):
            audit.verify_pinned_blobs(changed)
        changed = dict(self.blobs)
        changed.pop(key)
        with self.assertRaisesRegex(audit.AuditError, "inventory changed"):
            audit.verify_pinned_blobs(changed)

    def test_camera_publication_and_diagnostic_mutations_are_rejected(self) -> None:
        source = self.blobs[(audit.CAMERA_COMMIT, audit.CAMERA_JOBS_PATH)].decode()
        changed = source.replace(
            "contracts, launches, diagnostics = _launch_children(",
            "_publish_contracts(publish, {})\n        "
            "contracts, launches, diagnostics = _launch_children(",
            1,
        )
        self.assertNotEqual(changed, source)
        with self.assertRaisesRegex(audit.AuditError, "publisher gained a call site"):
            audit.analyze_camera_source(changed, self.camera_contract)

        changed = source.replace(
            "contracts, launches, diagnostics = _launch_children(",
            "publish_fn = _publish_contracts\n        "
            "publish_fn(publish, contracts)\n        "
            "contracts, launches, diagnostics = _launch_children(",
            1,
        )
        self.assertNotEqual(changed, source)
        with self.assertRaisesRegex(audit.AuditError, "publisher gained a name reference"):
            audit.analyze_camera_source(changed, self.camera_contract)

        contract = copy.deepcopy(self.camera_contract)
        contract["child_runtime"]["diagnostic_only"] = False
        with self.assertRaisesRegex(audit.AuditError, "diagnostic-only"):
            audit.analyze_camera_source(source, contract)
        contract = copy.deepcopy(self.camera_contract)
        contract["child_runtime"]["publish_crop_contracts"] = True
        with self.assertRaisesRegex(audit.AuditError, "publication boundary"):
            audit.analyze_camera_source(source, contract)

    def test_alignment_attempt_and_output_mutations_are_rejected(self) -> None:
        source = self.blobs[(audit.ALIGNMENT_COMMIT, audit.ALIGNMENT_JOBS_PATH)].decode()
        changed = source.replace("ordinal >= 3", "ordinal >= 2", 1)
        self.assertNotEqual(changed, source)
        with self.assertRaisesRegex(audit.AuditError, "control flow"):
            audit.analyze_alignment_source(changed, self.alignment_contract)

        changed = source.replace(
            "def _camera_attempt(job_id: Any) -> int:\n",
            "def _camera_attempt(job_id: Any) -> int:\n    return 3\n",
            1,
        )
        self.assertNotEqual(changed, source)
        with self.assertRaisesRegex(audit.AuditError, "control flow"):
            audit.analyze_alignment_source(changed, self.alignment_contract)

        changed = source.replace(
            '"camera-crop-replay-witness-002",', "", 1
        )
        self.assertNotEqual(changed, source)
        with self.assertRaisesRegex(audit.AuditError, "forbidden camera attempts"):
            audit.analyze_alignment_source(changed, self.alignment_contract)

        changed = source.replace(
            '"n3_alignment_contract.json"', '"n3_alignment_contract_v2.json"', 1
        )
        self.assertNotEqual(changed, source)
        with self.assertRaisesRegex(audit.AuditError, "output names"):
            audit.analyze_alignment_source(changed, self.alignment_contract)

    def test_upstream_activation_promotions_are_rejected(self) -> None:
        activation = audit.load_json(
            self.blobs[(audit.ACTIVATION_COMMIT, audit.ACTIVATION_PATH)],
            "activation",
        )
        cases = []
        changed = copy.deepcopy(activation)
        changed["runtime_slot_activation_matrix"]["runtime_byte_verified_slots"] = 1
        cases.append((changed, "runtime closure"))
        changed = copy.deepcopy(activation)
        changed["request_field_provenance"]["complete_rows"] = 1
        cases.append((changed, "field coverage"))
        changed = copy.deepcopy(activation)
        changed["selection_contract"]["observed_selected_requests"] = 1
        cases.append((changed, "selection state"))
        changed = copy.deepcopy(activation)
        changed["gate_state"]["live_quiescence_inferred"] = True
        cases.append((changed, "gate state"))
        changed = copy.deepcopy(activation)
        rows = changed["runtime_slot_activation_matrix"]["slots"]
        row = next(item for item in rows if item["slot_id"] == "N3.camera_crop_contract")
        row["prospective_output"]["producer_commit"] = audit.ALIGNMENT_COMMIT
        cases.append((changed, "prospective slot bindings"))
        changed = copy.deepcopy(activation)
        rows = changed["runtime_slot_activation_matrix"]["slots"]
        d1 = next(item for item in rows if item["slot_id"] == "D1.camera_crop_contract")
        n3 = next(item for item in rows if item["slot_id"] == "N3.camera_crop_contract")
        d1["prospective_output"], n3["prospective_output"] = (
            n3["prospective_output"], d1["prospective_output"]
        )
        cases.append((changed, "prospective slot bindings"))
        for changed, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(audit.AuditError, message):
                    audit.validate_activation_report(changed)

    def test_report_gate_promotions_are_rejected(self) -> None:
        cases = []
        changed = copy.deepcopy(self.report)
        changed["authorized_terminal_activation_chain"][
            "authorized_queue_entrypoint_terminal_activation_reachable"
        ] = True
        cases.append((changed, "activation chain changed"))
        changed = copy.deepcopy(self.report)
        changed["authorized_terminal_activation_chain"][
            "runtime_byte_verified_slot_count"
        ] = 1
        cases.append((changed, "activation chain changed"))
        changed = copy.deepcopy(self.report)
        changed["decision"]["safe_to_dispatch_replacement"] = True
        cases.append((changed, "Decision claim changed"))
        changed = copy.deepcopy(self.report)
        changed["new_activity"]["labels"] = 1
        cases.append((changed, "science activity"))
        changed = copy.deepcopy(self.report)
        changed["gate_state"]["safe_to_release_new_work"] = True
        cases.append((changed, "gate state"))
        changed = copy.deepcopy(self.report)
        changed["authorized_terminal_activation_chain"]["chains"][0][
            "authorized_queue_entrypoint_terminal_activation_reachable"
        ] = True
        cases.append((changed, "activation chain changed"))
        changed = copy.deepcopy(self.report)
        changed["authorized_terminal_activation_chain"]["chains"][0]["nodes"][0][
            "state"
        ] = "runtime_artifact_verified"
        cases.append((changed, "activation chain changed"))
        changed = copy.deepcopy(self.report)
        changed["authorized_terminal_activation_chain"]["chains"][0]["nodes"][0][
            "producer_job_id"
        ] = "camera-crop-replay-witness-003"
        cases.append((changed, "activation chain changed"))
        changed = copy.deepcopy(self.report)
        changed["camera_producer"]["publisher_call_site_count"] = 1
        cases.append((changed, "Camera producer summary changed"))
        changed = copy.deepcopy(self.report)
        changed["camera_producer"][
            "success_receipt_reachable_from_authorized_run_job_entrypoint"
        ] = True
        cases.append((changed, "Camera producer summary changed"))
        changed = copy.deepcopy(self.report)
        changed["decision"]["status"] = "ready_to_activate"
        cases.append((changed, "Decision claim changed"))
        changed = copy.deepcopy(self.report)
        changed["audit_activity"]["external_system_reads"] = 1
        cases.append((changed, "Audit activity changed"))
        changed = copy.deepcopy(self.report)
        changed["authenticated_source_blobs"][
            "canonical_framed_inventory_sha256"
        ] = "0" * 64
        cases.append((changed, "source inventory changed"))
        changed = copy.deepcopy(self.report)
        changed["authenticated_source_blobs"][
            "canonical_framed_inventory_bytes"
        ] += 1
        cases.append((changed, "source inventory changed"))
        changed = copy.deepcopy(self.report)
        changed["authenticated_source_blobs"]["sources"].append(
            copy.deepcopy(changed["authenticated_source_blobs"]["sources"][0])
        )
        cases.append((changed, "source inventory changed"))
        changed = copy.deepcopy(self.report)
        changed["camera_producer"]["functions"].append(
            copy.deepcopy(changed["camera_producer"]["functions"][0])
        )
        cases.append((changed, "Camera function identities changed"))
        for changed, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(audit.AuditError, message):
                    audit.validate_output_contract(changed)

    def test_audit_is_ast_only_and_does_not_import_producers(self) -> None:
        source = MODULE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            alias.name.split(".", 1)[0]
            for node in tree.body if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported.update(
            node.module.split(".", 1)[0]
            for node in tree.body if isinstance(node, ast.ImportFrom)
            and node.module is not None
        )
        for banned in (
            "importlib", "runpy", "requests", "urllib", "socket", "kubernetes"
        ):
            self.assertNotIn(banned, imported)
        calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
        for banned in (
            "build_wave", "build_wave_from_local_files", "run_job",
            "validate_local_prerequisites", "_publish_contracts",
        ):
            self.assertNotIn(banned, calls)
        activity = self.report["audit_activity"]
        self.assertEqual(activity["producer_modules_imported"], 0)
        self.assertEqual(activity["producer_functions_invoked"], 0)
        self.assertEqual(activity["external_system_reads"], 0)
        self.assertEqual(activity["raw_paths_opened"], 0)
        self.assertEqual(activity["runtime_bytes_read"], 0)

    def test_cli_check_and_hash_seed_determinism(self) -> None:
        subprocess.run(
            [sys.executable, str(MODULE), "--check", "--output", str(REPORT)],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        expected = REPORT.read_bytes()
        for seed in ("0", "1", "987654"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            completed = subprocess.run(
                [sys.executable, str(MODULE)],
                cwd=ROOT,
                env=env,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(completed.stdout, expected)


if __name__ == "__main__":
    unittest.main()
