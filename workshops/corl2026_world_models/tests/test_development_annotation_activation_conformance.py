from __future__ import annotations

import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


WORKSHOP = Path(__file__).resolve().parents[1]
ROOT = WORKSHOP.parents[1]
MODULE = (
    WORKSHOP / "analysis/audit_development_annotation_activation_conformance.py"
)
REPORT = (
    WORKSHOP / "results/development_annotation_activation_conformance.json"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_development_annotation_activation_conformance", MODULE
)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class DevelopmentAnnotationActivationConformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.blobs = audit.load_pinned_blobs()
        cls.report = audit.build_report(cls.blobs)

    def test_saved_report_is_exact_canonical_regeneration(self) -> None:
        saved = REPORT.read_bytes()
        self.assertEqual(saved, audit.canonical_bytes(self.report))
        self.assertEqual(audit.load_json(saved, "saved report"), self.report)
        self.assertEqual(audit.build_report(self.blobs), self.report)

    def test_exact_fifteen_blob_authentication(self) -> None:
        sources = self.report["authenticated_source_blobs"]
        self.assertEqual(sources["count"], 15)
        self.assertEqual(sources["source_bytes"], 1711890)
        self.assertEqual(len(sources["sources"]), 15)
        self.assertEqual(
            {(row["commit"], row["path"]) for row in sources["sources"]},
            set(audit.EXPECTED_SOURCES),
        )
        self.assertEqual(
            self.report["audit_activity"]["authenticated_git_blobs_read"], 15
        )
        self.assertEqual(
            self.report["audit_activity"]["authenticated_git_bytes_read"],
            1711890,
        )

    def test_full_nine_file_runtime_implementation_is_bound(self) -> None:
        entrypoints = self.report["existing_pipeline_entrypoints"]
        self.assertEqual(entrypoints["implementation_source_path_count"], 9)
        inventory = entrypoints["implementation_source_inventory"]
        self.assertEqual(set(inventory), set(audit.EXPECTED_IMPLEMENTATION_PATHS))
        self.assertEqual(
            {name: row["path"] for name, row in inventory.items()},
            audit.EXPECTED_IMPLEMENTATION_PATHS,
        )
        self.assertEqual(
            {row["commit"] for row in inventory.values()},
            {audit.SELECTION_COMMIT},
        )
        self.assertEqual(
            entrypoints["implementation_source_path_expressions"],
            audit.EXPECTED_SOURCE_PATH_EXPRESSIONS,
        )
        self.assertFalse(entrypoints["functions_imported_by_audit"])
        self.assertFalse(entrypoints["functions_invoked_by_audit"])
        self.assertTrue(
            entrypoints[
                "builder_release_flag_is_future_descriptor_behavior_not_current_release"
            ]
        )

    def test_entrypoint_hashes_and_prepare_call_graph_are_exact(self) -> None:
        functions = {
            (row["path"], row["name"]): row
            for row in self.report["existing_pipeline_entrypoints"]["functions"]
        }
        self.assertEqual(set(functions), set(audit.EXPECTED_FUNCTIONS))
        for key, expected in audit.EXPECTED_FUNCTIONS.items():
            row = functions[key]
            self.assertEqual(
                (row["start_line"], row["end_line"], row["bytes"], row["sha256"]),
                expected,
            )
        self.assertEqual(
            self.report["existing_pipeline_entrypoints"]["prepare_static_call_graph"],
            {
                "_build_request_inventory": 1,
                "annotation.select_requests": 2,
                "_assert_inventory_and_selection": 2,
                "_render_selected": 1,
            },
        )

    def test_runtime_slot_partition_and_prospective_producers_are_exact(self) -> None:
        matrix = self.report["runtime_slot_activation_matrix"]
        self.assertEqual(
            {key: matrix[key] for key in (
                "required_slots", "identity_bound_slots",
                "recovery_manifest_pointer_slots",
                "retained_git_payload_slots_outside_recovery_manifest",
                "unauthenticated_slots",
                "runtime_byte_verified_slots", "semantic_replay_complete_slots",
            )},
            {
                "required_slots": 14,
                "identity_bound_slots": 8,
                "recovery_manifest_pointer_slots": 5,
                "retained_git_payload_slots_outside_recovery_manifest": 3,
                "unauthenticated_slots": 6,
                "runtime_byte_verified_slots": 0,
                "semantic_replay_complete_slots": 0,
            },
        )
        self.assertEqual(
            matrix["source_state_counts"],
            {
                "declared_raw_pointer_in_recovery_manifest_current_availability_unknown": 5,
                "retained_git_blob_outside_recovery_manifest_runtime_materialization_unverified": 3,
                "no_authenticated_descriptor_in_pinned_sources": 6,
            },
        )
        self.assertEqual(
            matrix["recovery_manifest_pointer_slots"]
            + matrix["retained_git_payload_slots_outside_recovery_manifest"]
            + matrix["unauthenticated_slots"],
            matrix["required_slots"],
        )
        rows = matrix["slots"]
        self.assertEqual(len(rows), 14)
        self.assertEqual(len({row["slot_id"] for row in rows}), 14)
        prospective = {
            row["slot_id"]: row["prospective_output"]["output_name"]
            for row in rows if row["prospective_output"] is not None
        }
        self.assertEqual(
            prospective,
            {
                "D1.alignment_contract": "d1_alignment_contract.json",
                "D1.camera_crop_contract": "d1_camera_crop_contract.json",
                "D1.physical_alignment_receipt":
                    "d1_physical_alignment_receipt.json",
                "N3.alignment_contract": "n3_alignment_contract.json",
                "N3.camera_crop_contract": "n3_camera_crop_contract.json",
                "N3.physical_alignment_receipt":
                    "n3_physical_alignment_receipt.json",
            },
        )
        self.assertTrue(all(not row["runtime_byte_verified"] for row in rows))
        self.assertTrue(all(not row["semantic_replay_complete"] for row in rows))
        self.assertTrue(all(
            row["prospective_output_is_not_existing_evidence"]
            for row in rows if row["prospective_output"] is not None
        ))

    def test_request_field_partition_and_builder_expressions_are_exact(self) -> None:
        fields = self.report["request_field_provenance"]
        self.assertEqual(fields["field_count"], 30)
        self.assertEqual(
            {name: row["count"] for name, row in fields["groups"].items()},
            {
                "retained_mapping": 4,
                "request_provenance": 9,
                "alignment_contract": 11,
                "physical_mapping_identity": 1,
                "cross_source_derivation": 5,
            },
        )
        groups = [set(row["fields"]) for row in fields["groups"].values()]
        self.assertEqual(sum(len(group) for group in groups), 30)
        self.assertEqual(len(set().union(*groups)), 30)
        self.assertEqual(
            tuple(sorted(set().union(*groups))), audit.EXPECTED_REQUEST_KEYS
        )
        self.assertEqual(
            fields["builder_row_expressions"], audit.EXPECTED_ROW_EXPRESSIONS
        )
        self.assertEqual(fields["authenticated_value_field_count"], 4)
        self.assertEqual(fields["missing_value_field_count"], 26)
        self.assertEqual(fields["complete_rows"], 0)
        self.assertEqual(fields["total_rows"], 1152)
        self.assertFalse(fields["identity_substitution_permitted"])

    def test_inventory_and_selection_gates_remain_closed(self) -> None:
        inventory = self.report["request_inventory_contract"]
        self.assertEqual(inventory["rows"], 1152)
        self.assertEqual(inventory["cells"], 32)
        self.assertEqual(inventory["rows_by_model"], {"D1": 912, "N3": 240})
        self.assertEqual(inventory["request_schema_field_count"], 30)
        self.assertEqual(
            tuple(inventory["inventory_dictionary_fields"]),
            audit.EXPECTED_INVENTORY_FIELDS,
        )
        self.assertTrue(inventory["complete_inventory_required"])
        self.assertEqual(inventory["required_annotation_state"], "not_started")
        self.assertEqual(inventory["request_rows_materialized_by_audit"], 0)
        selection = self.report["selection_contract"]
        self.assertEqual(selection["seed"], 2026091302)
        self.assertEqual(selection["cap_per_episode"], 4)
        self.assertEqual(selection["algorithm"], audit.EXPECTED_ALGORITHM)
        self.assertEqual(selection["sampling_key_feasible_rows"], 0)
        self.assertEqual(selection["eligibility_feasible_rows"], 0)
        self.assertEqual(selection["eligibility_status"], "undetermined_not_ineligible")
        self.assertEqual(selection["expected_eligible_requests_contract_only"], 448)
        self.assertEqual(selection["expected_selected_requests_contract_only"], 128)
        self.assertEqual(selection["observed_selected_requests"], 0)
        self.assertFalse(selection["selection_manifest_created"])
        self.assertFalse(selection["uses_labels_visibility_or_forecast_quality"])

    def test_exact_science_counters_and_activation_decision(self) -> None:
        self.assertEqual(
            self.report["global_study_snapshot"], audit.EXPECTED_GLOBAL_SNAPSHOT
        )
        self.assertEqual(self.report["gate_state"]["core_episode_ceiling"], 232)
        self.assertEqual(
            self.report["gate_state"]["retained_snapshot_active_launched_cells"], 0
        )
        self.assertFalse(
            self.report["gate_state"]["current_GM_PVC_worker_state_reconciled"]
        )
        self.assertFalse(self.report["gate_state"]["live_quiescence_inferred"])
        self.assertFalse(self.report["gate_state"]["safe_to_release_new_work"])
        self.assertEqual(set(self.report["new_activity"].values()), {0})
        decision = self.report["activation_decision"]
        self.assertEqual(
            decision["status"],
            "not_activatable_from_authenticated_retained_evidence",
        )
        self.assertTrue(decision["pipeline_dependency_graph_fully_accounted_for"])
        self.assertEqual(decision["authenticated_field_value_coverage"], "4/30")
        self.assertEqual(decision["complete_rows"], "0/1152")
        self.assertEqual(decision["runtime_byte_verified_slots"], "0/14")
        self.assertEqual(decision["semantic_replay_complete_slots"], "0/14")
        self.assertFalse(decision["safe_to_release"])
        self.assertIn(
            "current GM PVC queue worker or live-quiescence state",
            self.report["claim_boundaries"]["not_established"],
        )

    def test_duplicate_and_nonfinite_json_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            audit.load_json(b'{"a":1,"a":2}', "duplicate")
        for value in (b'NaN', b'Infinity', b'-Infinity'):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "Non-finite JSON constant"):
                    audit.load_json(b'{"a":' + value + b'}', "nonfinite")

    def test_any_pinned_source_mutation_is_rejected(self) -> None:
        key = next(iter(audit.EXPECTED_SOURCES))
        changed = dict(self.blobs)
        changed[key] = changed[key] + b"\n"
        with self.assertRaisesRegex(ValueError, "Pinned source changed"):
            audit.verify_pinned_blobs(changed)
        changed = dict(self.blobs)
        changed.pop(key)
        with self.assertRaisesRegex(ValueError, "Pinned source inventory changed"):
            audit.verify_pinned_blobs(changed)

    def test_builder_expression_and_function_mutation_are_rejected(self) -> None:
        source = self.blobs[(audit.SELECTION_COMMIT, audit.ASSEMBLER_PATH)].decode()
        changed = source.replace(
            '"episode_id": source["recording_id"]',
            '"episode_id": source["cell_id"]',
            1,
        )
        self.assertNotEqual(changed, source)
        with self.assertRaisesRegex(ValueError, "Request row expressions changed"):
            audit.request_row_expressions(changed)
        workflow = self.blobs[(audit.SELECTION_COMMIT, audit.WORKFLOW_PATH)].decode()
        changed = workflow.replace(
            'return sha256_bytes(payload), row["source_request_id"]',
            'return sha256_bytes(payload), row["episode_id"]',
            1,
        )
        self.assertNotEqual(changed, workflow)
        with self.assertRaisesRegex(ValueError, "Function changed"):
            audit.function_descriptor(
                changed, audit.WORKFLOW_PATH, "select_requests"
            )

    def test_prospective_output_drift_is_rejected(self) -> None:
        alignment = self.blobs[
            (audit.ALIGNMENT_COMMIT, audit.ALIGNMENT_JOBS_PATH)
        ].decode()
        crop = self.blobs[(audit.CROP_COMMIT, audit.CROP_JOBS_PATH)].decode()
        changed = crop.replace(
            "n3_camera_crop_contract.json", "n3_camera_crop_contract_v2.json", 1
        )
        with self.assertRaisesRegex(ValueError, "Crop producer outputs changed"):
            audit.producer_outputs(alignment, changed)

    def test_report_gate_promotions_are_rejected(self) -> None:
        cases = []
        changed = copy.deepcopy(self.report)
        changed["activation_decision"]["safe_to_release"] = True
        cases.append((changed, "Activation release promoted"))
        changed = copy.deepcopy(self.report)
        changed["runtime_slot_activation_matrix"]["runtime_byte_verified_slots"] = 1
        cases.append((changed, "runtime verification promoted"))
        changed = copy.deepcopy(self.report)
        changed["request_field_provenance"]["complete_rows"] = 1
        cases.append((changed, "complete rows promoted"))
        changed = copy.deepcopy(self.report)
        changed["selection_contract"]["observed_selected_requests"] = 1
        cases.append((changed, "selection promoted"))
        changed = copy.deepcopy(self.report)
        changed["new_activity"]["labels"] = 1
        cases.append((changed, "science activity changed"))
        for changed, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    audit.validate_output_contract(changed)

    def test_upstream_report_promotions_are_rejected(self) -> None:
        selection = audit.load_json(
            self.blobs[(audit.SELECTION_COMMIT, audit.SELECTION_PATH)], "selection"
        )
        changed = copy.deepcopy(selection)
        changed["selection_readiness"]["sampling_key_feasible_rows"] = 1
        with self.assertRaisesRegex(ValueError, "Sampling feasibility promoted"):
            audit.validate_selection_report(changed)
        media = audit.load_json(
            self.blobs[(audit.SELECTION_COMMIT, audit.MEDIA_CLOSURE_PATH)], "media"
        )
        changed = copy.deepcopy(media)
        changed["top_level_input_closure"]["semantic_replay_complete_slots"] = 1
        with self.assertRaisesRegex(ValueError, "Runtime closure counts changed"):
            audit.validate_media_report(changed)

    def test_audit_has_no_runtime_import_or_builder_invocation(self) -> None:
        source = MODULE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_roots = {
            alias.name.split(".", 1)[0]
            for node in tree.body if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertNotIn("importlib", imported_roots)
        self.assertNotIn("runpy", imported_roots)
        self.assertNotIn("os", imported_roots)
        banned = {"build_formal_wave", "prepare", "select_requests"}
        calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
        self.assertTrue(banned.isdisjoint(calls))
        subprocess_calls = [
            ast.unparse(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
        ]
        self.assertEqual(subprocess_calls, ["subprocess.check_output"])
        self.assertNotIn("/data/", source)

    def test_cli_check_accepts_only_canonical_report(self) -> None:
        subprocess.run(
            [sys.executable, str(MODULE), "--check", str(REPORT)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            altered = Path(directory) / "altered.json"
            document = json.loads(REPORT.read_text())
            document["activation_decision"]["safe_to_release"] = True
            altered.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
            process = subprocess.run(
                [sys.executable, str(MODULE), "--check", str(altered)],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("Saved report is not canonical or current", process.stderr)

    def test_cli_output_is_hash_seed_deterministic(self) -> None:
        outputs = []
        for seed in ("1", "17", "991"):
            environment = dict(os.environ)
            environment["PYTHONHASHSEED"] = seed
            outputs.append(subprocess.check_output(
                [sys.executable, str(MODULE)], cwd=ROOT, env=environment
            ))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[1], outputs[2])
        self.assertEqual(outputs[0], REPORT.read_bytes())


if __name__ == "__main__":
    unittest.main()
