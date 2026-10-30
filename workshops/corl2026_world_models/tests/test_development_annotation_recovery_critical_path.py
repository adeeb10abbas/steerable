from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "analysis/audit_development_annotation_recovery_critical_path.py"
)
REPORT_PATH = (
    ROOT / "results/development_annotation_recovery_critical_path.json"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_development_annotation_recovery_critical_path", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def replace_once(payload: bytes, old: bytes, new: bytes) -> bytes:
    if payload.count(old) != 1:
        raise AssertionError(f"Expected one mutation target, found {payload.count(old)}: {old!r}")
    return payload.replace(old, new, 1)


class DevelopmentAnnotationRecoveryCriticalPathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.camera = audit.git_blob(audit.CAMERA_SOURCE_COMMIT, audit.CAMERA_SOURCE_PATH)
        cls.alignment = audit.git_blob(
            audit.ALIGNMENT_SOURCE_COMMIT, audit.ALIGNMENT_SOURCE_PATH
        )
        cls.queue = audit.git_blob(audit.QUEUE_SOURCE_COMMIT, audit.QUEUE_SOURCE_PATH)
        cls.report = json.loads(REPORT_PATH.read_text())

    def test_saved_report_is_exact_canonical_regeneration(self) -> None:
        self.assertEqual(REPORT_PATH.read_bytes(), audit.canonical_bytes(audit.audit()))

    def test_report_has_two_independent_open_cuts(self) -> None:
        critical = self.report["runtime_critical_path"]
        self.assertEqual(critical["partition_equation"], "6 + 5 + 3 = 14")
        self.assertEqual(critical["producer_partition_equation"], "2 + 4 = 6")
        self.assertEqual(critical["required_slots"], 14)
        self.assertEqual(critical["producer_chain_slots"], 6)
        self.assertEqual(critical["identity_bound_runtime_slots"], 8)
        self.assertFalse(critical["camera_recovery_independently_sufficient"])
        self.assertFalse(critical["runtime_payload_recovery_independently_sufficient"])
        self.assertEqual(
            [row.get("still_open_slots") for row in critical["counterfactuals"][:2]],
            [8, 6],
        )
        self.assertTrue(
            all(not row["runnable_manifest"] for row in critical["counterfactuals"])
        )
        self.assertTrue(
            all(
                row["operational_feasibility_or_order_asserted"] is False
                for row in critical["counterfactuals"]
            )
        )
        self.assertTrue(
            all(row["materializable_request_rows"] == 0 for row in critical["counterfactuals"])
        )

    def test_current_request_closure_is_not_promoted(self) -> None:
        closure = self.report["current_request_closure"]
        self.assertEqual(
            closure,
            {
                "authenticated_field_values": 4,
                "complete_rows": 0,
                "eligibility_decision_feasible_rows": 0,
                "missing_field_values": 26,
                "rows": 1152,
                "sampling_key_feasible_rows": 0,
                "schema_fields": 30,
                "selected_rows": 0,
                "selection_manifest_created": False,
            },
        )

    def test_core_ceiling_and_accounting_are_unchanged(self) -> None:
        snapshot = self.report["global_study_snapshot"]
        self.assertEqual(snapshot["planned_core_cells"], 232)
        self.assertEqual(snapshot["launched_core_cells"], 44)
        self.assertEqual(snapshot["terminal_valid_cells"], 40)
        self.assertEqual(snapshot["terminal_technical_invalid_cells"], 4)
        self.assertEqual(snapshot["active_launched_cells"], 0)
        self.assertEqual(snapshot["terminal_requests"], 1460)
        self.assertEqual(
            snapshot["nonbehavioral_requests"]
            + snapshot["valid_behavioral_requests"]
            + snapshot["technical_invalid_behavioral_requests"],
            1460,
        )
        self.assertEqual(snapshot["valid_behavioral_actions"], 18000)
        self.assertEqual(snapshot["valid_behavioral_observations"], 18040)
        self.assertEqual(snapshot["confirmation_inference_cells"], 0)

    def test_all_activity_and_release_gates_remain_zero_or_false(self) -> None:
        self.assertEqual(self.report["new_activity"], audit.EXPECTED_NEW_ACTIVITY)
        self.assertTrue(all(value == 0 for value in self.report["new_activity"].values()))
        decision = self.report["decision"]
        self.assertFalse(decision["runnable_annotation_media_manifest"])
        self.assertFalse(decision["safe_to_materialize_camera_replacement_descriptor"])
        self.assertFalse(decision["safe_to_dispatch_camera_replacement"])
        self.assertFalse(decision["safe_to_release_new_work"])
        self.assertFalse(decision["safe_to_materialize_annotation_package"])
        for key, value in self.report["gate_state"].items():
            if key not in {"core_episode_ceiling", "retained_snapshot_active_launched_cells"}:
                self.assertFalse(value, key)

    def test_exact_pin_and_tree_indexes(self) -> None:
        self.assertEqual(
            self.report["authenticated_source_blobs"], audit.PIN_INDEX
        )
        self.assertEqual(
            self.report["pinned_forecast_layout_python_tree"],
            {"tree": audit.FORECAST_SOURCE_TREE, **audit.FORECAST_PYTHON_INDEX},
        )

    def test_pinned_blob_rejects_content_drift(self) -> None:
        original = audit.git_blob(audit.CAMERA_SOURCE_COMMIT, audit.CAMERA_SOURCE_PATH)
        with mock.patch.object(audit, "git_blob", return_value=original + b" "):
            with self.assertRaisesRegex(ValueError, "Pinned blob changed"):
                audit.pinned_blob(audit.CAMERA_SOURCE_COMMIT, audit.CAMERA_SOURCE_PATH)

    def test_strict_json_rejects_duplicate_and_nonfinite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            audit.load_json(b'{"x": 1, "x": 2}', "duplicate")
        with self.assertRaisesRegex(ValueError, "Non-finite JSON constant"):
            audit.load_json(b'{"x": NaN}', "nonfinite")

    def test_camera_source_is_diagnostic_attempt_002(self) -> None:
        value = audit.analyze_camera_source(self.camera)
        self.assertEqual(value["job_id"], "camera-crop-replay-witness-002")
        self.assertEqual(value["attempt_ordinal"], 2)
        self.assertTrue(value["diagnostic_only"])
        self.assertFalse(value["publish_crop_contracts"])
        self.assertEqual(value["publisher_reference_count_from_authorized_run_job"], 0)
        self.assertFalse(value["authorized_terminal_success_path"])

    def test_camera_source_rejects_ordinal_only_rename(self) -> None:
        changed = replace_once(
            self.camera,
            b'JOB_ID = "camera-crop-replay-witness-002"',
            b'JOB_ID = "camera-crop-replay-witness-003"',
        )
        with self.assertRaisesRegex(ValueError, "Camera producer job changed"):
            audit.analyze_camera_source(changed)

    def test_camera_source_rejects_direct_publisher_escape(self) -> None:
        marker = b"    descriptor, expected_implementation = _runtime_descriptor(args)"
        changed = replace_once(
            self.camera,
            marker,
            marker + b"\n    _publish_contracts(Path('/tmp'), {})",
        )
        with self.assertRaisesRegex(ValueError, "Function changed|gained a publisher reference"):
            audit.analyze_camera_source(changed)

    def test_camera_source_rejects_aliased_publisher_escape(self) -> None:
        marker = b"    descriptor, expected_implementation = _runtime_descriptor(args)"
        changed = replace_once(
            self.camera,
            marker,
            marker + b"\n    escaped_publisher = _publish_contracts",
        )
        with self.assertRaisesRegex(ValueError, "Function changed|gained a publisher reference"):
            audit.analyze_camera_source(changed)

    def test_camera_source_rejects_early_success_return(self) -> None:
        marker = b"    descriptor, expected_implementation = _runtime_descriptor(args)"
        changed = replace_once(
            self.camera,
            marker,
            b'    return {"status": "passed"}\n' + marker,
        )
        with self.assertRaisesRegex(ValueError, "Function changed"):
            audit.analyze_camera_source(changed)

    def test_camera_source_rejects_indirect_helper_escape(self) -> None:
        marker = b"    descriptor, expected_implementation = _runtime_descriptor(args)"
        changed = replace_once(
            self.camera,
            marker,
            marker + b"\n    escaped_publisher(publish, contracts)",
        )
        changed += b"\n\ndef escaped_publisher(publish, contracts):\n    return _publish_contracts(publish, contracts)\n"
        with self.assertRaisesRegex(ValueError, "Function changed"):
            audit.analyze_camera_source(changed)

    def test_camera_source_rejects_output_name_drift(self) -> None:
        changed = self.camera.replace(
            b"n3_camera_crop_contract.json", b"n3_camera_crop_contract_v2.json"
        )
        with self.assertRaisesRegex(ValueError, "Camera output names changed"):
            audit.analyze_camera_source(changed)

    def test_alignment_source_requires_attempt_003_or_later(self) -> None:
        value = audit.analyze_alignment_source(self.alignment)
        self.assertEqual(value["minimum_attempt_ordinal"], 3)
        self.assertEqual(
            value["forbidden_job_ids"],
            ["camera-crop-replay-witness-001", "camera-crop-replay-witness-002"],
        )
        self.assertEqual(value["required_status"], "passed")
        self.assertEqual(
            value["required_decision"], "qualified_from_original_camera_pixels"
        )
        self.assertTrue(value["requires_original_camera_pixels"])
        self.assertTrue(value["requires_safe_for_physical_alignment_input"])
        self.assertEqual(value["required_published_crop_contracts"], 2)

    def test_alignment_source_rejects_lower_attempt_minimum(self) -> None:
        changed = replace_once(self.alignment, b"ordinal >= 3 and", b"ordinal >= 2 and")
        with self.assertRaisesRegex(ValueError, "Function changed|minimum guard changed"):
            audit.analyze_alignment_source(changed)

    def test_alignment_source_rejects_forbidden_attempt_removal(self) -> None:
        changed = replace_once(
            self.alignment,
            b'    "camera-crop-replay-witness-002",\n',
            b"",
        )
        with self.assertRaisesRegex(ValueError, "Forbidden camera attempts changed"):
            audit.analyze_alignment_source(changed)

    def test_alignment_source_rejects_terminal_decision_drift(self) -> None:
        changed = replace_once(
            self.alignment,
            b'        "decision": "qualified_from_original_camera_pixels",',
            b'        "decision": "qualified",',
        )
        with self.assertRaisesRegex(ValueError, "Function changed|qualified_from_original_camera_pixels"):
            audit.analyze_alignment_source(changed)

    def test_alignment_source_rejects_safe_flag_false(self) -> None:
        changed = replace_once(
            self.alignment,
            b'        "safe_for_physical_alignment_input": True,',
            b'        "safe_for_physical_alignment_input": False,',
        )
        with self.assertRaisesRegex(ValueError, "Function changed"):
            audit.analyze_alignment_source(changed)

    def test_alignment_source_rejects_dead_minimum_guard(self) -> None:
        marker = b"    ordinal = int(match.group(1))\n"
        changed = replace_once(
            self.alignment,
            marker,
            marker + b"    if False and ordinal >= 3:\n        pass\n",
        )
        changed = replace_once(changed, b"        ordinal >= 3 and", b"        True and")
        with self.assertRaisesRegex(ValueError, "Function changed"):
            audit.analyze_alignment_source(changed)

    def test_generic_queue_has_no_scientific_authority_fields(self) -> None:
        value = audit.analyze_queue_source(self.queue)
        self.assertEqual(value["descriptor_field_count"], 7)
        self.assertEqual(value["scientific_gate_fields_enforced"], [])
        self.assertTrue(value["generic_queue_admission_is_not_scientific_authority"])

    def test_queue_source_rejects_scientific_gate_literal(self) -> None:
        marker = b"def normalize_job(job):\n"
        changed = replace_once(
            self.queue,
            marker,
            marker + b"    scientific_gate = 'safe_to_release_new_work'\n",
        )
        with self.assertRaisesRegex(ValueError, "Function changed|scientific-gate fields"):
            audit.analyze_queue_source(changed)

    def test_queue_source_rejects_descriptor_field_expansion(self) -> None:
        changed = replace_once(
            self.queue,
            b"'publish_log_tail_bytes'}",
            b"'publish_log_tail_bytes', 'safe_to_release_new_work'}",
        )
        with self.assertRaisesRegex(ValueError, "Function changed|descriptor fields changed"):
            audit.analyze_queue_source(changed)

    def test_queue_source_rejects_claim_cutoff_removal(self) -> None:
        changed = replace_once(
            self.queue,
            b"or (cutoff is not None and time.time() >= cutoff)):",
            b"or False):",
        )
        with self.assertRaisesRegex(ValueError, "Function changed"):
            audit.analyze_queue_source(changed)

    def test_source_tree_inventory_rejects_injected_attempt_003_literal(self) -> None:
        _, blobs = audit.scan_forecast_python_tree()
        changed = dict(blobs)
        changed[audit.CAMERA_SOURCE_PATH] += (
            b'\nINJECTED_CANDIDATE = "camera-crop-replay-witness-003"\n'
        )
        with self.assertRaisesRegex(ValueError, "literal inventory changed"):
            audit.camera_producer_inventory(changed)

    def test_source_tree_inventory_rejects_attempt_003_producer_assignment(self) -> None:
        _, blobs = audit.scan_forecast_python_tree()
        changed = dict(blobs)
        changed[audit.CAMERA_SOURCE_PATH] = replace_once(
            changed[audit.CAMERA_SOURCE_PATH],
            b'JOB_ID = "camera-crop-replay-witness-002"',
            b'JOB_ID = "camera-crop-replay-witness-003"',
        )
        with self.assertRaisesRegex(ValueError, "producer inventory changed"):
            audit.camera_producer_inventory(changed)

    def test_source_tree_inventory_rejects_computed_job_id_override(self) -> None:
        _, blobs = audit.scan_forecast_python_tree()
        changed = dict(blobs)
        changed[audit.CAMERA_SOURCE_PATH] += (
            b'\nJOB_ID = "camera-crop-replay-witness-" + "003"\n'
        )
        with self.assertRaisesRegex(ValueError, "nonliteral top-level JOB_ID"):
            audit.camera_producer_inventory(changed)

    def test_counterfactual_partition_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Critical-path partition changed"):
            audit.counterfactuals(producer_slots=6, identity_bound_slots=7, total=14)
        with self.assertRaisesRegex(ValueError, "Critical-path partition changed"):
            audit.counterfactuals(producer_slots=5, identity_bound_slots=8, total=14)
        with self.assertRaisesRegex(ValueError, "Critical-path partition changed"):
            audit.counterfactuals(producer_slots=6, identity_bound_slots=8, total=13)

    def test_claim_boundaries_reject_live_or_dispatch_promotion(self) -> None:
        boundary = self.report["claim_boundaries"]
        self.assertTrue(boundary["counterfactuals_are_not_observed_state"])
        self.assertTrue(boundary["pinned_subtree_absence_is_not_repo_global_or_external_absence"])
        self.assertTrue(boundary["retained_empty_queue_is_not_live_quiescence"])
        self.assertTrue(boundary["camera_recovery_is_necessary_but_not_independently_sufficient"])
        joined = " ".join(boundary["not_established"])
        self.assertIn("current GM PVC", joined)
        self.assertIn("authority", joined)
        self.assertIn("runtime byte verification", joined)
        self.assertIn("labels or accuracy", joined)

    def test_positive_claim_is_narrowed_to_literal_python_scan(self) -> None:
        encoded = REPORT_PATH.read_text()
        self.assertIn("module-level literal camera producer JOB_ID assignments", encoded)
        self.assertIn("dynamic_or_non_module_level_future_producers_excluded_by_scan", encoded)
        self.assertNotIn("authorized terminal camera replacement candidate", encoded)

    def test_audit_performed_no_external_or_runtime_activity(self) -> None:
        activity = self.report["audit_activity"]
        for key in (
            "external_system_reads",
            "raw_paths_opened",
            "runtime_bytes_read",
            "producer_modules_imported",
            "producer_functions_invoked",
            "queue_descriptors_created",
            "remote_contacts",
        ):
            self.assertEqual(activity[key], 0, key)


if __name__ == "__main__":
    unittest.main()
