from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


WORKSHOP = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    WORKSHOP / "analysis/audit_development_annotation_media_input_closure.py"
)
REPORT_PATH = (
    WORKSHOP / "results/development_annotation_media_input_closure.json"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_development_annotation_media_input_closure", MODULE_PATH
)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(audit)


class DevelopmentAnnotationMediaInputClosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.blobs = audit.load_pinned_blobs()
        cls.report = audit.build_report(cls.blobs)

    def mutated_json_blob(
        self,
        key: tuple[str, str],
        mutate,
        *,
        resign: bool = False,
    ) -> dict[tuple[str, str], bytes]:
        blobs = dict(self.blobs)
        value = audit.load_json(blobs[key], "test mutation")
        mutate(value)
        if resign:
            value.pop("payload_sha256", None)
            value["payload_sha256"] = hashlib.sha256(
                audit.compact_bytes(value)
            ).hexdigest()
        blobs[key] = audit.canonical_bytes(value)
        return blobs

    def test_saved_report_is_exact_canonical_regeneration(self) -> None:
        saved = REPORT_PATH.read_bytes()
        self.assertEqual(saved, audit.canonical_bytes(self.report))
        self.assertEqual(
            audit.load_json(saved, "saved closure report"), self.report
        )
        self.assertEqual(audit.build_report(self.blobs), self.report)

    def test_exact_source_authentication(self) -> None:
        sources = self.report["authenticated_source_blobs"]
        self.assertEqual(sources["count"], 8)
        self.assertEqual(sources["source_bytes"], 892007)
        self.assertEqual(sources["canonical_bytes"], 1604)
        self.assertEqual(
            sources["sha256"],
            "c0dea0a177d96a8f513df555020b5a15ee3b6bdbbef4cde2c9eaa321ce9631e1",
        )
        self.assertEqual(len(sources["sources"]), 8)

    def test_exact_fourteen_slot_partition(self) -> None:
        closure = self.report["top_level_input_closure"]
        self.assertEqual(
            {
                key: closure[key]
                for key in (
                    "required_slots",
                    "exact_identity_bound_slots",
                    "recovery_manifest_pointer_slots",
                    "retained_git_payload_slots_outside_recovery_manifest",
                    "unauthenticated_slots",
                    "current_runtime_byte_verified_slots",
                    "semantic_replay_complete_slots",
                    "recovery_pointer_declared_bytes",
                    "retained_git_payload_bytes",
                )
            },
            {
                "required_slots": 14,
                "exact_identity_bound_slots": 8,
                "recovery_manifest_pointer_slots": 5,
                "retained_git_payload_slots_outside_recovery_manifest": 3,
                "unauthenticated_slots": 6,
                "current_runtime_byte_verified_slots": 0,
                "semantic_replay_complete_slots": 0,
                "recovery_pointer_declared_bytes": 20803049,
                "retained_git_payload_bytes": 122133,
            },
        )
        rows = closure["slots"]
        self.assertEqual(len(rows), 14)
        self.assertEqual(len({row["slot_id"] for row in rows}), 14)
        self.assertEqual(
            closure["direct_recovery_dependency_ids"],
            [
                "dependency-33",
                "dependency-35",
                "dependency-38",
                "dependency-52",
                "dependency-54",
            ],
        )
        self.assertEqual(
            closure["unfilled_required_slot_ids"],
            [
                "D1.alignment_contract",
                "D1.camera_crop_contract",
                "D1.physical_alignment_receipt",
                "N3.alignment_contract",
                "N3.camera_crop_contract",
                "N3.physical_alignment_receipt",
            ],
        )

    def test_media_join_domain_is_exact_and_unique(self) -> None:
        domain = self.report["media_join_domain"]
        self.assertEqual(domain["cells"], 32)
        self.assertEqual(domain["requests"], 1152)
        self.assertEqual(domain["requests_by_model"], {"D1": 912, "N3": 240})
        self.assertEqual(domain["requests_per_cell"], {"D1": 57, "N3": 15})
        self.assertEqual(domain["duplicate_join_keys"], 0)
        self.assertEqual(
            domain["join_key_index"],
            {
                "count": 1152,
                "canonical_bytes": 57280,
                "canonicalization": "ordered UTF-8 rows with LF terminators",
                "sha256": "a736e247c263db5f91639954407e2d47deadce400a8d74af84e002e5f49daac4",
            },
        )

    def test_recovery_substitutions_are_explicitly_rejected(self) -> None:
        boundary = self.report["recovery_manifest_boundary"]
        self.assertEqual(boundary["all_declared_dependencies"], 56)
        self.assertEqual(boundary["all_declared_bytes"], 32664538)
        self.assertEqual(boundary["direct_top_level_media_slots"], 5)
        self.assertEqual(boundary["non_direct_top_level_dependencies"], 51)
        self.assertTrue(
            boundary["compiler_output_inventory_is_not_compiler_job_receipt"]
        )
        self.assertTrue(
            boundary["raw_request_inventories_are_not_timing_job_receipts"]
        )
        self.assertFalse(boundary["transitive_dependency_closure_authenticated"])

    def test_report_cannot_be_mistaken_for_release_or_observation(self) -> None:
        runtime = self.report["runtime_input_contract"]
        self.assertTrue(runtime["emitted_job_descriptor_released"])
        self.assertTrue(runtime["closure_schema_is_not_runtime_input_schema"])
        self.assertFalse(runtime["runnable_input_manifest_created"])
        self.assertFalse(runtime["queue_descriptor_created"])
        self.assertFalse(self.report["gate_state"]["safe_to_release_new_work"])
        self.assertEqual(
            set(self.report["new_activity"].values()), {0}
        )
        expectations = self.report[
            "contract_expectations_not_observed_results"
        ]
        self.assertEqual(expectations["selected_requests_expected"], 128)
        self.assertEqual(expectations["observed_selected_requests"], 0)
        self.assertFalse(expectations["production_eligibility_determined"])
        self.assertFalse(expectations["production_selection_manifest_created"])

    def test_duplicate_and_nonfinite_json_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            audit.load_json(b'{"a":1,"a":2}', "duplicate")
        with self.assertRaisesRegex(ValueError, "Non-finite JSON constant"):
            audit.load_json(b'{"a":NaN}', "nonfinite")

    def test_coherent_status_repin_cannot_open_release(self) -> None:
        key = (audit.STATUS_COMMIT, audit.STATUS_PATH)

        def mutate(value: dict) -> None:
            value["results_publication_incident"][
                "safe_to_release_new_work"
            ] = True

        with self.assertRaisesRegex(ValueError, "Current release incident"):
            audit.build_report(self.mutated_json_blob(key, mutate))

    def test_coherent_contract_repin_cannot_change_join_order(self) -> None:
        key = (audit.MEDIA_COMMIT, audit.MEDIA_CONTRACT_PATH)

        def mutate(value: dict) -> None:
            value["input_policy"]["join_key"] = [
                "cell_id",
                "model_id",
                "request_index",
            ]

        with self.assertRaisesRegex(ValueError, "Media join key"):
            audit.build_report(self.mutated_json_blob(key, mutate))

    def test_coherent_lineage_repin_cannot_substitute_inventory(self) -> None:
        key = (audit.LINEAGE_COMMIT, audit.LINEAGE_PATH)

        def mutate(value: dict) -> None:
            dependencies = value["recovery_manifest"]["dependencies"]
            inventory = next(
                row for row in dependencies
                if row["dependency_id"] == "dependency-32"
            )
            receipt = next(
                row for row in dependencies
                if row["dependency_id"] == "dependency-33"
            )
            receipt["name"] = inventory["name"]

        with self.assertRaisesRegex(
            ValueError, "Direct media slot dependency changed"
        ):
            audit.build_report(self.mutated_json_blob(key, mutate))

    def test_coherent_readiness_repin_cannot_invent_crop_contract(self) -> None:
        key = (audit.READINESS_COMMIT, audit.READINESS_PATH)

        def mutate(value: dict) -> None:
            value["camera_and_media_gate"]["crop_contract_published"] = True

        with self.assertRaisesRegex(ValueError, "closed gates"):
            audit.build_report(self.mutated_json_blob(key, mutate))

    def test_coherent_receipt_repin_cannot_change_model(self) -> None:
        key = (audit.RESULTS_COMMIT, audit.D1_TIMING_JOB_RECEIPT_PATH)

        def mutate(value: dict) -> None:
            value["model_id"] = "N3"

        with self.assertRaisesRegex(ValueError, "Timing job gate"):
            audit.build_report(
                self.mutated_json_blob(key, mutate, resign=True)
            )

    def test_coherent_source_repin_cannot_disable_release_guard(self) -> None:
        key = (audit.MEDIA_COMMIT, audit.MEDIA_JOBS_PATH)
        blobs = dict(self.blobs)
        original = blobs[key]
        changed = original.replace(b'"released": True', b'"released": False')
        self.assertNotEqual(changed, original)
        blobs[key] = changed
        with self.assertRaisesRegex(ValueError, "release behavior"):
            audit.build_report(blobs)

    def test_audit_has_no_raw_filesystem_api(self) -> None:
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        imported_roots = {
            alias.name.split(".", 1)[0]
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertNotIn("os", imported_roots)
        self.assertNotIn("shutil", imported_roots)
        git_blob = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "git_blob"
        )
        strings = {
            child.value for child in ast.walk(git_blob)
            if isinstance(child, ast.Constant) and isinstance(child.value, str)
        }
        self.assertIn("--no-replace-objects", strings)
        self.assertIn("show", strings)


if __name__ == "__main__":
    unittest.main()
