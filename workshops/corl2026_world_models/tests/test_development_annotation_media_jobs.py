from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


WORKSHOP = Path(__file__).resolve().parents[1]
MODULE = (
    WORKSHOP
    / "experiments/forecast_layout/development_annotation_media_jobs.py"
)
SPEC = importlib.util.spec_from_file_location(
    "development_annotation_media_jobs", MODULE
)
jobs = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(jobs)


class DevelopmentAnnotationMediaQueueTests(unittest.TestCase):
    def _descriptor(self, name: str, ordinal: int) -> dict:
        return {
            "path": str(jobs.RAW_ROOT / "test-inputs" / f"{name}.json"),
            "sha256": f"{ordinal:064x}",
            "bytes": 1000 + ordinal,
        }

    def _inputs(self) -> dict:
        ordinal = 1
        compiler = {}
        for name in ("job_receipt", "compiler_receipt"):
            compiler[name] = self._descriptor("compiler-" + name, ordinal)
            ordinal += 1
        models = []
        for model in jobs.MODELS:
            row = {"model_id": model}
            for field in (
                "request_provenance", "timing_sidecar", "timing_job_receipt",
                "camera_crop_contract", "alignment_contract",
                "physical_alignment_receipt",
            ):
                row[field] = self._descriptor(f"{model}-{field}", ordinal)
                ordinal += 1
            models.append(row)
        return {
            "schema_version": jobs.INPUT_SCHEMA,
            "study_id": jobs.STUDY_ID,
            "mode": jobs.MODE,
            "raw_root": str(jobs.RAW_ROOT),
            "compiler": compiler,
            "models": models,
        }

    def _implementation(self) -> dict[str, dict]:
        return {
            name: {
                "path": str(path),
                "sha256": f"{index + 100:064x}",
                "bytes": 10000 + index,
            }
            for index, (name, path) in enumerate(
                sorted(jobs._source_paths(jobs.REPOSITORY_ROOT).items())
            )
        }

    def _cluster_implementation(self, study_commit: str) -> dict[str, dict]:
        source_root = jobs.CONTROL_ROOT / "sources" / study_commit
        return {
            name: {
                "path": str(source_root / path.relative_to(jobs.REPOSITORY_ROOT)),
                "sha256": f"{index + 7000:064x}",
                "bytes": 20000 + index,
            }
            for index, (name, path) in enumerate(
                sorted(jobs._source_paths(jobs.REPOSITORY_ROOT).items())
            )
        }

    def _terminal_preparation_receipt(
        self, study_commit: str, index_descriptor: dict,
    ) -> tuple[dict, dict, dict]:
        job_dir = jobs.CONTROL_ROOT / "jobs" / jobs.JOB_ID
        output_root = job_dir / jobs.OUTPUT_RELATIVE
        receipt_descriptor = {
            "path": str(job_dir / "publish" / jobs.PUBLISH_RECEIPT_NAME),
            "sha256": "d" * 64,
            "bytes": 24000,
        }
        implementation = self._cluster_implementation(study_commit)
        selected_zero = 4
        rendered = jobs.EXPECTED_SELECTED * 6 - selected_zero
        receipt = jobs.queue.signed_document({
            "schema_version": jobs.JOB_RECEIPT_SCHEMA,
            "namespace": jobs.NAMESPACE,
            "study_id": jobs.STUDY_ID,
            "mode": jobs.MODE,
            "status": "passed",
            "decision": "go_for_human_pixel_blindness_review_only",
            "job_id": jobs.JOB_ID,
            "job_dir": str(job_dir),
            "study_commit": study_commit,
            "queue_role": jobs.WORKER_ROLE,
            "worker_id": jobs.WORKER_ROLE,
            "runtime_identity": {
                "hostname": jobs.WORKER_ROLE + "-fixture",
                "pod_uid": "pod-fixture",
                "pid": 123,
            },
            "queue_descriptor": {
                "path": str(job_dir / "descriptor.json"),
                "sha256": "1" * 64,
                "bytes": 1000,
            },
            "queue_claim": {
                "path": str(job_dir / "claim" / "owner.json"),
                "sha256": "2" * 64,
                "bytes": 1100,
            },
            "implementation": implementation,
            "inputs": jobs.validate_input_shape(self._inputs()),
            "outputs": {
                "input_manifest": {
                    "path": str(job_dir / jobs.MANIFEST_RELATIVE),
                    "sha256": "3" * 64,
                    "bytes": 1200,
                },
                "preparation_receipt": {
                    "path": str(output_root / "preparation_receipt.json"),
                    "sha256": "4" * 64,
                    "bytes": 1300,
                },
                "output_root": str(output_root),
                "output_file_count": 12 + 3 * rendered,
                "output_files_sha256": "5" * 64,
                "publish_tranche_index": index_descriptor,
            },
            "counts": {
                "episodes": 32,
                "requests": 1152,
                "requests_by_model": {"N3": 240, "D1": 912},
                "source_timing_capable_requests": 480,
                "d1_forecast_timing_unavailable_requests": 672,
                "timing_camera_action_eligible_requests": 448,
                "selected_requests": 128,
                "selected_requests_by_model": {"N3": 64, "D1": 64},
                "selected_request_zero_count": selected_zero,
                "source_extraction_records": rendered,
                "rendered_pngs": rendered,
                "unique_pixel_review_assets": rendered,
                "unique_sanitized_png_assets": rendered - 10,
                "publish_tranches": 3,
                "total_unique_sanitized_png_bytes": 1234567,
                "camera_replay_exact_runtime_sessions": 2,
                "original_camera_frames_replayed_exact_runtime": rendered - 256,
                "human_pixel_blindness_reviews": 0,
                "human_labels": 0,
            },
            "science_counts": jobs._zero_science_counts(),
            "human_pixel_blindness_review_complete": False,
            "rater_packets_created": False,
            "safe_for_rater_distribution": False,
            "safe_to_release_confirmation": False,
            "raw_outputs_recoverable_on_gm_pvc": True,
            "published_files": [jobs.PUBLISH_RECEIPT_NAME, jobs.PUBLISH_INDEX_NAME],
            "claim_boundary": jobs.PREPARATION_CLAIM_BOUNDARY,
            "completed_at_utc": "2026-09-13T12:00:00Z",
        })
        return receipt, receipt_descriptor, implementation

    def _tranche_index(self) -> dict:
        tranches = []
        asset_ordinal = 1
        for ordinal in range(1, 4):
            assets = []
            for _ in range(2):
                digest = f"{asset_ordinal + 1000:064x}"
                assets.append({
                    "asset_sha256": digest,
                    "source_path": f"annotation_media/source_{digest}.png",
                    "bytes": 2000 + asset_ordinal,
                    "width_px": 320,
                    "height_px": 168 if ordinal != 3 else 176,
                    "aliases": [{
                        "source_image_id": f"source_{digest}",
                        "source_request_id": f"request_{digest[:32]}",
                        "image_role": "current",
                        "join_key": {
                            "model_id": "N3" if ordinal != 3 else "D1",
                            "cell_id": f"cell-{ordinal}",
                            "request_index": asset_ordinal,
                        },
                        "camera_id": "over_shoulder_left_camera",
                        "camera_crop_id": (
                            "n3-over-shoulder-left-168x320-v1"
                            if ordinal != 3
                            else "d1-over-shoulder-left-176x320-v1"
                        ),
                        "camera_crop_sha256": f"{asset_ordinal + 2000:064x}",
                        "alignment_receipt_id": f"alignment-{ordinal}",
                        "alignment_receipt_sha256": f"{asset_ordinal + 3000:064x}",
                        "render_receipt_id": f"render-{asset_ordinal}",
                        "render_receipt_sha256": f"{asset_ordinal + 4000:064x}",
                        "source_lineage_record_sha256": f"{asset_ordinal + 5000:064x}",
                    }],
                })
                asset_ordinal += 1
            tranches.append({
                "tranche_id": f"development-annotation-media-tranche-{ordinal:03d}",
                "ordinal": ordinal,
                "asset_count": len(assets),
                "alias_count": len(assets),
                "asset_bytes": sum(row["bytes"] for row in assets),
                "asset_budget_bytes": jobs.PUBLISH_TRANCHE_ASSET_BUDGET_BYTES,
                "publisher_file_limit_bytes": jobs.PUBLISH_FILE_LIMIT_BYTES,
                "publisher_job_limit_bytes": jobs.PUBLISH_JOB_LIMIT_BYTES,
                "assets": assets,
            })
        return jobs.queue.signed_document({
            "schema_version": jobs.TRANCHE_INDEX_SCHEMA,
            "study_id": jobs.STUDY_ID,
            "stage": "development",
            "status": "prepared_for_bounded_result_return_pending_human_review",
            "visibility": "RESTRICTED ANALYST MANIFEST; NEVER DISTRIBUTE TO RATERS",
            "unique_asset_count": 6,
            "render_alias_count": 6,
            "tranche_count": 3,
            "total_unique_asset_bytes": sum(
                row["asset_bytes"] for row in tranches
            ),
            "source_extraction_lineage": {
                "path": "source_extraction_lineage.json",
                "sha256": "a" * 64,
                "bytes": 20000,
            },
            "rendered_png_manifest": {
                "path": "rendered_png_manifest.json",
                "sha256": "b" * 64,
                "bytes": 10000,
            },
            "deduplication_key": "annotation_media_sha256",
            "tranches_non_overlapping": True,
            "each_asset_published_once": True,
            "human_pixel_blindness_review_complete": False,
            "safe_for_rater_distribution": False,
            "safe_to_release_confirmation": False,
            "tranches": tranches,
        })

    def test_source_has_no_duplicate_literal_dictionary_keys(self) -> None:
        tree = ast.parse(MODULE.read_text(encoding="utf-8"))
        duplicates = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = [
                child.value for child in node.keys
                if isinstance(child, ast.Constant) and isinstance(child.value, str)
            ]
            duplicates.extend((node.lineno, key) for key in set(keys) if keys.count(key) > 1)
        self.assertEqual(duplicates, [])

    def test_contract_is_exact_zero_science_and_bounded(self) -> None:
        path = jobs.REPOSITORY_ROOT / jobs.CONTRACT_RELATIVE
        observed = jobs.queue.load_json(path, "contract")
        self.assertEqual(observed, jobs._expected_contract())
        self.assertEqual(observed["science_counts"], jobs._zero_science_counts())
        self.assertEqual(
            observed["output_policy"]["publisher_file_limit_bytes"],
            16 * 1024 * 1024,
        )
        self.assertEqual(
            observed["output_policy"]["publish_tranche_asset_budget_bytes"],
            48 * 1024 * 1024,
        )
        self.assertIs(observed["safe_for_rater_distribution"], False)
        self.assertIs(observed["safe_to_release_confirmation"], False)

    def test_input_shape_requires_two_models_and_unique_pvc_paths(self) -> None:
        normalized = jobs.validate_input_shape(self._inputs())
        self.assertEqual([row["model_id"] for row in normalized["models"]], ["D1", "N3"])
        missing = self._inputs()
        missing["models"].pop()
        with self.assertRaisesRegex(jobs.AnnotationMediaQueueError, "exactly two"):
            jobs.validate_input_shape(missing)
        duplicated = self._inputs()
        duplicated["models"][1]["timing_sidecar"] = copy.deepcopy(
            duplicated["models"][0]["timing_sidecar"]
        )
        with self.assertRaisesRegex(jobs.AnnotationMediaQueueError, "ambiguous"):
            jobs.validate_input_shape(duplicated)
        escaped = self._inputs()
        escaped["compiler"]["job_receipt"]["path"] = "/tmp/outside.json"
        with self.assertRaisesRegex(jobs.AnnotationMediaQueueError, "outside"):
            jobs.validate_input_shape(escaped)
        dotdot = self._inputs()
        dotdot["compiler"]["job_receipt"]["path"] = str(
            jobs.RAW_ROOT / ".." / "outside.json"
        )
        with self.assertRaisesRegex(
            jobs.AnnotationMediaQueueError, "lexically normalized"
        ):
            jobs.validate_input_shape(dotdot)

    def test_formal_wave_is_deterministic_hash_bound_and_not_a_rater_release(self) -> None:
        implementation = self._implementation()
        with mock.patch.object(jobs, "_local_implementation", return_value=implementation):
            first = jobs.build_formal_wave(study_commit="a" * 40, inputs=self._inputs())
            second = jobs.build_formal_wave(study_commit="a" * 40, inputs=self._inputs())
        self.assertEqual(first, second)
        self.assertEqual(first["science_counts"], jobs._zero_science_counts())
        self.assertIs(first["human_pixel_blindness_review_complete"], False)
        self.assertIs(first["rater_packets_created"], False)
        self.assertIs(first["safe_for_rater_distribution"], False)
        descriptor = first["jobs"][0]
        self.assertEqual(descriptor["role"], jobs.WORKER_ROLE)
        self.assertIn("formal-full", descriptor["argv"])
        encoded = descriptor["argv"][descriptor["argv"].index("--inputs-json") + 1]
        self.assertEqual(json.loads(encoded), first["inputs"])
        command = " ".join(descriptor["argv"])
        for identity in implementation.values():
            self.assertIn(identity["sha256"], command)

    def test_tranche_index_rejects_overlap_and_per_file_overflow(self) -> None:
        index = self._tranche_index()
        self.assertEqual(jobs._validate_tranche_index(index), index)
        overlap = copy.deepcopy(index)
        overlap.pop("payload_sha256")
        overlap["tranches"][1]["assets"][0]["asset_sha256"] = (
            overlap["tranches"][0]["assets"][0]["asset_sha256"]
        )
        overlap = jobs.queue.signed_document(overlap)
        with self.assertRaisesRegex(jobs.AnnotationMediaQueueError, "multiple tranches"):
            jobs._validate_tranche_index(overlap)
        overflow = copy.deepcopy(index)
        overflow.pop("payload_sha256")
        overflow["tranches"][0]["assets"][0]["bytes"] = jobs.PUBLISH_FILE_LIMIT_BYTES
        overflow = jobs.queue.signed_document(overflow)
        with self.assertRaisesRegex(jobs.AnnotationMediaQueueError, "per-file"):
            jobs._validate_tranche_index(overflow)
        public = copy.deepcopy(index)
        public.pop("payload_sha256")
        public["visibility"] = "PUBLIC RATER PACKET"
        public = jobs.queue.signed_document(public)
        with self.assertRaisesRegex(jobs.AnnotationMediaQueueError, "state changed"):
            jobs._validate_tranche_index(public)

    def test_tranche_wave_assigns_each_nonoverlapping_tranche_once(self) -> None:
        implementation = self._implementation()
        receipt = self._descriptor("preparation-job", 6000)
        index_descriptor = self._descriptor("tranche-index", 6001)
        index = self._tranche_index()
        with mock.patch.object(jobs, "_local_implementation", return_value=implementation):
            wave = jobs.build_tranche_wave(
                study_commit="b" * 40,
                preparation_job_receipt=receipt,
                tranche_index_descriptor=index_descriptor,
                tranche_index=index,
            )
        self.assertEqual(len(wave["jobs"]), 3)
        self.assertEqual(
            [row["job_id"] for row in wave["jobs"]],
            [row["tranche_id"] for row in index["tranches"]],
        )
        self.assertEqual(
            [row["role"] for row in wave["jobs"]],
            list(jobs.TRANCHE_WORKER_ROLES[:3]),
        )
        self.assertTrue(wave["each_asset_published_once"])
        self.assertFalse(wave["safe_for_rater_distribution"])

    def test_terminal_preparation_receipt_binds_exact_tranche_index(self) -> None:
        commit = "b" * 40
        job_dir = jobs.CONTROL_ROOT / "jobs" / jobs.JOB_ID
        index_descriptor = {
            "path": str(job_dir / jobs.OUTPUT_RELATIVE / jobs.PUBLISH_INDEX_NAME),
            "sha256": "c" * 64,
            "bytes": 19000,
        }
        receipt, receipt_descriptor, implementation = (
            self._terminal_preparation_receipt(commit, index_descriptor)
        )
        self.assertEqual(
            jobs._validate_terminal_preparation_job(
                receipt,
                receipt_descriptor=receipt_descriptor,
                index_descriptor=index_descriptor,
                study_commit=commit,
                expected_implementation=implementation,
            ),
            receipt,
        )
        mismatched = copy.deepcopy(index_descriptor)
        mismatched["sha256"] = "e" * 64
        with self.assertRaisesRegex(
            jobs.AnnotationMediaQueueError, "exact preparation output descriptor"
        ):
            jobs._validate_terminal_preparation_job(
                receipt,
                receipt_descriptor=receipt_descriptor,
                index_descriptor=mismatched,
                study_commit=commit,
                expected_implementation=implementation,
            )

    def test_terminal_preparation_receipt_rejects_science_or_authority_drift(self) -> None:
        commit = "b" * 40
        job_dir = jobs.CONTROL_ROOT / "jobs" / jobs.JOB_ID
        index_descriptor = {
            "path": str(job_dir / jobs.OUTPUT_RELATIVE / jobs.PUBLISH_INDEX_NAME),
            "sha256": "c" * 64,
            "bytes": 19000,
        }
        receipt, receipt_descriptor, implementation = (
            self._terminal_preparation_receipt(commit, index_descriptor)
        )
        for mutation in (
            {"science_counts": {}},
            {"science_counts": {
                **jobs._zero_science_counts(), "labels_created_by_job": 1,
            }},
            {"rater_packets_created": True},
            {"safe_to_release_confirmation": True},
            {"mode": "confirmation"},
        ):
            changed = copy.deepcopy(receipt)
            changed.pop("payload_sha256")
            changed.update(mutation)
            changed = jobs.queue.signed_document(changed)
            with self.subTest(mutation=mutation):
                with self.assertRaisesRegex(
                    jobs.AnnotationMediaQueueError, "exact restricted gate"
                ):
                    jobs._validate_terminal_preparation_job(
                        changed,
                        receipt_descriptor=receipt_descriptor,
                        index_descriptor=index_descriptor,
                        study_commit=commit,
                        expected_implementation=implementation,
                    )

    def test_runtime_formal_descriptor_reconstructs_emitter(self) -> None:
        implementation = self._implementation()
        expected = jobs._job_descriptor(
            study_commit="c" * 40,
            implementation=implementation,
            inputs=self._inputs(),
        )
        args = jobs._parser().parse_args(expected["argv"][2:])
        observed, observed_implementation, observed_inputs = jobs._runtime_descriptor(args)
        self.assertEqual(observed, expected)
        self.assertEqual(observed_implementation.keys(), implementation.keys())
        self.assertEqual(observed_inputs, jobs.validate_input_shape(self._inputs()))

    def test_main_publish_receipt_failure_cannot_expose_partial_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job_dir = Path(temporary) / "job"
            raw = job_dir / "raw"
            publish = job_dir / "publish"
            raw.mkdir(parents=True)
            publish.mkdir()
            index = raw / "publish_tranche_index.json"
            index.write_text("{}\n", encoding="utf-8")
            failure = RuntimeError("injected failure after index copy")
            with mock.patch.object(
                jobs.queue, "immutable_json", side_effect=failure
            ):
                with self.assertRaisesRegex(RuntimeError, "after index copy"):
                    jobs._publish_main_transactionally(
                        raw=raw,
                        publish=publish,
                        index_source=index,
                        receipt={"status": "passed"},
                    )
            self.assertEqual(list(publish.iterdir()), [])
            self.assertTrue((raw / "main_publish_staging" / jobs.PUBLISH_INDEX_NAME).is_file())
            jobs._write_failure_receipt(None, job_dir, failure)
            self.assertEqual(
                {path.name for path in publish.iterdir()},
                {jobs.PUBLISH_FAILURE_NAME},
            )


if __name__ == "__main__":
    unittest.main()
