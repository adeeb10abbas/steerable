from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


WORKSHOP = Path(__file__).resolve().parents[1]
MODULE = (
    WORKSHOP
    / "experiments/forecast_layout/development_physical_alignment_jobs.py"
)
SPEC = importlib.util.spec_from_file_location(
    "development_physical_alignment_jobs", MODULE
)
jobs = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(jobs)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def descriptor(path: Path, label: str, size: int = 100) -> dict:
    return {"path": str(path), "sha256": digest(label), "bytes": size}


class DevelopmentPhysicalAlignmentJobTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _compiler_receipt(self) -> dict:
        job_dir = jobs.CONTROL_ROOT / "jobs" / jobs.COMPILER_JOB_ID
        primary = {}
        for model in jobs.MODELS:
            for suffix in (
                "timing_request_inventory", "request_provenance", "freeze_cells"
            ):
                name = f"{model.lower()}_development_{suffix}.json"
                primary[name] = descriptor(
                    job_dir / "raw/compiler_bundle" / name, name
                )
        return jobs.queue.signed_document({
            "schema_version": jobs.COMPILER_JOB_SCHEMA,
            "namespace": jobs.NAMESPACE,
            "study_id": jobs.STUDY_ID,
            "mode": "formal_full",
            "status": "passed",
            "decision": "go",
            "job_id": jobs.COMPILER_JOB_ID,
            "job_dir": str(job_dir),
            "study_commit": jobs.COMPILER_SOURCE_COMMIT,
            "formal_cohort_complete": True,
            "safe_for_timing_binding": True,
            "safe_to_release_confirmation": False,
            "confirmation_released": False,
            "behavioral_policy_skill_evaluated": False,
            "raw_outputs_recoverable_on_gm_pvc": True,
            "counts": {
                "compiled_cells": 32,
                "compiled_source_behavioral_requests": 1152,
                "compiled_source_behavioral_actions": 14400,
            },
            "science_counts": jobs._zero_science_counts(),
            "compiler_science_activity": jobs._compiler_zero_science_activity(),
            "published_files": [
                "development_evidence_compiler_job_receipt.json"
            ],
            "outputs": {
                "compiler_receipt": {
                    "path": str(job_dir / "raw/compiler_bundle/compiler_receipt.json"),
                    "sha256": jobs.COMPILER_RECEIPT_SHA256,
                    "bytes": jobs.COMPILER_RECEIPT_BYTES,
                },
                "compiler_output_inventory": {
                    "path": str(job_dir / "raw/compiler_output_inventory.json"),
                    "sha256": jobs.COMPILER_INVENTORY_SHA256,
                    "bytes": jobs.COMPILER_INVENTORY_BYTES,
                },
                "primary_compiled_outputs": primary,
                "semantically_validated_nonreceipt_file_count": 70,
            },
        })

    def _timing_receipt(self, model: str) -> dict:
        job_id = jobs.TIMING_JOB_IDS[model]
        value = {
            "schema_version": jobs.TIMING_JOB_SCHEMA,
            "namespace": jobs.NAMESPACE,
            "study_id": jobs.STUDY_ID,
            "model_id": model,
            "status": "passed",
            "decision": "go",
            "job_id": job_id,
            "job_dir": str(jobs.CONTROL_ROOT / "jobs" / job_id),
            "referenced_behavioral_cells": 16,
            "referenced_behavioral_model_requests": jobs.REQUEST_COUNTS[model],
            "science_counts": jobs._timing_zero_science_counts(),
            "physical_time_qualified": True,
            "development_timing_sidecar_valid": True,
            "behavioral_policy_skill_evaluated": False,
            "safe_to_release_confirmation": False,
            "raw_outputs_recoverable_on_gm_pvc": True,
            "source_request_receipt_sha256s": [
                digest(f"{model}-{index}")
                for index in range(jobs.REQUEST_COUNTS[model])
            ],
            "outputs": {
                "raw_development_timing_sidecar": {
                    "path": str(
                        jobs.CONTROL_ROOT / "jobs" / job_id / "raw"
                        / f"{model.lower()}_development_timing_sidecar.json"
                    ),
                    **jobs.TIMING_SIDECARS[model],
                }
            },
        }
        if model == "D1":
            value.update({
                "request_timing_coverage": jobs.EXPECTED_D1_COVERAGE,
                "physical_time_coverage_complete": False,
                "physical_time_qualified_scope": (
                    "source-proven full conditioning-origin decodes only"
                ),
            })
        return jobs.queue.signed_document(value)

    def _camera_receipt(self, attempt: int = 3) -> dict:
        job_id = f"camera-crop-replay-witness-{attempt:03d}"
        publish = jobs.CONTROL_ROOT / "jobs" / job_id / "publish"
        outputs = {}
        for model in jobs.MODELS:
            name = f"{model.lower()}_camera_crop_contract.json"
            outputs[model] = {
                **descriptor(publish / name, f"camera-{model}"),
                "schema_version": jobs.CAMERA_CONTRACT_SCHEMA,
                "model_id": model,
                "camera_crop_id": jobs.EXPECTED_CAMERA_CROP_IDS[model],
                "payload_sha256": digest(f"payload-{model}"),
            }
        return jobs.queue.signed_document({
            "schema_version": jobs.CAMERA_JOB_SCHEMA,
            "namespace": jobs.NAMESPACE,
            "study_id": jobs.STUDY_ID,
            "status": "passed",
            "decision": "qualified_from_original_camera_pixels",
            "job_id": job_id,
            "job_dir": str(jobs.CONTROL_ROOT / "jobs" / job_id),
            "science_counts": jobs._zero_science_counts(),
            "simulator_state_render_used": False,
            "whole_frame_identity": False,
            "safe_for_physical_alignment_input": True,
            "safe_to_release_confirmation": False,
            "confirmation_released": False,
            "behavioral_policy_skill_evaluated": False,
            "raw_outputs_recoverable_on_gm_pvc": True,
            "outputs": {jobs.CAMERA_OUTPUT_KEY: outputs},
        })

    def _normalized_input(self) -> dict:
        compiler = jobs._validate_compiler_job(
            self._compiler_receipt(),
            file_identity={
                "sha256": jobs.COMPILER_JOB_FILE_SHA256,
                "bytes": jobs.COMPILER_JOB_FILE_BYTES,
            },
        )
        timing = [
            jobs._validate_timing_job(
                model,
                self._timing_receipt(model),
                file_identity=jobs.TIMING_JOB_FILES[model],
            )
            for model in jobs.MODELS
        ]
        camera_receipt = self._camera_receipt()
        camera_job, camera_outputs = jobs._validate_camera_job(
            camera_receipt,
            file_identity={"sha256": digest("camera-job"), "bytes": 1234},
        )
        return jobs.validate_input_shape({
            "schema_version": jobs.INPUT_SCHEMA,
            "study_id": jobs.STUDY_ID,
            "mode": jobs.MODE,
            "results_commit": "1" * 40,
            "compiler": compiler,
            "timing": timing,
            "camera": {
                "job_receipt": camera_job,
                "crop_contracts": [
                    {
                        "model_id": model,
                        "contract": {
                            key: camera_outputs[model][key]
                            for key in ("path", "sha256", "bytes")
                        },
                        "contract_payload_sha256": digest(f"payload-{model}"),
                        "camera_terminal_output": camera_outputs[model],
                    }
                    for model in jobs.MODELS
                ],
            },
        })

    def test_contract_is_exact_worker00_zero_authority(self) -> None:
        contract = jobs.queue.load_json(
            WORKSHOP
            / "experiments/forecast_layout/development_physical_alignment_contract.json",
            "test contract",
        )
        self.assertEqual(contract, jobs._expected_contract())
        self.assertEqual(contract["job"]["role"], "wmf-forecast-0912-worker-00")
        self.assertEqual(contract["science_counts"], jobs._zero_science_counts())
        self.assertFalse(contract["release"]["confirmation_release_argument"])
        self.assertFalse(contract["confirmation_authority"])

    def test_compiler004_gate_rejects_missing_or_extra_science_keys(self) -> None:
        receipt = self._compiler_receipt()
        jobs._validate_compiler_job(
            receipt,
            file_identity={
                "sha256": jobs.COMPILER_JOB_FILE_SHA256,
                "bytes": jobs.COMPILER_JOB_FILE_BYTES,
            },
        )
        for science in ({}, {**jobs._zero_science_counts(), "extra": 0}):
            changed = copy.deepcopy(receipt)
            changed["science_counts"] = science
            with self.assertRaisesRegex(
                jobs.DevelopmentAlignmentJobError, "science_counts"
            ):
                jobs._validate_compiler_job(
                    changed,
                    file_identity={
                        "sha256": jobs.COMPILER_JOB_FILE_SHA256,
                        "bytes": jobs.COMPILER_JOB_FILE_BYTES,
                    },
                )

    def test_timing_gates_pin_receipts_sidecars_and_d1_missingness(self) -> None:
        for model in jobs.MODELS:
            jobs._validate_timing_job(
                model,
                self._timing_receipt(model),
                file_identity=jobs.TIMING_JOB_FILES[model],
            )
        changed = self._timing_receipt("D1")
        changed["request_timing_coverage"] = {
            **jobs.EXPECTED_D1_COVERAGE,
            "source_unmapped_incremental_decode_request_count": 671,
        }
        with self.assertRaisesRegex(
            jobs.DevelopmentAlignmentJobError, "missingness boundary"
        ):
            jobs._validate_timing_job(
                "D1", changed, file_identity=jobs.TIMING_JOB_FILES["D1"]
            )

    def test_camera_gate_rejects_diagnostic_attempts_and_science_drift(self) -> None:
        valid = self._camera_receipt(3)
        jobs._validate_camera_job(
            valid, file_identity={"sha256": digest("camera-job"), "bytes": 1}
        )
        for attempt in (1, 2):
            with self.assertRaisesRegex(
                jobs.DevelopmentAlignmentJobError, "cannot feed alignment"
            ):
                jobs._validate_camera_job(
                    self._camera_receipt(attempt),
                    file_identity={"sha256": digest("bad"), "bytes": 1},
                )
        changed = self._camera_receipt(3)
        changed.pop("payload_sha256")
        changed["science_counts"] = {}
        changed = jobs.queue.signed_document(changed)
        with self.assertRaisesRegex(
            jobs.DevelopmentAlignmentJobError, "science_counts"
        ):
            jobs._validate_camera_job(
                changed,
                file_identity={"sha256": digest("bad"), "bytes": 1},
            )
        diagnostic = self._camera_receipt(3)
        diagnostic.pop("payload_sha256")
        diagnostic["diagnostic_only"] = True
        diagnostic = jobs.queue.signed_document(diagnostic)
        with self.assertRaisesRegex(
            jobs.DevelopmentAlignmentJobError, "receipt fields changed"
        ):
            jobs._validate_camera_job(
                diagnostic,
                file_identity={"sha256": digest("bad"), "bytes": 1},
            )

    def test_camera_terminal_declaration_binds_crop_id_and_payload(self) -> None:
        receipt = self._camera_receipt(3)
        _, declarations = jobs._validate_camera_job(
            receipt, file_identity={"sha256": digest("camera-job"), "bytes": 1}
        )
        for model in jobs.MODELS:
            declaration = declarations[model]
            value = {
                "schema_version": jobs.CAMERA_CONTRACT_SCHEMA,
                "model_id": model,
                "camera_crop_id": jobs.EXPECTED_CAMERA_CROP_IDS[model],
                "payload_sha256": declaration["payload_sha256"],
            }
            local = {
                "path": str(self.root / f"{model}.json"),
                "sha256": declaration["sha256"],
                "bytes": declaration["bytes"],
            }
            jobs._bind_crop_file_to_camera_declaration(
                model=model, value=value, file_identity=local,
                declaration=declaration,
            )
            changed = dict(value)
            changed["payload_sha256"] = digest("invented-" + model)
            with self.assertRaisesRegex(
                jobs.DevelopmentAlignmentJobError, "terminal declaration"
            ):
                jobs._bind_crop_file_to_camera_declaration(
                    model=model, value=changed, file_identity=local,
                    declaration=declaration,
                )

    def test_crop_gate_accepts_only_exact_validator_success(self) -> None:
        path = self.root / "crop.json"
        value = jobs.queue.signed_document({"model_id": "N3"})
        jobs.queue.immutable_json(path, value)
        calls = []

        class Camera:
            @staticmethod
            def validate_camera_crop_contract(candidate, expected_model=None):
                calls.append((candidate, expected_model))

        returned, identity = jobs._validate_crop_file(
            model="N3",
            path=path,
            digest=jobs.queue.sha256_file(path),
            camera_replay=Camera,
        )
        self.assertEqual(returned, value)
        self.assertEqual(identity, jobs.queue.file_identity(path))
        self.assertEqual(calls, [(value, "N3")])

    def test_input_shape_rejects_duplicate_paths(self) -> None:
        value = self._normalized_input()
        changed = copy.deepcopy(value)
        changed["timing"][1]["sidecar"] = copy.deepcopy(
            changed["timing"][0]["sidecar"]
        )
        with self.assertRaisesRegex(
            jobs.DevelopmentAlignmentJobError, "paths are ambiguous"
        ):
            jobs.validate_input_shape(changed)

    def test_descriptor_shape_rejects_lexical_parent_escape(self) -> None:
        changed = {
            "path": str(jobs.RAW_ROOT / "control/jobs/../outside.json"),
            "sha256": digest("outside"),
            "bytes": 1,
        }
        with self.assertRaisesRegex(
            jobs.DevelopmentAlignmentJobError, "lexically normalized"
        ):
            jobs._descriptor_shape(changed, "adversarial descriptor")

    def test_wave_is_deterministic_and_only_releases_one_cpu_job(self) -> None:
        inputs = self._normalized_input()
        implementation = {
            name: {"sha256": digest(name)}
            for name in jobs._source_paths(jobs.REPOSITORY_ROOT)
        }
        first = jobs._job_descriptor(
            study_commit="2" * 40,
            implementation=implementation,
            inputs=inputs,
        )
        second = jobs._job_descriptor(
            study_commit="2" * 40,
            implementation=implementation,
            inputs=inputs,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["job_id"], jobs.JOB_ID)
        self.assertEqual(first["role"], "wmf-forecast-0912-worker-00")
        self.assertEqual(first["argv"][0], str(jobs.ROBOLAB_PYTHON))

    def test_release_evidence_is_exact_full_two_model_and_unlabelled(self) -> None:
        spec = self.root / jobs.ABLATION_SPEC_RELATIVE
        planned = self.root / jobs.PLANNED_CELLS_RELATIVE
        spec.parent.mkdir(parents=True)
        spec.write_text("{}\n", encoding="utf-8")
        planned.write_text("cell_id\n", encoding="utf-8")
        fragments = {
            model: {"development_cells": [{"cell_receipt": {}}] * 16}
            for model in jobs.MODELS
        }
        evidence = jobs._build_release_evidence(
            source_root=self.root,
            fragments=fragments,
            inputs=self._normalized_input(),
        )
        self.assertEqual(evidence["schema_version"], "wmf-development-release-evidence-v1")
        self.assertEqual(evidence["cohort_branch"], "full_two_model")
        self.assertEqual(evidence["qualified_model_ids"], ["N3", "D1"])
        self.assertIsNone(evidence["annotation"])
        self.assertIsNone(evidence["resource_budget_policy"])
        self.assertEqual(len(evidence["model_evidence"]), 2)

    def test_freeze_call_is_literal_alignment_only(self) -> None:
        calls = []

        class FakeFreeze:
            @staticmethod
            def write_bundle(evidence_path, bundle_path, *, confirmation_release):
                calls.append((evidence_path, bundle_path, confirmation_release))
                return {"status": "alignment_qualified"}

        jobs._write_alignment_bundle(
            FakeFreeze, self.root / "evidence.json", self.root / "bundle"
        )
        self.assertEqual(calls[0][2], False)

    def test_alignment_bundle_gate_rejects_any_confirmation_artifact(self) -> None:
        bundle = self.root / "alignment_bundle"
        bundle.mkdir()
        inputs = self._normalized_input()
        timing_inputs = {row["model_id"]: row for row in inputs["timing"]}
        crop_inputs = {
            row["model_id"]: row
            for row in inputs["camera"]["crop_contracts"]
        }
        expected_mappings = {}
        expected_alignment_unsigned = {}

        class FakeFreeze:
            MAPPING_SCHEMA = "wmf-forecast-physical-alignment-receipt-v1"
            D1_GENERATED_TARGETS_SCOPE = (
                "request_timing_bindings.decoded_output_timing."
                "source_timing_mapping_applies == true"
            )
            MODEL_LIMITS = {
                "N3": {
                    "returned_action_horizon": 32,
                    "unchanged_executed_prefix_horizon": 32,
                    "action_space": "joint_pos",
                    "seed_semantics": (
                        "matched effective policy seed per layout block"
                    ),
                    "temporal_context": (
                        "isolated full temporal/cache reset before every episode"
                    ),
                },
                "D1": {
                    "returned_action_horizon": 24,
                    "unchanged_executed_prefix_horizon": 8,
                    "action_space": "joint_pos",
                    "seed_semantics": (
                        "fixed effective model noise 1140; requests are not "
                        "independent draws"
                    ),
                    "temporal_context": (
                        "isolated official two-rank full temporal/cache reset "
                        "before every episode"
                    ),
                },
            }
            canonical_bytes = staticmethod(jobs.queue.compact_bytes)
            pretty_json_bytes = staticmethod(jobs.queue.canonical_bytes)
            sha256_bytes = staticmethod(jobs.queue.sha256_bytes)
            verify_signed = staticmethod(jobs.queue.verify_signed_document)

            @staticmethod
            def derive_bundle(evidence_path, *, require_annotation):
                if require_annotation:
                    raise AssertionError("alignment validation requested annotation")
                return expected_mappings, expected_alignment_unsigned, None

        targets = {
            "N3": ((32, 32), (1, 1)),
            "D1": ((2, 6), (1, 3)),
        }
        counts = {
            "N3": ((224, 224), (240, 224)),
            "D1": ((224, 224), (224, 224)),
        }
        for model in jobs.MODELS:
            primary_id, early_id = targets[model]
            primary_counts, early_counts = counts[model]
            primary = {
                "generated_frame_index": primary_id[0],
                "target_physical_time_s": primary_id[1] / 15,
                "target_executed_action_offset": primary_id[1],
                "status": "qualified",
                "eligible_request_count": primary_counts[0],
                "full_prefix_request_count": primary_counts[1],
                "max_camera_timestamp_residual_s": 0.001,
                "max_physics_timestamp_residual_s": 0.002,
            }
            early = {
                "generated_frame_index": early_id[0],
                "target_physical_time_s": early_id[1] / 15,
                "target_executed_action_offset": early_id[1],
                "status": "qualified",
                "eligible_request_count": early_counts[0],
                "full_prefix_request_count": early_counts[1],
                "max_camera_timestamp_residual_s": 0.001,
                "max_physics_timestamp_residual_s": 0.002,
            }
            rows = []
            for index, boundary in (
                ((index, index) for index in range(1, 33))
                if model == "N3" else ((1, 3), (2, 6))
            ):
                rows.append({
                    "generated_frame_index": index,
                    "target_physical_time_s": boundary / 15,
                    "target_executed_action_offset": boundary,
                    "status": "qualified",
                    "eligible_request_count": (
                        240 if model == "N3" and index <= 2 else 224
                    ),
                    "full_prefix_request_count": 224,
                    "max_camera_timestamp_residual_s": 0.001,
                    "max_physics_timestamp_residual_s": 0.002,
                })
            boundary = {
                "generated_target_source": "request_timing_sidecar.generated_targets",
                "time_source_kind": "native_runtime_exposed_target_offsets",
                "clock_bridge": (
                    "elapsed physical seconds from request current original-camera capture"
                ),
                "presentation_video_fps_used": False,
                "conditioning_fps_used_as_target_timing": False,
                "generated_frame_index_interpreted_as_action_index": False,
            }
            if model == "D1":
                boundary.update({
                    "generated_targets_scope": FakeFreeze.D1_GENERATED_TARGETS_SCOPE,
                    "request_timing_coverage": jobs.EXPECTED_D1_COVERAGE,
                    "incremental_standalone_decodes_assigned_target_times": False,
                    "timing_unmapped_requests_eligible": False,
                })
            mapping = jobs.queue.signed_document({
                "schema_version": FakeFreeze.MAPPING_SCHEMA,
                "receipt_id": f"wmf1-development-{model.lower()}-physical-alignment-v1",
                "study_id": jobs.STUDY_ID,
                "model_id": model,
                "status": "qualified_from_complete_development_native_timing",
                "development_cell_ids": [
                    f"{model}-cell-{index:02d}" for index in range(16)
                ],
                "development_cell_count": 16,
                "development_request_count": jobs.REQUEST_COUNTS[model],
                "request_semantics": FakeFreeze.MODEL_LIMITS[model],
                "timing_claim_boundary": boundary,
                "camera": {
                    "camera_id": "over_shoulder_left_camera",
                    "camera_crop_id": jobs.EXPECTED_CAMERA_CROP_IDS[model],
                    "camera_crop_sha256": crop_inputs[model][
                        "contract_payload_sha256"
                    ],
                    "image_width_px": 320,
                    "image_height_px": 168 if model == "N3" else 176,
                    "crop_operation": f"fixture-{model}",
                },
                "measured_clock_intervals": {
                    "control_step_s_min": 1 / 15,
                    "control_step_s_max": 1 / 15,
                    "captured_frame_interval_s_min": 1 / 15,
                    "captured_frame_interval_s_max": 1 / 15,
                    "timestamp_tolerance_s": 1 / 30,
                    "tolerance_rule": (
                        "min(half minimum positive native control interval, half minimum "
                        "positive original-camera capture interval)"
                    ),
                },
                "primary_target": primary,
                "early_target": early,
                "frame_to_physical_time": rows,
                "source_receipts": {
                    "development_cell_receipts": [
                        descriptor(
                            jobs.CONTROL_ROOT / "jobs/alignment-fixture/raw"
                            / f"{model}-cell-{index}.json",
                            f"cell-{model}-{index}",
                        )
                        for index in range(16)
                    ],
                    "adapter_completions": [
                        descriptor(
                            jobs.CONTROL_ROOT / "jobs/alignment-fixture/raw"
                            / f"{model}-completion-{index}.json",
                            f"completion-{model}-{index}",
                        )
                        for index in range(16)
                    ],
                    "adapter_journals": [
                        descriptor(
                            jobs.CONTROL_ROOT / "jobs/alignment-fixture/raw"
                            / f"{model}-journal-{index}.json",
                            f"journal-{model}-{index}",
                        )
                        for index in range(16)
                    ],
                    "official_request_receipt_sha256s": [
                        digest(f"request-{model}-{index}")
                        for index in range(jobs.REQUEST_COUNTS[model])
                    ],
                    "generated_target_timing": timing_inputs[model]["sidecar"],
                    "resource_receipts": [None] * 16,
                    "camera_crop_contract": crop_inputs[model]["contract"],
                },
            })
            expected_mappings[model] = mapping
            mapping_path = bundle / jobs.OUTPUT_NAMES[model]["mapping"]
            jobs.queue.immutable_json(mapping_path, mapping)
            unsigned = {
                "contract_id": f"wmf1-{model.lower()}-alignment-v1",
                "model_id": model,
                "mapping_receipt_id": mapping["receipt_id"],
                "mapping_receipt_sha256": None,
                "primary_horizon_s": primary["target_physical_time_s"],
                "generated_frame_index": primary_id[0],
                "target_executed_action_offset": primary_id[1],
                "control_step_s": 1 / 15,
                "captured_frame_interval_s": 1 / 15,
                "timestamp_tolerance_s": 1 / 30,
                "camera_id": "over_shoulder_left_camera",
                "camera_crop_id": jobs.EXPECTED_CAMERA_CROP_IDS[model],
                "camera_crop_sha256": crop_inputs[model][
                    "contract_payload_sha256"
                ],
                "image_width_px": 320,
                "image_height_px": 168 if model == "N3" else 176,
                "early_horizon": {
                    "horizon_s": early["target_physical_time_s"],
                    "generated_frame_index": early_id[0],
                    "target_executed_action_offset": early_id[1],
                },
            }
            expected_alignment_unsigned[model] = copy.deepcopy(unsigned)
            unsigned["mapping_receipt_sha256"] = jobs.queue.sha256_file(mapping_path)
            alignment = {
                **unsigned,
                "contract_sha256": jobs.queue.sha256_bytes(
                    jobs.queue.compact_bytes(unsigned)
                ),
            }
            jobs.queue.immutable_json(
                bundle / jobs.OUTPUT_NAMES[model]["alignment"], alignment
            )
        mappings, alignments = jobs._validate_alignment_bundle(
            bundle, freeze=FakeFreeze, inputs=inputs,
            evidence_path=self.root / "evidence.json",
        )
        self.assertEqual(set(mappings), set(jobs.MODELS))
        self.assertEqual(set(alignments), set(jobs.MODELS))
        mapping_path = bundle / jobs.OUTPUT_NAMES["N3"]["mapping"]
        alignment_path = bundle / jobs.OUTPUT_NAMES["N3"]["alignment"]
        original_mapping = expected_mappings["N3"]
        changed_semantics = copy.deepcopy(original_mapping)
        changed_semantics.pop("payload_sha256")
        changed_semantics["request_semantics"]["action_space"] = "invented"
        changed_semantics = jobs.queue.signed_document(changed_semantics)
        expected_mappings["N3"] = changed_semantics
        mapping_path.write_bytes(jobs.queue.canonical_bytes(changed_semantics))
        changed_alignment = copy.deepcopy(expected_alignment_unsigned["N3"])
        changed_alignment["mapping_receipt_sha256"] = jobs.queue.sha256_file(
            mapping_path
        )
        changed_alignment["contract_sha256"] = jobs.queue.sha256_bytes(
            jobs.queue.compact_bytes(changed_alignment)
        )
        alignment_path.write_bytes(jobs.queue.canonical_bytes(changed_alignment))
        with self.assertRaisesRegex(
            jobs.DevelopmentAlignmentJobError, "request semantics changed"
        ):
            jobs._validate_alignment_bundle(
                bundle, freeze=FakeFreeze, inputs=inputs,
                evidence_path=self.root / "evidence.json",
            )
        expected_mappings["N3"] = original_mapping
        mapping_path.write_bytes(jobs.queue.canonical_bytes(original_mapping))
        restored_alignment = copy.deepcopy(expected_alignment_unsigned["N3"])
        restored_alignment["mapping_receipt_sha256"] = jobs.queue.sha256_file(
            mapping_path
        )
        restored_alignment["contract_sha256"] = jobs.queue.sha256_bytes(
            jobs.queue.compact_bytes(restored_alignment)
        )
        alignment_path.write_bytes(jobs.queue.canonical_bytes(restored_alignment))
        (bundle / "confirmation_release_freeze.json").write_text(
            "{}\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            jobs.DevelopmentAlignmentJobError, "file inventory changed"
        ):
            jobs._validate_alignment_bundle(
                bundle, freeze=FakeFreeze, inputs=inputs,
                evidence_path=self.root / "evidence.json",
            )
        (bundle / "confirmation_release_freeze.json").unlink()
        changed_mapping = copy.deepcopy(expected_mappings["N3"])
        changed_mapping.pop("payload_sha256")
        changed_mapping["frame_to_physical_time"][10].update({
            "target_physical_time_s": 999,
            "eligible_request_count": 999999,
            "full_prefix_request_count": -7,
            "max_camera_timestamp_residual_s": 999,
            "max_physics_timestamp_residual_s": -1,
        })
        changed_mapping = jobs.queue.signed_document(changed_mapping)
        expected_mappings["N3"] = changed_mapping
        mapping_path.write_bytes(jobs.queue.canonical_bytes(changed_mapping))
        changed_alignment = jobs.queue.load_json(alignment_path, "changed alignment")
        changed_alignment.pop("contract_sha256")
        changed_alignment["mapping_receipt_sha256"] = jobs.queue.sha256_file(mapping_path)
        changed_alignment["contract_sha256"] = jobs.queue.sha256_bytes(
            jobs.queue.compact_bytes(changed_alignment)
        )
        alignment_path.write_bytes(jobs.queue.canonical_bytes(changed_alignment))
        with self.assertRaisesRegex(
            jobs.DevelopmentAlignmentJobError, "frame 11"
        ):
            jobs._validate_alignment_bundle(
                bundle, freeze=FakeFreeze, inputs=inputs,
                evidence_path=self.root / "evidence.json",
            )

    def _transaction_fixture(self, name: str):
        job = self.root / name
        raw = job / "raw"
        publish = job / "publish"
        bundle = raw / "physical_alignment_bundle"
        bundle.mkdir(parents=True)
        publish.mkdir()
        for output in set(jobs.SUCCESS_PUBLISH_FILES) - {jobs.PUBLISH_RECEIPT_NAME}:
            (bundle / output).write_bytes((output + "\n").encode("utf-8"))
        context = SimpleNamespace(
            job_id=jobs.JOB_ID,
            job_dir=job,
            study_commit="3" * 40,
            role=jobs.WORKER_ROLE,
            worker_id=jobs.WORKER_ROLE,
            hostname=jobs.WORKER_ROLE + "-test",
            pod_uid="test-pod",
            descriptor_identity=descriptor(job / "descriptor.json", "descriptor"),
            claim_identity=descriptor(job / "claim.json", "claim"),
        )
        return job, raw, publish, bundle, context

    def test_publication_is_five_file_atomic_and_byte_identical(self) -> None:
        _, raw, publish, bundle, context = self._transaction_fixture("success")
        receipt = jobs._publish_transactionally(
            raw=raw,
            publish=publish,
            bundle=bundle,
            context=context,
            implementation={},
            inputs=self._normalized_input(),
            input_identity=descriptor(raw / "input.json", "input"),
            evidence_identity=descriptor(raw / "evidence.json", "evidence"),
            raw_manifest_identity=descriptor(raw / "manifest.json", "manifest"),
        )
        self.assertEqual(
            {path.name for path in publish.iterdir()},
            set(jobs.SUCCESS_PUBLISH_FILES),
        )
        self.assertEqual(receipt["science_counts"], jobs._zero_science_counts())
        self.assertFalse(receipt["confirmation_authority"])
        for model in jobs.MODELS:
            for role in ("mapping", "alignment"):
                name = jobs.OUTPUT_NAMES[model][role]
                self.assertEqual((publish / name).read_bytes(), (bundle / name).read_bytes())

    def test_receipt_failure_after_four_copies_never_exposes_partial_publish(self) -> None:
        _, raw, publish, bundle, context = self._transaction_fixture("failure")

        def fail_after_write(path, value, *, maximum_bytes):
            jobs.queue.immutable_json(path, value, maximum_bytes=maximum_bytes)
            raise RuntimeError("injected receipt failure")

        with self.assertRaisesRegex(RuntimeError, "injected receipt failure"):
            jobs._publish_transactionally(
                raw=raw,
                publish=publish,
                bundle=bundle,
                context=context,
                implementation={},
                inputs=self._normalized_input(),
                input_identity=descriptor(raw / "input.json", "input"),
                evidence_identity=descriptor(raw / "evidence.json", "evidence"),
                raw_manifest_identity=descriptor(raw / "manifest.json", "manifest"),
                receipt_writer=fail_after_write,
            )
        self.assertTrue(publish.is_dir())
        self.assertEqual(list(publish.iterdir()), [])
        staging = context.job_dir / jobs.STAGING_RELATIVE
        self.assertEqual(
            {path.name for path in staging.iterdir()},
            set(jobs.SUCCESS_PUBLISH_FILES),
        )

    def test_source_has_no_duplicate_dict_keys_or_confirmation_true(self) -> None:
        source = MODULE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                keys = [
                    key.value for key in node.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                ]
                self.assertEqual(len(keys), len(set(keys)))
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "write_bundle":
                keyword = next(
                    item for item in node.keywords
                    if item.arg == "confirmation_release"
                )
                self.assertIsInstance(keyword.value, ast.Constant)
                self.assertIs(keyword.value.value, False)


if __name__ == "__main__":
    unittest.main()
