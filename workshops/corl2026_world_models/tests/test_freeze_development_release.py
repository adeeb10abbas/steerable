from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


MODULE = Path(__file__).resolve().parents[1] / "analysis/freeze_development_release.py"
SPEC = importlib.util.spec_from_file_location("freeze_development_release", MODULE)
freeze = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(freeze)

ANNOTATION_MODULE = Path(__file__).resolve().parents[1] / "analysis/forecast_annotation_workflow.py"
ANNOTATION_SPEC = importlib.util.spec_from_file_location("freeze_annotation_compat", ANNOTATION_MODULE)
annotation = importlib.util.module_from_spec(ANNOTATION_SPEC)
assert ANNOTATION_SPEC.loader is not None
ANNOTATION_SPEC.loader.exec_module(annotation)

WORKSHOP = Path(__file__).resolve().parents[1]
FORECAST = WORKSHOP / "experiments/forecast_layout"
# Deliberately non-corresponding generated-frame and executed-action indices.
# These elapsed seconds stand in for an explicit native runtime field; the
# release tool must discover action offsets from recorder clocks, not ordinals.
SYNTHETIC_NATIVE_TARGETS = (
    (0, 0.0),
    (4, 0.06666666666666667),
    (9, 0.2),
    (14, 0.5333333333333333),
)


def write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def descriptor(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": freeze.sha256_file(path),
        "bytes": path.stat().st_size,
    }


def signed(value: dict) -> dict:
    return freeze.sign_document(value)


def d1_missingness_timing_fixture(
    request_indices: list[int] | None = None,
    *,
    truncated_ordinals: set[int] | None = None,
) -> tuple[dict, list[str], list[dict]]:
    if request_indices is None:
        request_indices = list(range(4))
    truncated_ordinals = set() if truncated_ordinals is None else truncated_ordinals
    targets = [
        {
            "generated_frame_index": 1,
            "target_physical_time_s": 0.2,
            "native_runtime_field": "request_timing_sidecar.generated_targets",
        },
        {
            "generated_frame_index": 2,
            "target_physical_time_s": 0.4,
            "native_runtime_field": "request_timing_sidecar.generated_targets",
        },
    ]
    hashes: list[str] = []
    receipts: list[dict] = []
    bindings: list[dict] = []
    mapped = unmapped = matched_requests = matched_targets = 0
    truncated_requests = truncated_targets = 0
    for ordinal, request_index in enumerate(request_indices):
        request_hash = hashlib.sha256(
            f"d1-request-{ordinal}-{request_index}".encode()
        ).hexdigest()
        hashes.append(request_hash)
        kind = (
            "full_conditioning_origin"
            if request_index % 4 == 0
            else "incremental_standalone"
        )
        shape = freeze.D1_DECODE_SHAPES[kind]
        receipts.append({
            "request_index": request_index,
            "latent_video": {"shape": list(shape["latent_shape"])},
            "offline_decode": {
                "decoded_tensor": {"shape": list(shape["decoded_tensor_shape"])},
                "decoded_rgb": {"shape": list(shape["decoded_rgb_shape"])},
            },
        })
        applies = kind == "full_conditioning_origin"
        target_status = (
            "not_executed_in_truncated_prefix"
            if ordinal in truncated_ordinals
            else "matched_native_request_clocks"
        )
        target_bindings = (
            [
                {
                    "generated_frame_index": target["generated_frame_index"],
                    "authority_target_physical_time_s": target["target_physical_time_s"],
                    "status": target_status,
                }
                for target in targets
            ]
            if applies
            else []
        )
        mapped += int(applies)
        unmapped += int(not applies)
        matched = applies and ordinal not in truncated_ordinals
        truncated = applies and ordinal in truncated_ordinals
        matched_requests += int(matched)
        truncated_requests += int(truncated)
        matched_targets += len(target_bindings) if matched else 0
        truncated_targets += len(target_bindings) if truncated else 0
        bindings.append({
            "request_index": request_index,
            "source_request_receipt": {
                "path": f"/evidence/request-{request_index}.json",
                "sha256": request_hash,
                "bytes": 1,
            },
            "decoded_output_timing": {
                "schedule_kind": kind,
                "request_index_modulo_four": request_index % 4,
                "latent_shape": list(shape["latent_shape"]),
                "decoded_tensor_shape": list(shape["decoded_tensor_shape"]),
                "decoded_rgb_shape": list(shape["decoded_rgb_shape"]),
                "authority_target_frame_indices_present": True,
                "authority_target_frame_indices": [1, 2],
                "source_timing_status": shape["source_timing_status"],
                "source_timing_mapping_applies": applies,
                "unmapped_reason": None if applies else "fresh standalone VAE cache",
            },
            "target_bindings": target_bindings,
            "timing_eligible_target_count": len(target_bindings) if matched else 0,
            "eligible_for_timed_target_sampling": bool(matched),
        })
    timing = {
        "schema_version": freeze.TIMING_MISSINGNESS_SCHEMA,
        "status": freeze.D1_TIMING_STATUS,
        "native_runtime_field": "request_timing_sidecar.generated_targets",
        "time_source_kind": "native_runtime_exposed_target_offsets",
        "clock_bridge": (
            "elapsed physical seconds from request current original-camera capture"
        ),
        "generated_targets": targets,
        "generated_targets_scope": freeze.D1_GENERATED_TARGETS_SCOPE,
        "incremental_standalone_decodes_assigned_target_times": False,
        "confirmation_release_compatible": False,
        "request_timing_bindings": bindings,
        "request_timing_coverage": {
            "total_request_count": len(request_indices),
            "source_proven_full_decode_request_count": mapped,
            "source_unmapped_incremental_decode_request_count": unmapped,
            "native_clock_matched_request_count": matched_requests,
            "action_prefix_truncated_request_count": truncated_requests,
            "native_clock_matched_target_binding_count": matched_targets,
            "action_prefix_truncated_target_binding_count": truncated_targets,
            "unmapped_potential_authority_target_count": unmapped * len(targets),
        },
    }
    return timing, hashes, receipts


def journal_event(sequence: int, previous: str | None, kind: str, payload: dict) -> dict:
    base = {
        "sequence": sequence,
        "kind": kind,
        "wall_time_ns": 2_000_000_000 + sequence,
        "monotonic_ns": 3_000_000_000 + sequence,
        "previous_event_sha256": previous,
        "payload": payload,
    }
    return {**base, "event_sha256": freeze.sha256_bytes(freeze.recorder_canonical_bytes(base))}


class EvidenceFixture:
    def __init__(self, root: Path):
        self.root = root
        self.cell_entries: list[dict] = []
        self.request_hashes: list[str] = []
        self.crop_path = self._make_crop()
        self._make_cells()
        self.timing_path = self._make_timing()
        self.evidence = {
            "schema_version": freeze.EVIDENCE_SCHEMA,
            "study_id": freeze.STUDY_ID,
            "cohort_branch": "reduced_n3",
            "qualified_model_ids": ["N3"],
            "ablation_spec": descriptor(FORECAST / "ablation_spec.json"),
            "planned_cells": descriptor(FORECAST / "planned_cells.csv"),
            "model_evidence": [{
                "model_id": "N3",
                "camera_crop_contract": descriptor(self.crop_path),
                "generated_target_timing_receipt": descriptor(self.timing_path),
                "development_cells": self.cell_entries,
            }],
            "annotation": None,
            "resource_budget_policy": None,
        }
        self.evidence_path = write_json(root / "development_evidence.json", self.evidence)

    def _make_crop(self) -> Path:
        return write_json(self.root / "crop.json", signed({
            "schema_version": freeze.CROP_SCHEMA,
            "study_id": freeze.STUDY_ID,
            "model_id": "N3",
            "status": "qualified_from_original_camera_pixels",
            "camera_id": "over_shoulder_left_camera",
            "camera_crop_id": "n3-primary-original-camera-v1",
            "crop_operation": "identity original RGB; no simulator-state render",
            "image_width_px": 320,
            "image_height_px": 180,
            "simulator_state_render_used": False,
        }))

    def _make_request(self, cell_root: Path, cell_id: str, index: int) -> dict:
        native_targets = [
            {"generated_frame_index": frame_index, "target_physical_time_s": target_time}
            for frame_index, target_time in SYNTHETIC_NATIVE_TARGETS
        ]
        path = write_json(cell_root / "requests" / f"{index:04d}.json", {
            "schema_version": freeze.MODEL_LIMITS["N3"]["request_schema"],
            "status": "passed",
            "study_id": freeze.STUDY_ID,
            "cell_id": cell_id,
            "request_index": index,
            "action_step_start": index * 32,
            "behavioral_model_request": True,
            "decoded_future_shape": [33, 180, 320, 3],
            "native_generated_target_times": native_targets,
        })
        item = descriptor(path)
        self.request_hashes.append(item["sha256"])
        return item

    def _make_resource(self, cell_root: Path, cell_id: str, index: int) -> dict:
        path = write_json(cell_root / "resource.json", signed({
            "schema_version": freeze.RESOURCE_SCHEMA,
            "study_id": freeze.STUDY_ID,
            "model_id": "N3",
            "cell_id": cell_id,
            "measurement_complete": True,
            "episode_wall_seconds": 100.0 + index,
            "inference_wall_seconds_total": 80.0 + index,
            "raw_recording_bytes": 1_000_000 + index,
            "peak_gpu_allocated_bytes": 2_000_000_000 + index,
            "peak_gpu_reserved_bytes": 2_500_000_000 + index,
            "gpu_count": 1,
        }))
        return descriptor(path)

    def _make_cells(self) -> None:
        rows = []
        with (FORECAST / "planned_cells.csv").open(encoding="utf-8") as handle:
            import csv
            rows = [row for row in csv.DictReader(handle)
                    if row["phase"] == "development" and row["model_config"] == "N3"]
        self.assert_cell_count = len(rows)
        for cell_index, planned in enumerate(rows):
            cell_id = planned["cell_id"]
            cell_root = self.root / "cells" / cell_id
            request_entries = [self._make_request(cell_root, cell_id, index) for index in range(15)]
            events = []
            previous = None

            def append(kind: str, payload: dict) -> None:
                nonlocal previous
                row = journal_event(len(events), previous, kind, payload)
                events.append(row)
                previous = row["event_sha256"]

            append("attempt_started", {"identity": {"cell_id": cell_id}})
            for action in range(451):
                capture_ns = 1_000_000_000 + action * 66_666_667
                append("observation_captured", {
                    "observation_id": f"obs_{action:06d}",
                    "control_step": action,
                    "phase": "settled_reset" if action == 0 else "post_action",
                    "clock": {
                        "physics_step": action * 8,
                        "physics_time_s": action / 15,
                        "control_step": action,
                        "cameras": {
                            "over_shoulder_left_camera": {
                                "frame_id": f"left:{action}",
                                "capture_time_ns": capture_ns,
                                "timestamp_source": "Isaac native simulation-time camera counter",
                            },
                            "over_shoulder_right_camera": {
                                "frame_id": f"right:{action}",
                                "capture_time_ns": capture_ns,
                                "timestamp_source": "Isaac native simulation-time camera counter",
                            },
                            "wrist_cam": {
                                "frame_id": f"wrist:{action}",
                                "capture_time_ns": capture_ns,
                                "timestamp_source": "Isaac native simulation-time camera counter",
                            },
                        },
                    },
                })
            for request_index in range(15):
                start = request_index * 32
                append("model_request_packed", {
                    "request_index": request_index,
                    "action_step_start": start,
                    "current_observation_id": f"obs_{start:06d}",
                })
            append("attempt_finalized", {"stop_reason": "action_cap"})
            journal_path = cell_root / "events.partial.jsonl"
            journal_path.parent.mkdir(parents=True, exist_ok=True)
            journal_path.write_bytes(b"".join(freeze.canonical_bytes(row) + b"\n" for row in events))
            completion = {
                "schema_version": "wmf-forecast-recording-attempt-v1",
                "study_id": freeze.STUDY_ID,
                "identity": {
                    "cell_id": cell_id,
                    "stage": "development",
                    "model_config": "N3",
                },
                "stop_reason": "action_cap",
                "behavioral_result_valid": True,
                "technical_invalid": False,
                "right_censored": False,
                "actions_executed": 450,
                "observation_count": 451,
                "request_count": 15,
                "request_execution": [
                    {
                        "request_index": index,
                        "action_step_start": index * 32,
                        "returned_actions": 32,
                        "eligible_executable_prefix_actions": 32,
                        "executed_actions": 2 if index == 14 else 32,
                    }
                    for index in range(15)
                ],
                "event_count": len(events),
                "journal_path": str(journal_path.resolve()),
                "journal_tail_sha256": previous,
            }
            completion_path = write_json(cell_root / "completion.json", completion)
            journal_descriptor = descriptor(journal_path)
            journal_descriptor.update(event_count=len(events), tail_sha256=previous)
            cell = {
                "schema_version": freeze.MODEL_LIMITS["N3"]["cell_schema"],
                "status": "passed",
                "study_id": freeze.STUDY_ID,
                "cell_id": cell_id,
                "model_config": "N3",
                "layout_pair_id": planned["layout_pair_id"],
                "layout_arm": planned["layout_arm"],
                "command": planned["command"],
                "prompt": (
                    "Put the Rubik's cube to the left of the bowl."
                    if planned["command"] == "left"
                    else "Put the Rubik's cube to the right of the bowl."
                ),
                "actions_executed": 450,
                "observation_count": 451,
                "behavioral_model_request_count": 15,
                "behavioral_episode_count": 1,
                "generation_qualification_request_count": 0,
                "adapter_completion": descriptor(completion_path),
                "adapter_journal": journal_descriptor,
            }
            cell_path = write_json(cell_root / "cell_receipt.json", cell)
            self.cell_entries.append({
                "cell_receipt": descriptor(cell_path),
                "server_request_receipts": request_entries,
                "resource_receipt": self._make_resource(cell_root, cell_id, cell_index),
            })

    def _make_timing(self) -> Path:
        return write_json(self.root / "n3_generated_timing.json", signed({
            "schema_version": freeze.TIMING_SCHEMA,
            "study_id": freeze.STUDY_ID,
            "model_id": "N3",
            "status": "qualified_from_native_runtime_metadata",
            "time_source_kind": "native_runtime_exposed_target_offsets",
            "native_runtime_field": "native_generated_target_times",
            "clock_bridge": "elapsed physical seconds from request current original-camera capture",
            "presentation_video_fps_used": False,
            "conditioning_fps_used_as_target_timing": False,
            "generated_frame_index_interpreted_as_action_index": False,
            "source_request_receipt_sha256s": self.request_hashes,
            "generated_targets": [
                {
                    "generated_frame_index": frame_index,
                    "target_physical_time_s": target_time,
                    "native_runtime_field": "native_generated_target_times",
                }
                for frame_index, target_time in SYNTHETIC_NATIVE_TARGETS
            ],
        }))

    def add_annotation_and_budget(self) -> None:
        mappings, unsigned, _ = freeze.derive_bundle(self.evidence_path, require_annotation=False)
        mapping_payload = freeze.pretty_json_bytes(mappings["N3"])
        contract = dict(unsigned["N3"])
        contract["mapping_receipt_sha256"] = freeze.sha256_bytes(mapping_payload)
        contract["contract_sha256"] = freeze.sha256_bytes(freeze.canonical_bytes(contract))

        rubric_path = write_json(self.root / "rubric.json", {
            "schema_version": "wmf-forecast-annotation-rubric-v1",
            "study_id": freeze.STUDY_ID,
            "status": "frozen",
            "freeze_scope": "development_duplicate_validation_and_confirmation",
        })
        restricted_map = write_json(self.root / "restricted_map.json", {"locked": True})
        rater_a = write_json(self.root / "rater_a.json", {"rater": "a", "locked": True})
        rater_b = write_json(self.root / "rater_b.json", {"rater": "b", "locked": True})
        summary_path = write_json(self.root / "development_summary.json", signed({
            "schema_version": freeze.DEVELOPMENT_SUMMARY_SCHEMA,
            "study_id": freeze.STUDY_ID,
            "qualified_model_ids": ["N3"],
            "qualified_alignment_contract_sha256_by_model": {
                "N3": contract["contract_sha256"],
            },
            "status": "EMPIRICAL_DEVELOPMENT_RESULT_REQUIRES_EXPLICIT_USABILITY_DECISION",
            "distinct_raters_attested": True,
            "eligible_duplicate_count": 10,
            "movement_resolution_threshold_relative_image_diagonal": 0.0125,
            "movement_disagreement_definition": freeze.MOVEMENT_DISAGREEMENT_DEFINITION,
            "quantile_probability": 0.95,
            "quantile_method": freeze.MOVEMENT_QUANTILE_METHOD,
            "rubric_sha256": freeze.sha256_file(rubric_path),
            "source_files": {
                "restricted_map": {
                    "path": str(restricted_map.resolve()),
                    "artifact_sha256": freeze.artifact_sha256(restricted_map),
                },
                "rater_a_responses": {
                    "path": str(rater_a.resolve()),
                    "artifact_sha256": freeze.artifact_sha256(rater_a),
                },
                "rater_b_responses": {
                    "path": str(rater_b.resolve()),
                    "artifact_sha256": freeze.artifact_sha256(rater_b),
                },
            },
            "restricted_map_sha256": freeze.artifact_sha256(restricted_map),
            "rater_a_response_sha256": freeze.artifact_sha256(rater_a),
            "rater_b_response_sha256": freeze.artifact_sha256(rater_b),
            "annotation_seconds": {"rater_a_total": 120.0, "rater_b_total": 130.0},
        }))
        adjudication_map = write_json(self.root / "adjudication_map.json", {"locked": True})
        consensus_path = write_json(self.root / "final_consensus.json", signed({
            "schema_version": freeze.FINAL_CONSENSUS_SCHEMA,
            "study_id": freeze.STUDY_ID,
            "stage": "development",
            "status": "mechanically_merged_from_locked_blind_responses",
            "adjudication_map_path": str(adjudication_map.resolve()),
            "adjudication_map_sha256": freeze.sha256_file(adjudication_map),
            "source_restricted_map_sha256": freeze.artifact_sha256(restricted_map),
            "first_pass_response_sha256_by_slot": {
                "rater_a": freeze.artifact_sha256(rater_a),
                "rater_b": freeze.artifact_sha256(rater_b),
            },
            "adjudicator_response_path": None,
            "adjudicator_response_sha256": None,
        }))
        self.evidence["annotation"] = {
            "rubric": descriptor(rubric_path),
            "development_summary": descriptor(summary_path),
            "final_consensus": descriptor(consensus_path),
            "usability_decision": {
                "measurement_usable": True,
                "decided_by": "authorized-reviewer-test",
                "decided_at": "2026-09-13T01:00:00Z",
                "basis": "test-only inspected duplicate disagreement and coverage",
                "development_summary_sha256": freeze.sha256_file(summary_path),
            },
        }
        self.evidence["resource_budget_policy"] = {
            "headroom_multiplier": 1.25,
            "selected_execution_host": "GM B200 worker pool",
            "max_parallel_blocks_by_model": {"N3": 2},
            "authorized_gpu_memory_bytes_per_gpu": 180_000_000_000,
            "confirmation_cells_per_model": 96,
            "confirmation_annotation_judgment_ceiling": 4608,
        }
        self.evidence_path = write_json(self.evidence_path, self.evidence)


class DevelopmentReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = EvidenceFixture(self.root)
        self.assertEqual(self.fixture.assert_cell_count, 16)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_alignment_derives_native_primary_and_earliest_horizon(self) -> None:
        output = self.root / "alignment"
        result = freeze.write_bundle(self.fixture.evidence_path, output, confirmation_release=False)
        self.assertEqual(result["status"], "alignment_qualified")
        alignment = json.loads((output / "n3_alignment_contract.json").read_text())
        self.assertEqual(alignment["generated_frame_index"], 14)
        self.assertEqual(alignment["target_executed_action_offset"], 8)
        self.assertAlmostEqual(alignment["primary_horizon_s"], 8 / 15)
        self.assertEqual(alignment["early_horizon"]["generated_frame_index"], 4)
        self.assertEqual(alignment["early_horizon"]["target_executed_action_offset"], 1)
        validated = annotation._validate_alignment_contracts(
            [alignment], stage="development", cohort_branch="reduced_n3"
        )
        self.assertEqual(validated["N3"], alignment)
        mapping = json.loads((output / "n3_physical_alignment_receipt.json").read_text())
        self.assertFalse(mapping["timing_claim_boundary"]["presentation_video_fps_used"])
        self.assertEqual(mapping["development_cell_count"], 16)
        self.assertEqual(mapping["development_request_count"], 240)

    def test_d1_missingness_mask_keeps_incremental_decodes_ineligible(self) -> None:
        timing, hashes, receipts = d1_missingness_timing_fixture()
        self.assertEqual(
            freeze._d1_timing_applicability(
                timing, request_hashes=hashes, request_receipts=receipts
            ),
            [True, False, False, False],
        )

        forged = copy.deepcopy(timing)
        forged["request_timing_bindings"][1]["target_bindings"] = [
            copy.deepcopy(forged["request_timing_bindings"][0]["target_bindings"][0])
        ]
        with self.assertRaisesRegex(
            freeze.FreezeError, "incremental decode binding 1 acquired timing eligibility"
        ):
            freeze._d1_timing_applicability(
                forged, request_hashes=hashes, request_receipts=receipts
            )

    def test_d1_missingness_discriminator_and_coverage_fail_closed(self) -> None:
        timing, hashes, receipts = d1_missingness_timing_fixture()
        cases = []
        wrong_schedule = copy.deepcopy(timing)
        wrong_schedule["request_timing_bindings"][1]["decoded_output_timing"][
            "source_timing_mapping_applies"
        ] = True
        cases.append((wrong_schedule, "decode discriminator changed"))
        omitted_target = copy.deepcopy(timing)
        omitted_target["request_timing_bindings"][0]["target_bindings"].pop()
        cases.append((omitted_target, "omitted an authority target"))
        wrong_coverage = copy.deepcopy(timing)
        wrong_coverage["request_timing_coverage"][
            "source_unmapped_incremental_decode_request_count"
        ] -= 1
        cases.append((wrong_coverage, "coverage summary changed"))
        for forged, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                freeze.FreezeError, message
            ):
                freeze._d1_timing_applicability(
                    forged, request_hashes=hashes, request_receipts=receipts
                )

    def test_d1_alignment_uses_only_source_proven_full_decode_requests(self) -> None:
        request_indices = list(range(57)) * 16
        truncated = {cell_index * 57 + 56 for cell_index in range(16)}
        timing, hashes, receipts = d1_missingness_timing_fixture(
            request_indices, truncated_ordinals=truncated
        )
        self.assertEqual(timing["request_timing_coverage"], freeze.D1_FORMAL_TIMING_COVERAGE)
        applicability = freeze._d1_timing_applicability(
            timing, request_hashes=hashes, request_receipts=receipts
        )
        self.assertEqual(sum(applicability), 240)

        observations = []
        for action in range(451):
            capture_ns = 1_000_000_000 + action * 66_666_667
            observations.append({
                "clock": {
                    "physics_step": action * 8,
                    "physics_time_s": action / 15,
                    "control_step": action,
                    "cameras": {
                        "over_shoulder_left_camera": {
                            "frame_id": f"left:{action}",
                            "capture_time_ns": capture_ns,
                            "timestamp_source": "native test camera clock",
                        }
                    },
                }
            })

        expected_cells: set[str] = set()
        entries: list[dict] = []
        validated_cells: dict[str, dict] = {}
        for cell_index in range(16):
            cell_id = f"d1-development-cell-{cell_index:02d}"
            expected_cells.add(cell_id)
            identity_path = write_json(
                self.root / "d1-identities" / f"{cell_id}.json",
                {"cell_id": cell_id},
            )
            identity = descriptor(identity_path)
            entries.append({"cell_receipt": identity})
            start = cell_index * 57
            validated_cells[cell_id] = {
                "cell_id": cell_id,
                "cell_receipt": identity,
                "adapter_completion": identity,
                "adapter_journal": identity,
                "request_receipt_sha256s": hashes[start:start + 57],
                "request_receipts": receipts[start:start + 57],
                "observations": observations,
                "request_execution": [
                    {
                        "request_index": request_index,
                        "action_step_start": request_index * 8,
                        "executed_actions": 2 if request_index == 56 else 8,
                    }
                    for request_index in range(57)
                ],
                "resource_receipt": None,
                "resource": None,
            }

        def validated_cell(
            entry: dict,
            *,
            model: str,
            expected_cell_id: str,
            evidence_base: Path,
            require_resource: bool,
        ) -> dict:
            del entry, evidence_base, require_resource
            self.assertEqual(model, "D1")
            return validated_cells[expected_cell_id]

        crop = {
            "camera_id": "over_shoulder_left_camera",
            "camera_crop_id": "d1-primary-original-camera-v1",
            "payload_sha256": "a" * 64,
            "image_width_px": 640,
            "image_height_px": 352,
            "crop_operation": "identity original RGB; no simulator-state render",
            "file_sha256": freeze.sha256_file(self.fixture.crop_path),
            "path": str(self.fixture.crop_path),
        }
        model_evidence = {
            "model_id": "D1",
            "development_cells": entries,
        }
        with mock.patch.object(
            freeze, "_validate_cell", side_effect=validated_cell
        ), mock.patch.object(
            freeze,
            "_validate_generated_timing",
            return_value=(timing, self.fixture.timing_path, applicability),
        ), mock.patch.object(freeze, "_validate_crop", return_value=crop):
            mapping, _, _ = freeze._derive_model_alignment(
                model_evidence,
                model="D1",
                expected_cells=expected_cells,
                evidence_base=self.root,
                require_resources=False,
            )

        self.assertEqual(
            mapping["timing_claim_boundary"]["request_timing_coverage"],
            freeze.D1_FORMAL_TIMING_COVERAGE,
        )
        self.assertFalse(
            mapping["timing_claim_boundary"]["timing_unmapped_requests_eligible"]
        )
        qualified = [
            row for row in mapping["frame_to_physical_time"]
            if row["status"] == "qualified"
        ]
        self.assertEqual(
            [row["target_executed_action_offset"] for row in qualified], [3, 6]
        )
        self.assertTrue(all(row["eligible_request_count"] == 224 for row in qualified))
        self.assertTrue(all(row["full_prefix_request_count"] == 224 for row in qualified))

    def test_alignment_only_defers_absent_resource_receipts(self) -> None:
        for cell in self.fixture.evidence["model_evidence"][0]["development_cells"]:
            cell["resource_receipt"] = None
        self.fixture.evidence_path = write_json(
            self.fixture.evidence_path, self.fixture.evidence
        )
        mappings, contracts, annotation = freeze.derive_bundle(
            self.fixture.evidence_path, require_annotation=False
        )
        self.assertIsNone(annotation)
        self.assertEqual(mappings["N3"]["development_cell_count"], 16)
        self.assertEqual(contracts["N3"]["model_id"], "N3")

    def test_confirmation_derivation_still_requires_every_resource_receipt(self) -> None:
        self.fixture.evidence["model_evidence"][0]["development_cells"][0][
            "resource_receipt"
        ] = None
        self.fixture.evidence_path = write_json(
            self.fixture.evidence_path, self.fixture.evidence
        )
        with self.assertRaisesRegex(
            freeze.FreezeError, "resource receipt is required for confirmation"
        ):
            freeze.derive_bundle(self.fixture.evidence_path, require_annotation=True)

    def test_alignment_only_authenticates_a_supplied_resource_receipt(self) -> None:
        resource = self.fixture.evidence["model_evidence"][0]["development_cells"][0][
            "resource_receipt"
        ]
        Path(resource["path"]).write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(freeze.FreezeError, "resource receipt.*hash mismatch"):
            freeze.derive_bundle(self.fixture.evidence_path, require_annotation=False)

    def test_confirmation_freeze_requires_labels_and_writes_nothing_on_failure(self) -> None:
        output = self.root / "not-written"
        with self.assertRaisesRegex(freeze.FreezeError, "annotation evidence is missing"):
            freeze.write_bundle(self.fixture.evidence_path, output, confirmation_release=True)
        self.assertFalse(output.exists())

    def test_confirmation_freeze_and_runtime_validation(self) -> None:
        self.fixture.add_annotation_and_budget()
        output = self.root / "release"
        with mock.patch.object(freeze, "_deep_validate_annotation_evidence"):
            result = freeze.write_bundle(self.fixture.evidence_path, output, confirmation_release=True)
        release_path = output / result["release_freeze"]
        loaded = freeze.validate_release_freeze(
            release_path, result["release_freeze_sha256"], expected_model="N3"
        )
        self.assertTrue(loaded["release_decision"]["eligible"])
        self.assertEqual(
            loaded["annotation"]["movement_resolution"]["threshold_relative_image_diagonal"],
            0.0125,
        )
        self.assertEqual(
            loaded["resource_budget"]["confirmation_cells_per_model"], 96
        )
        with self.assertRaisesRegex(freeze.FreezeError, "D1 is not qualified"):
            freeze.validate_release_freeze(
                release_path, result["release_freeze_sha256"], expected_model="D1"
            )

    def test_prohibited_fps_timing_fails_closed(self) -> None:
        timing = json.loads(self.fixture.timing_path.read_text())
        timing["conditioning_fps_used_as_target_timing"] = True
        timing.pop("payload_sha256")
        write_json(self.fixture.timing_path, signed(timing))
        model = self.fixture.evidence["model_evidence"][0]
        model["generated_target_timing_receipt"] = descriptor(self.fixture.timing_path)
        self.fixture.evidence_path = write_json(self.fixture.evidence_path, self.fixture.evidence)
        with self.assertRaisesRegex(freeze.FreezeError, "prohibited FPS/frame/action"):
            freeze.write_bundle(
                self.fixture.evidence_path, self.root / "blocked", confirmation_release=False
            )

    def test_missing_development_cell_fails_closed(self) -> None:
        self.fixture.evidence["model_evidence"][0]["development_cells"].pop()
        self.fixture.evidence_path = write_json(self.fixture.evidence_path, self.fixture.evidence)
        with self.assertRaisesRegex(freeze.FreezeError, "exactly 16 development cells"):
            freeze.write_bundle(
                self.fixture.evidence_path, self.root / "blocked", confirmation_release=False
            )

    def test_request_receipt_tamper_is_detected(self) -> None:
        request_path = Path(
            self.fixture.evidence["model_evidence"][0]["development_cells"][0]
            ["server_request_receipts"][0]["path"]
        )
        request = json.loads(request_path.read_text())
        request["action_step_start"] = 1
        write_json(request_path, request)
        with self.assertRaisesRegex(freeze.FreezeError, "file hash mismatch"):
            freeze.write_bundle(
                self.fixture.evidence_path, self.root / "blocked", confirmation_release=False
            )

    def test_named_native_timing_field_must_exist_in_actual_request_receipts(self) -> None:
        timing = json.loads(self.fixture.timing_path.read_text())
        timing.pop("payload_sha256")
        timing["native_runtime_field"] = "plausible_but_absent.target_times"
        for row in timing["generated_targets"]:
            row["native_runtime_field"] = timing["native_runtime_field"]
        write_json(self.fixture.timing_path, signed(timing))
        self.fixture.evidence["model_evidence"][0]["generated_target_timing_receipt"] = descriptor(
            self.fixture.timing_path
        )
        self.fixture.evidence_path = write_json(self.fixture.evidence_path, self.fixture.evidence)
        with self.assertRaisesRegex(freeze.FreezeError, "lacks native timing field"):
            freeze.write_bundle(
                self.fixture.evidence_path, self.root / "blocked", confirmation_release=False
            )

    def test_native_timing_target_must_name_a_real_decoded_frame(self) -> None:
        timing = json.loads(self.fixture.timing_path.read_text())
        timing.pop("payload_sha256")
        timing["generated_targets"][-1]["generated_frame_index"] = 33
        write_json(self.fixture.timing_path, signed(timing))
        for cell_entry in self.fixture.evidence["model_evidence"][0]["development_cells"]:
            for request_descriptor in cell_entry["server_request_receipts"]:
                request_path = Path(request_descriptor["path"])
                request = json.loads(request_path.read_text())
                request["native_generated_target_times"][-1]["generated_frame_index"] = 33
                write_json(request_path, request)
                request_descriptor.update(descriptor(request_path))
        timing = json.loads(self.fixture.timing_path.read_text())
        timing.pop("payload_sha256")
        timing["source_request_receipt_sha256s"] = [
            request["sha256"]
            for cell in self.fixture.evidence["model_evidence"][0]["development_cells"]
            for request in cell["server_request_receipts"]
        ]
        write_json(self.fixture.timing_path, signed(timing))
        self.fixture.evidence["model_evidence"][0]["generated_target_timing_receipt"] = descriptor(
            self.fixture.timing_path
        )
        self.fixture.evidence_path = write_json(self.fixture.evidence_path, self.fixture.evidence)
        with self.assertRaisesRegex(freeze.FreezeError, "nonexistent decoded frame"):
            freeze.write_bundle(
                self.fixture.evidence_path, self.root / "blocked", confirmation_release=False
            )

    def test_manifest_and_output_symlinks_fail_before_resolution(self) -> None:
        evidence_link = self.root / "evidence-link.json"
        evidence_link.symlink_to(self.fixture.evidence_path)
        with self.assertRaisesRegex(freeze.FreezeError, "manifest is a symlink"):
            freeze.write_bundle(
                evidence_link, self.root / "blocked-evidence", confirmation_release=False
            )
        output_target = self.root / "actual-output"
        output_link = self.root / "output-link"
        output_link.symlink_to(output_target)
        with self.assertRaisesRegex(freeze.FreezeError, "output directory is a symlink"):
            freeze.write_bundle(
                self.fixture.evidence_path, output_link, confirmation_release=False
            )
        self.assertFalse(output_target.exists())

    def test_unusable_development_measurement_cannot_release(self) -> None:
        self.fixture.add_annotation_and_budget()
        self.fixture.evidence["annotation"]["usability_decision"]["measurement_usable"] = False
        self.fixture.evidence_path = write_json(self.fixture.evidence_path, self.fixture.evidence)
        with mock.patch.object(freeze, "_deep_validate_annotation_evidence"):
            with self.assertRaisesRegex(freeze.FreezeError, "not explicitly approved"):
                freeze.write_bundle(
                    self.fixture.evidence_path, self.root / "blocked", confirmation_release=True
                )

    def test_runtime_validator_rejects_symlink_before_resolve(self) -> None:
        self.fixture.add_annotation_and_budget()
        output = self.root / "release"
        with mock.patch.object(freeze, "_deep_validate_annotation_evidence"):
            result = freeze.write_bundle(self.fixture.evidence_path, output, confirmation_release=True)
        release_path = output / result["release_freeze"]
        link = self.root / "release-link.json"
        link.symlink_to(release_path)
        with self.assertRaisesRegex(freeze.FreezeError, "is a symlink"):
            freeze.validate_release_freeze(
                link, result["release_freeze_sha256"], expected_model="N3"
            )

    def test_deep_annotation_workflow_validation_is_mandatory(self) -> None:
        self.fixture.add_annotation_and_budget()
        with self.assertRaisesRegex(freeze.FreezeError, "annotation workflow deep validation failed"):
            freeze.write_bundle(
                self.fixture.evidence_path, self.root / "blocked", confirmation_release=True
            )


if __name__ == "__main__":
    unittest.main()
