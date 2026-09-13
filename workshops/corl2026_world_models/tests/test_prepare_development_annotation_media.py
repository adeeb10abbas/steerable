from __future__ import annotations

import copy
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


WORKSHOP = Path(__file__).resolve().parents[1]
MODULE = WORKSHOP / "analysis/prepare_development_annotation_media.py"
SPEC = importlib.util.spec_from_file_location(
    "prepare_development_annotation_media", MODULE
)
bridge = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(bridge)


def write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


class DevelopmentAnnotationMediaBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _alignment(self, model: str) -> dict:
        width, height = bridge.EXPECTED_IMAGE_DIMENSIONS[model]
        offset = 32 if model == "N3" else 6
        unsigned = {
            "contract_id": f"wmf1-{model.lower()}-alignment-v1",
            "model_id": model,
            "mapping_receipt_id": f"wmf1-development-{model.lower()}-physical-alignment-v1",
            "mapping_receipt_sha256": ("a" if model == "N3" else "b") * 64,
            "primary_horizon_s": offset / 15,
            "generated_frame_index": 32 if model == "N3" else 2,
            "target_executed_action_offset": offset,
            "control_step_s": 1 / 15,
            "captured_frame_interval_s": 1 / 15,
            "timestamp_tolerance_s": 1 / 30,
            "camera_id": "over_shoulder_left_camera",
            "camera_crop_id": bridge.EXPECTED_CROP_IDS[model],
            "camera_crop_sha256": ("c" if model == "N3" else "d") * 64,
            "image_width_px": width,
            "image_height_px": height,
            "early_horizon": {
                "horizon_s": 1 / 15 if model == "N3" else 3 / 15,
                "generated_frame_index": 1,
                "target_executed_action_offset": 1 if model == "N3" else 3,
            },
        }
        return {
            **unsigned,
            "contract_sha256": bridge.sha256_bytes(bridge.canonical_bytes(unsigned)),
        }

    def _mapping_fixture(
        self, model: str
    ) -> tuple[dict, Path, dict, dict, dict, dict]:
        alignment = self._alignment(model)
        timing_descriptor = {
            "path": str(self.root / f"{model.lower()}_timing.json"),
            "sha256": ("7" if model == "N3" else "8") * 64,
            "bytes": 1234,
        }
        target = {
            "generated_frame_index": alignment["generated_frame_index"],
            "target_physical_time_s": alignment["primary_horizon_s"],
            "status": "qualified",
            "target_executed_action_offset": alignment[
                "target_executed_action_offset"
            ],
            "eligible_request_count": 224,
            "full_prefix_request_count": 224,
            "max_camera_timestamp_residual_s": 0.001,
            "max_physics_timestamp_residual_s": 0.002,
        }
        early = {
            "generated_frame_index": alignment["early_horizon"][
                "generated_frame_index"
            ],
            "target_physical_time_s": alignment["early_horizon"]["horizon_s"],
            "status": "qualified",
            "target_executed_action_offset": alignment["early_horizon"][
                "target_executed_action_offset"
            ],
            "eligible_request_count": 240 if model == "N3" else 224,
            "full_prefix_request_count": 224,
            "max_camera_timestamp_residual_s": 0.001,
            "max_physics_timestamp_residual_s": 0.002,
        }
        boundary = {
            "generated_target_source": "signed_fixture",
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
                "generated_targets_scope": bridge.freeze.D1_GENERATED_TARGETS_SCOPE,
                "request_timing_coverage": bridge.EXPECTED_D1_COVERAGE,
                "incremental_standalone_decodes_assigned_target_times": False,
                "timing_unmapped_requests_eligible": False,
            })
        crop = {
            "camera_id": "over_shoulder_left_camera",
            "camera_crop_id": bridge.EXPECTED_CROP_IDS[model],
            "payload_sha256": alignment["camera_crop_sha256"],
            "crop_operation": bridge.EXPECTED_CROP_OPERATIONS[model],
        }
        expected_cells = sorted(
            cell_id
            for cell_id, row in bridge.annotation._planned_cells(
                "development", "full_two_model"
            ).items()
            if row[0] == model
        )
        if model == "N3":
            mapping_rows = []
            for index in range(1, 33):
                row = copy.deepcopy(target)
                row.update({
                    "generated_frame_index": index,
                    "target_physical_time_s": index / 15,
                    "target_executed_action_offset": index,
                    "eligible_request_count": 240 if index <= 2 else 224,
                })
                mapping_rows.append(row)
            early = mapping_rows[0]
            target = mapping_rows[-1]
        else:
            mapping_rows = [early, target]
        mapping = bridge.sign_document({
            "schema_version": bridge.MAPPING_SCHEMA,
            "receipt_id": alignment["mapping_receipt_id"],
            "study_id": bridge.STUDY_ID,
            "status": "qualified_from_complete_development_native_timing",
            "model_id": model,
            "development_cell_ids": expected_cells,
            "development_cell_count": 16,
            "development_request_count": bridge.EXPECTED_REQUESTS[model],
            "request_semantics": {
                key: bridge.freeze.MODEL_LIMITS[model][key]
                for key in (
                    "returned_action_horizon",
                    "unchanged_executed_prefix_horizon",
                    "action_space", "seed_semantics", "temporal_context",
                )
            },
            "timing_claim_boundary": boundary,
            "camera": {
                "camera_id": crop["camera_id"],
                "camera_crop_id": crop["camera_crop_id"],
                "camera_crop_sha256": crop["payload_sha256"],
                "image_width_px": bridge.EXPECTED_IMAGE_DIMENSIONS[model][0],
                "image_height_px": bridge.EXPECTED_IMAGE_DIMENSIONS[model][1],
                "crop_operation": crop["crop_operation"],
            },
            "measured_clock_intervals": {
                "control_step_s_min": 1 / 15,
                "control_step_s_max": 1 / 15,
                "captured_frame_interval_s_min": 1 / 15,
                "captured_frame_interval_s_max": 1 / 15,
                "timestamp_tolerance_s": alignment["timestamp_tolerance_s"],
                "tolerance_rule": "signed fixture",
            },
            "frame_to_physical_time": mapping_rows,
            "primary_target": target,
            "early_target": early,
            "source_receipts": {"generated_target_timing": timing_descriptor},
        })
        timing_sidecar = {
            "generated_targets": [
                {
                    "generated_frame_index": row["generated_frame_index"],
                    "target_physical_time_s": row["target_physical_time_s"],
                    "native_runtime_field": (
                        "request_timing_sidecar.generated_targets"
                    ),
                }
                for row in mapping_rows
            ],
            "request_timing_bindings": [],
        }
        requests_per_cell = 15 if model == "N3" else 57
        prefix = 32 if model == "N3" else 8
        for _cell in range(16):
            for request_index in range(requests_per_cell):
                executed = 2 if request_index == requests_per_cell - 1 else prefix
                applies = model == "N3" or request_index % 4 == 0
                targets = []
                if applies:
                    for row in mapping_rows:
                        offset = row["target_executed_action_offset"]
                        matched = offset <= executed
                        target_binding = {
                            "generated_frame_index": row["generated_frame_index"],
                            "executed_control_boundary": offset,
                            "authority_target_physical_time_s": row[
                                "target_physical_time_s"
                            ],
                            "status": (
                                "matched_native_request_clocks" if matched
                                else "not_executed_in_truncated_prefix"
                            ),
                        }
                        if matched:
                            target_binding.update({
                                "camera_authority_residual_s": 0.001,
                                "physics_authority_residual_s": 0.002,
                            })
                        targets.append(target_binding)
                binding = {
                    "executed_actions": executed,
                    "target_bindings": targets,
                }
                if model == "D1":
                    binding["decoded_output_timing"] = {
                        "source_timing_mapping_applies": applies,
                    }
                timing_sidecar["request_timing_bindings"].append(binding)
        mapping_path = write_json(self.root / f"{model.lower()}_mapping.json", mapping)
        alignment_unsigned = dict(alignment)
        alignment_unsigned.pop("contract_sha256")
        alignment_unsigned["mapping_receipt_sha256"] = bridge.sha256_file(mapping_path)
        alignment = {
            **alignment_unsigned,
            "contract_sha256": bridge.sha256_bytes(
                bridge.canonical_bytes(alignment_unsigned)
            ),
        }
        return (
            mapping, mapping_path, alignment, crop, timing_descriptor,
            timing_sidecar,
        )

    def _recording_artifacts(
        self, *, cell_id: str, recording_id: str, model: str,
        layout: str, condition: str, request_count: int,
    ) -> tuple[dict, dict]:
        source_video_sha = hashlib.sha256((cell_id + ":video").encode()).hexdigest()
        chunk = 32 if model == "N3" else 8
        actions = [
            {
                "action_index": action_index,
                "request_index": action_index // chunk,
                "executed_action_sha256": hashlib.sha256(
                    f"{cell_id}:action:{action_index}".encode()
                ).hexdigest(),
                "control_timestamp": "2026-09-13T00:00:00Z",
                "physics_step_id": f"{cell_id}:physics:{action_index}",
                "camera_frame_id": f"{cell_id}:camera:{action_index}",
            }
            for action_index in range(450)
        ]
        action = bridge.sign_document({
            "schema_version": bridge.annotation.ACTION_MANIFEST_SCHEMA,
            "study_id": bridge.STUDY_ID,
            "cell_id": cell_id,
            "recording_id": recording_id,
            "model_id": model,
            "executed_action_count": 450,
            "actions": actions,
        })
        action_path = write_json(self.root / "compiler" / model / cell_id / "action.json", action)
        action_descriptor = bridge.descriptor(action_path)
        recording = bridge.sign_document({
            "schema_version": bridge.annotation.RECORDING_RECEIPT_SCHEMA,
            "study_id": bridge.STUDY_ID,
            "receipt_id": "recording-" + hashlib.sha256(cell_id.encode()).hexdigest()[:16],
            "stage": "development",
            "cell_id": cell_id,
            "recording_id": recording_id,
            "model_id": model,
            "layout_pair_id": layout,
            "condition_id": condition,
            "recording_status": "valid_complete",
            "executed_action_count": 450,
            "censor_reason": None,
            "source_video_id": "video-" + hashlib.sha256(cell_id.encode()).hexdigest()[:16],
            "source_video_sha256": source_video_sha,
            "action_manifest_path": action_path.name,
            "action_manifest_sha256": action_descriptor["sha256"],
        })
        recording_path = write_json(action_path.parent / "recording.json", recording)
        return action_descriptor, bridge.descriptor(recording_path)

    def _full_inputs(self) -> tuple[dict, dict, dict, dict]:
        roster_by_model: dict[str, list[dict]] = {model: [] for model in bridge.MODELS}
        provenance_by_model: dict[str, list[dict]] = {model: [] for model in bridge.MODELS}
        timing_by_model: dict[str, dict] = {}
        alignment_by_model = {model: self._alignment(model) for model in bridge.MODELS}
        mapping_sha = {
            model: alignment_by_model[model]["mapping_receipt_sha256"]
            for model in bridge.MODELS
        }
        planned = bridge.annotation._planned_cells("development", "full_two_model")
        for model in bridge.MODELS:
            bindings = []
            request_count = 15 if model == "N3" else 57
            chunk = 32 if model == "N3" else 8
            for cell_id, (planned_model, layout, condition) in sorted(planned.items()):
                if planned_model != model:
                    continue
                recording_id = "episode-" + hashlib.sha256(cell_id.encode()).hexdigest()[:16]
                action_descriptor, recording_descriptor = self._recording_artifacts(
                    cell_id=cell_id,
                    recording_id=recording_id,
                    model=model,
                    layout=layout,
                    condition=condition,
                    request_count=request_count,
                )
                video_sha = hashlib.sha256((cell_id + ":video").encode()).hexdigest()
                video_id = "video-" + hashlib.sha256(cell_id.encode()).hexdigest()[:16]
                roster_by_model[model].append({
                    "cell_id": cell_id,
                    "recording_id": recording_id,
                    "model_id": model,
                    "layout_pair_id": layout,
                    "condition_id": condition,
                    "recording_status": "valid_complete",
                    "executed_action_count": 450,
                    "censor_reason": None,
                    "recording_receipt_path": recording_descriptor["path"],
                    "recording_receipt_sha256": recording_descriptor["sha256"],
                    "action_manifest_path": action_descriptor["path"],
                    "action_manifest_sha256": action_descriptor["sha256"],
                    "source_video_id": video_id,
                    "source_video_sha256": video_sha,
                })
                for request_index in range(request_count):
                    final = request_index == request_count - 1
                    executed = 2 if final else chunk
                    start = request_index * chunk
                    request_sha = hashlib.sha256(
                        f"{model}:{cell_id}:{request_index}:receipt".encode()
                    ).hexdigest()
                    response_sha = hashlib.sha256(
                        f"{model}:{cell_id}:{request_index}:response".encode()
                    ).hexdigest()
                    standard_descriptor = {
                        "path": f"/raw/{request_sha}.json",
                        "sha256": request_sha,
                        "bytes": 100,
                    }
                    completion_descriptor = {
                        "path": f"/raw/{cell_id}/completion.json",
                        "sha256": hashlib.sha256((cell_id + ":completion").encode()).hexdigest(),
                        "bytes": 101,
                    }
                    journal_descriptor = {
                        "path": f"/raw/{cell_id}/journal.jsonl",
                        "sha256": hashlib.sha256((cell_id + ":journal").encode()).hexdigest(),
                        "bytes": 102,
                        "event_count": 1000,
                        "tail_sha256": hashlib.sha256((cell_id + ":tail").encode()).hexdigest(),
                    }
                    current = {
                        "observation_id": f"obs_{start:06d}",
                        "control_step": start,
                        "physics_step": start,
                        "physics_time_s": start / 15,
                        "camera_frame_native_id": start,
                        "camera_frame_id": f"over_shoulder_left_camera:native-int:{start}",
                        "camera_capture_time_ns": start * 66_666_667,
                        "camera_timestamp_source": "synthetic_test_clock",
                        "payload_sha256": hashlib.sha256(f"{cell_id}:obs:{start}".encode()).hexdigest(),
                        "payload_artifact": {
                            "path": f"/raw/{cell_id}/obs.npz",
                            "sha256": hashlib.sha256(f"{cell_id}:npz".encode()).hexdigest(),
                            "bytes": 103,
                        },
                    }
                    preceding = None if request_index == 0 else {
                        **current,
                        "observation_id": f"obs_{start - 1:06d}",
                        "control_step": start - 1,
                        "physics_step": start - 1,
                        "physics_time_s": (start - 1) / 15,
                        "camera_frame_native_id": start - 1,
                        "camera_frame_id": f"over_shoulder_left_camera:native-int:{start - 1}",
                        "camera_capture_time_ns": (start - 1) * 66_666_667,
                    }
                    source = {
                        "source_request_id": "request_" + request_sha[:32],
                        "cell_id": cell_id,
                        "recording_id": recording_id,
                        "model_id": model,
                        "layout_pair_id": layout,
                        "condition_id": condition,
                        "request_index": request_index,
                        "action_step_start": start,
                        "executed_prefix_actions": executed,
                        "current_observation_id": current["observation_id"],
                        "preceding_observation_id": None if preceding is None else preceding["observation_id"],
                        "history_mode": "persistence_at_initial_request" if request_index == 0 else "preceding_observation",
                        "current_observation": current,
                        "preceding_observation": preceding,
                        "official_request_receipt": standard_descriptor,
                        "recorder_model_request": {},
                        "recorder_response_payload_sha256": response_sha,
                        "adapter_completion": completion_descriptor,
                        "adapter_journal": journal_descriptor,
                        "source_video_id": video_id,
                        "source_video": {"path": f"/raw/{cell_id}/video.mp4", "sha256": video_sha, "bytes": 104},
                        "model_output_or_action_modified": False,
                        "model_identity": {},
                        "model_context": {},
                        "action_manifest": action_descriptor,
                        "recording_receipt": recording_descriptor,
                    }
                    provenance_by_model[model].append(source)
                    applies = model == "N3" or request_index % 4 == 0
                    primary_offset = alignment_by_model[model]["target_executed_action_offset"]
                    early = alignment_by_model[model]["early_horizon"]
                    targets = []
                    if applies:
                        for frame_index, offset, horizon in (
                            (
                                alignment_by_model[model]["generated_frame_index"],
                                primary_offset,
                                alignment_by_model[model]["primary_horizon_s"],
                            ),
                            (
                                early["generated_frame_index"],
                                early["target_executed_action_offset"],
                                early["horizon_s"],
                            ),
                        ):
                            matched = offset <= executed
                            target = {
                                "generated_frame_index": frame_index,
                                "executed_control_boundary": offset,
                                "authority_target_physical_time_s": horizon,
                                "status": "matched_native_request_clocks" if matched else "not_executed_in_truncated_prefix",
                                "target_observation_id": f"obs_{start + offset:06d}" if matched else None,
                            }
                            if matched:
                                target.update({
                                    "camera_authority_residual_s": 0.001,
                                    "physics_authority_residual_s": 0.002,
                                    "target_camera_frame_id": start + offset,
                                    "target_camera_capture_time_ns": (start + offset) * 66_666_667,
                                    "target_physics_step": start + offset,
                                    "target_physics_time_s": (start + offset) / 15,
                                })
                            targets.append(target)
                    binding = {
                        "cell_id": cell_id,
                        "request_index": request_index,
                        "source_request_receipt": standard_descriptor,
                        "adapter_completion": completion_descriptor,
                        "adapter_journal": journal_descriptor,
                        "action_step_start": start,
                        "executed_actions": executed,
                        "request_current_observation_id": current["observation_id"],
                        "recorder_response_request_binding": {"payload_sha256": response_sha},
                        "camera_id": "over_shoulder_left_camera",
                        "target_bindings": targets,
                        "model_output_or_action_modified": False,
                    }
                    if model == "D1":
                        binding.update({
                            "decoded_output_timing": {
                                "source_timing_mapping_applies": applies,
                                "source_timing_status": (
                                    "source_proven_full_conditioning_origin" if applies
                                    else "unmapped_incremental_standalone_decode"
                                ),
                            },
                            "timing_eligible_target_count": sum(
                                row["status"] == "matched_native_request_clocks" for row in targets
                            ),
                            "eligible_for_timed_target_sampling": any(
                                row["status"] == "matched_native_request_clocks" for row in targets
                            ),
                        })
                    bindings.append(binding)
            timing_by_model[model] = {"request_timing_bindings": bindings}
        return roster_by_model, provenance_by_model, timing_by_model, alignment_by_model, mapping_sha

    def _inventory_and_selection(self):
        roster, provenance, sidecars, alignments, mapping_sha = self._full_inputs()
        inventory, rich = bridge._build_request_inventory(
            roster_by_model=roster,
            provenance_by_model=provenance,
            timing_by_model=sidecars,
            alignment_by_model=alignments,
            mapping_sha_by_model=mapping_sha,
            provenance_base_by_model={model: self.root for model in bridge.MODELS},
            inventory_finalized_at="2026-09-13T13:00:00Z",
        )
        payload = bridge.pretty_bytes(inventory)
        selection = bridge.annotation.select_requests(
            inventory,
            seed=bridge.annotation.REQUEST_SAMPLING_SEED,
            cap_per_episode=bridge.annotation.REQUEST_SAMPLE_CAP,
            inventory_sha256=bridge.sha256_bytes(payload),
        )
        return inventory, selection, rich, sidecars

    def test_exact_full_inventory_missingness_and_draw(self) -> None:
        inventory, selection, _, _ = self._inventory_and_selection()
        bridge._assert_inventory_and_selection(inventory, selection)
        self.assertEqual(len(inventory["episode_roster"]), 32)
        self.assertEqual(len(inventory["requests"]), 1152)
        unavailable = [
            row for row in inventory["requests"]
            if row["technical_invalid_reason"] == "forecast_timing_unavailable"
        ]
        self.assertEqual(len(unavailable), 672)
        self.assertTrue(all(row["timestamp_error_s"] is None for row in unavailable))
        self.assertEqual(selection["counts"]["eligible"], 448)
        self.assertEqual(selection["counts"]["selected"], 128)
        self.assertEqual(
            {row["eligible_request_inclusion_probability_exact"]
             for row in selection["requests"] if row["timing_camera_action_eligible"]},
            {"4/14"},
        )
        self.assertTrue(all(row["source"]["early_horizon_supported"] for row in selection["requests"]))

    def test_cross_request_join_is_rejected(self) -> None:
        roster, provenance, sidecars, alignments, mapping_sha = self._full_inputs()
        sidecars["N3"]["request_timing_bindings"][0]["cell_id"] = "cross-cell"
        with self.assertRaisesRegex(bridge.AnnotationMediaBridgeError, "lacks timing binding"):
            bridge._build_request_inventory(
                roster_by_model=roster,
                provenance_by_model=provenance,
                timing_by_model=sidecars,
                alignment_by_model=alignments,
                mapping_sha_by_model=mapping_sha,
                provenance_base_by_model={model: self.root for model in bridge.MODELS},
                inventory_finalized_at="2026-09-13T13:00:00Z",
            )

    def test_selection_must_cover_every_inventory_request_and_episode_once(self) -> None:
        inventory, selection, _, _ = self._inventory_and_selection()
        incomplete = copy.deepcopy(selection)
        incomplete.pop("payload_sha256")
        incomplete["requests"] = [
            row for row in incomplete["requests"]
            if row["timing_camera_action_eligible"]
        ]
        incomplete["episodes"] = []
        incomplete = bridge.sign_document(incomplete)
        with self.assertRaisesRegex(
            bridge.AnnotationMediaBridgeError, "exactly 1,152 wrappers"
        ):
            bridge._assert_inventory_and_selection(inventory, incomplete)

        duplicated = copy.deepcopy(selection)
        duplicated.pop("payload_sha256")
        duplicated["requests"][-1] = copy.deepcopy(duplicated["requests"][0])
        duplicated = bridge.sign_document(duplicated)
        with self.assertRaisesRegex(
            bridge.AnnotationMediaBridgeError, "source coverage changed"
        ):
            bridge._assert_inventory_and_selection(inventory, duplicated)

        missing_episodes = copy.deepcopy(selection)
        missing_episodes.pop("payload_sha256")
        missing_episodes["episodes"] = []
        missing_episodes = bridge.sign_document(missing_episodes)
        with self.assertRaisesRegex(
            bridge.AnnotationMediaBridgeError, "exactly 32 episode rows"
        ):
            bridge._assert_inventory_and_selection(inventory, missing_episodes)

        alternate = copy.deepcopy(selection)
        alternate.pop("payload_sha256")
        first_cell = alternate["episodes"][0]["cell_id"]
        selected_row = next(
            row for row in alternate["requests"]
            if row["source"]["cell_id"] == first_cell
            and row["timing_camera_action_eligible"] and row["selected"]
        )
        unselected_row = next(
            row for row in alternate["requests"]
            if row["source"]["cell_id"] == first_cell
            and row["timing_camera_action_eligible"] and not row["selected"]
        )
        selected_row["selected"] = False
        unselected_row["selected"] = True
        alternate = bridge.sign_document(alternate)
        with self.assertRaisesRegex(
            bridge.AnnotationMediaBridgeError, "exact frozen deterministic draw"
        ):
            bridge._assert_inventory_and_selection(inventory, alternate)

    def test_unmapped_d1_request_cannot_acquire_target_binding(self) -> None:
        roster, provenance, sidecars, alignments, mapping_sha = self._full_inputs()
        binding = next(
            row for row in sidecars["D1"]["request_timing_bindings"]
            if row["request_index"] == 1
        )
        binding["target_bindings"] = [{"fabricated": True}]
        with self.assertRaisesRegex(bridge.AnnotationMediaBridgeError, "acquired a target"):
            bridge._build_request_inventory(
                roster_by_model=roster,
                provenance_by_model=provenance,
                timing_by_model=sidecars,
                alignment_by_model=alignments,
                mapping_sha_by_model=mapping_sha,
                provenance_base_by_model={model: self.root for model in bridge.MODELS},
                inventory_finalized_at="2026-09-13T13:00:00Z",
            )

    def test_missing_native_residual_is_rejected(self) -> None:
        roster, provenance, sidecars, alignments, mapping_sha = self._full_inputs()
        target = sidecars["N3"]["request_timing_bindings"][0]["target_bindings"][0]
        target.pop("camera_authority_residual_s")
        with self.assertRaisesRegex(bridge.AnnotationMediaBridgeError, "residual is invalid"):
            bridge._build_request_inventory(
                roster_by_model=roster,
                provenance_by_model=provenance,
                timing_by_model=sidecars,
                alignment_by_model=alignments,
                mapping_sha_by_model=mapping_sha,
                provenance_base_by_model={model: self.root for model in bridge.MODELS},
                inventory_finalized_at="2026-09-13T13:00:00Z",
            )

    def test_mapping_requires_two_exact_qualified_224_request_targets(self) -> None:
        n3_mapping, n3_path, n3_alignment, n3_crop, n3_timing, n3_sidecar = (
            self._mapping_fixture("N3")
        )
        bridge._validate_mapping_and_alignment(
            model="N3", mapping=n3_mapping, mapping_path=n3_path,
            alignment=n3_alignment, crop=n3_crop, timing_descriptor=n3_timing,
            timing_sidecar=n3_sidecar,
            authoritative_mapping=n3_mapping,
            authoritative_alignment=n3_alignment,
        )
        self.assertEqual(n3_mapping["early_target"]["eligible_request_count"], 240)
        (
            mapping, path, alignment, crop, timing_descriptor, timing_sidecar,
        ) = self._mapping_fixture("D1")
        bridge._validate_mapping_and_alignment(
            model="D1", mapping=mapping, mapping_path=path,
            alignment=alignment, crop=crop, timing_descriptor=timing_descriptor,
            timing_sidecar=timing_sidecar,
            authoritative_mapping=mapping,
            authoritative_alignment=alignment,
        )
        authoritative_mapping = copy.deepcopy(mapping)
        authoritative_alignment = copy.deepcopy(alignment)
        for field, value in (("status", "unsupported"),
                             ("eligible_request_count", 223),
                             ("full_prefix_request_count", 223)):
            changed = copy.deepcopy(mapping)
            changed.pop("payload_sha256")
            changed["primary_target"][field] = value
            changed["frame_to_physical_time"][-1][field] = value
            changed = bridge.sign_document(changed)
            changed_path = write_json(self.root / f"bad_{field}.json", changed)
            changed_alignment = copy.deepcopy(alignment)
            changed_alignment.pop("contract_sha256")
            changed_alignment["mapping_receipt_sha256"] = bridge.sha256_file(
                changed_path
            )
            changed_alignment["contract_sha256"] = bridge.sha256_bytes(
                bridge.canonical_bytes(changed_alignment)
            )
            with self.assertRaisesRegex(
                bridge.AnnotationMediaBridgeError,
                "does not match exact qualified coverage",
            ):
                bridge._validate_mapping_and_alignment(
                    model="D1", mapping=changed, mapping_path=changed_path,
                    alignment=changed_alignment, crop=crop,
                    timing_descriptor=timing_descriptor,
                    timing_sidecar=timing_sidecar,
                    authoritative_mapping=authoritative_mapping,
                    authoritative_alignment=authoritative_alignment,
                )

    def test_mapping_requires_exact_d1_missingness_boundary(self) -> None:
        (
            mapping, _, alignment, crop, timing_descriptor, timing_sidecar,
        ) = self._mapping_fixture("D1")
        authoritative_mapping = copy.deepcopy(mapping)
        authoritative_alignment = copy.deepcopy(alignment)
        changed = copy.deepcopy(mapping)
        changed.pop("payload_sha256")
        changed["timing_claim_boundary"]["timing_unmapped_requests_eligible"] = True
        changed = bridge.sign_document(changed)
        path = write_json(self.root / "bad_missingness_mapping.json", changed)
        alignment.pop("contract_sha256")
        alignment["mapping_receipt_sha256"] = bridge.sha256_file(path)
        alignment["contract_sha256"] = bridge.sha256_bytes(
            bridge.canonical_bytes(alignment)
        )
        with self.assertRaisesRegex(
            bridge.AnnotationMediaBridgeError, "missingness boundary changed"
        ):
            bridge._validate_mapping_and_alignment(
                model="D1", mapping=changed, mapping_path=path,
                alignment=alignment, crop=crop, timing_descriptor=timing_descriptor,
                timing_sidecar=timing_sidecar,
                authoritative_mapping=authoritative_mapping,
                authoritative_alignment=authoritative_alignment,
            )

    def test_mapping_rejects_extra_science_label_or_confirmation_authority(self) -> None:
        (
            mapping, _, alignment, crop, timing_descriptor, timing_sidecar,
        ) = self._mapping_fixture("N3")
        authoritative_mapping = copy.deepcopy(mapping)
        authoritative_alignment = copy.deepcopy(alignment)
        changed = copy.deepcopy(mapping)
        changed.pop("payload_sha256")
        changed.update({
            "labels_created": 9,
            "science_counts": {"robot_episodes": 1},
            "confirmation_released": True,
        })
        changed = bridge.sign_document(changed)
        path = write_json(self.root / "mapping_with_extra_authority.json", changed)
        alignment.pop("contract_sha256")
        alignment["mapping_receipt_sha256"] = bridge.sha256_file(path)
        alignment["contract_sha256"] = bridge.sha256_bytes(
            bridge.canonical_bytes(alignment)
        )
        with self.assertRaisesRegex(
            bridge.AnnotationMediaBridgeError,
            "physical alignment receipt fields changed",
        ):
            bridge._validate_mapping_and_alignment(
                model="N3", mapping=changed, mapping_path=path,
                alignment=alignment, crop=crop, timing_descriptor=timing_descriptor,
                timing_sidecar=timing_sidecar,
                authoritative_mapping=authoritative_mapping,
                authoritative_alignment=authoritative_alignment,
            )

    def test_mapping_requires_frozen_semantics_and_canonical_target_choice(self) -> None:
        (
            mapping, _, alignment, crop, timing_descriptor, timing_sidecar,
        ) = self._mapping_fixture("N3")
        authoritative_mapping = copy.deepcopy(mapping)
        authoritative_alignment = copy.deepcopy(alignment)
        cases = []
        bad_semantics = copy.deepcopy(mapping)
        bad_semantics.pop("payload_sha256")
        bad_semantics["request_semantics"]["returned_action_horizon"] = 31
        cases.append((bad_semantics, "request semantics changed"))
        noncanonical = copy.deepcopy(mapping)
        noncanonical.pop("payload_sha256")
        noncanonical["primary_target"] = copy.deepcopy(
            noncanonical["frame_to_physical_time"][-2]
        )
        cases.append((noncanonical, "primary mapping target identity changed"))
        timing_rewrite = copy.deepcopy(mapping)
        timing_rewrite.pop("payload_sha256")
        timing_rewrite["frame_to_physical_time"][15][
            "target_physical_time_s"
        ] = 0.123
        cases.append((timing_rewrite, "differs from timing sidecar"))
        nonprimary_row_rewrite = copy.deepcopy(mapping)
        nonprimary_row_rewrite.pop("payload_sha256")
        nonprimary_row_rewrite["frame_to_physical_time"][10].update({
            "eligible_request_count": 999999,
            "full_prefix_request_count": -7,
            "max_camera_timestamp_residual_s": 999,
            "max_physics_timestamp_residual_s": -1,
        })
        cases.append((
            nonprimary_row_rewrite,
            "target row coverage/residuals changed",
        ))
        for ordinal, (unsigned, message) in enumerate(cases):
            changed = bridge.sign_document(unsigned)
            path = write_json(self.root / f"semantic_{ordinal}.json", changed)
            changed_alignment = copy.deepcopy(alignment)
            changed_alignment.pop("contract_sha256")
            changed_alignment["mapping_receipt_sha256"] = bridge.sha256_file(path)
            changed_alignment["contract_sha256"] = bridge.sha256_bytes(
                bridge.canonical_bytes(changed_alignment)
            )
            with self.assertRaisesRegex(bridge.AnnotationMediaBridgeError, message):
                bridge._validate_mapping_and_alignment(
                    model="N3", mapping=changed, mapping_path=path,
                    alignment=changed_alignment, crop=crop,
                    timing_descriptor=timing_descriptor,
                    timing_sidecar=timing_sidecar,
                    authoritative_mapping=authoritative_mapping,
                    authoritative_alignment=authoritative_alignment,
                )

    def test_mapping_rejects_rebound_ids_clocks_and_source_lineage(self) -> None:
        (
            mapping, _, alignment, crop, timing_descriptor, timing_sidecar,
        ) = self._mapping_fixture("N3")
        authoritative_mapping = copy.deepcopy(mapping)
        authoritative_alignment = copy.deepcopy(alignment)
        changed = copy.deepcopy(mapping)
        changed.pop("payload_sha256")
        changed["receipt_id"] = "invented-mapping-id"
        changed["timing_claim_boundary"]["generated_target_source"] = "invented"
        changed["measured_clock_intervals"].update({
            "control_step_s_min": 99,
            "control_step_s_max": -1,
            "captured_frame_interval_s_min": 88,
            "captured_frame_interval_s_max": -2,
            "timestamp_tolerance_s": 777,
            "tolerance_rule": "invented",
        })
        changed["source_receipts"] = {
            "development_cell_receipts": [],
            "adapter_completions": [],
            "adapter_journals": [],
            "official_request_receipt_sha256s": [],
            "generated_target_timing": timing_descriptor,
            "resource_receipts": [],
            "camera_crop_contract": {
                "path": str(self.root / "invented-camera.json"),
                "sha256": "9" * 64,
                "bytes": 99,
            },
        }
        changed = bridge.sign_document(changed)
        path = write_json(self.root / "rebound_physical_mapping.json", changed)
        changed_alignment = copy.deepcopy(alignment)
        changed_alignment.pop("contract_sha256")
        changed_alignment.update({
            "contract_id": "invented-alignment-id",
            "mapping_receipt_id": "invented-mapping-id",
            "mapping_receipt_sha256": bridge.sha256_file(path),
            "control_step_s": 99,
            "captured_frame_interval_s": 88,
            "timestamp_tolerance_s": 777,
        })
        changed_alignment["contract_sha256"] = bridge.sha256_bytes(
            bridge.canonical_bytes(changed_alignment)
        )
        with self.assertRaisesRegex(
            bridge.AnnotationMediaBridgeError, "authoritative source replay"
        ):
            bridge._validate_mapping_and_alignment(
                model="N3", mapping=changed, mapping_path=path,
                alignment=changed_alignment, crop=crop,
                timing_descriptor=timing_descriptor,
                timing_sidecar=timing_sidecar,
                authoritative_mapping=authoritative_mapping,
                authoritative_alignment=authoritative_alignment,
            )

    def test_authoritative_alignment_is_rederived_with_zero_resource_authority(self) -> None:
        mapping, _, alignment, _, timing_descriptor, _ = self._mapping_fixture("N3")
        unsigned = copy.deepcopy(alignment)
        unsigned.pop("contract_sha256")
        unsigned["mapping_receipt_sha256"] = None
        fragment = {"development_cells": [{"fixture": True}]}
        crop_descriptor = {
            "path": str(self.root / "crop.json"),
            "sha256": "c" * 64,
            "bytes": 123,
        }
        with mock.patch.object(
            bridge.freeze,
            "_derive_model_alignment",
            return_value=(mapping, unsigned, {}),
        ) as derive:
            observed_mapping, observed_alignment = (
                bridge._rederive_authoritative_physical_alignment(
                    model="N3",
                    fragment=fragment,
                    timing_descriptor=timing_descriptor,
                    crop_descriptor=crop_descriptor,
                    evidence_base=self.root,
                )
            )
        self.assertEqual(observed_mapping, mapping)
        expected_mapping_sha = bridge.sha256_bytes(bridge.pretty_bytes(mapping))
        self.assertEqual(
            observed_alignment["mapping_receipt_sha256"], expected_mapping_sha
        )
        expected_unsigned = dict(observed_alignment)
        observed_contract_sha = expected_unsigned.pop("contract_sha256")
        self.assertEqual(
            observed_contract_sha,
            bridge.sha256_bytes(bridge.canonical_bytes(expected_unsigned)),
        )
        self.assertIs(derive.call_args.kwargs["require_resources"], False)
        self.assertEqual(
            derive.call_args.args[0]["development_cells"],
            fragment["development_cells"],
        )

    def test_zero_science_receipts_require_exact_schema_not_empty_or_extra(self) -> None:
        cases = (
            bridge.EXPECTED_COMPILER_JOB_SCIENCE_COUNTS,
            bridge.EXPECTED_TIMING_JOB_SCIENCE_COUNTS,
        )
        for expected in cases:
            bridge._zero_science({"science_counts": expected}, "fixture", expected)
            for changed in ({}, {**expected, "unknown_activity": 0}):
                with self.assertRaisesRegex(
                    bridge.AnnotationMediaBridgeError, "exact zero-activity schema"
                ):
                    bridge._zero_science(
                        {"science_counts": changed}, "fixture", expected
                    )
        bridge._exact_compiler_science_activity(
            bridge.EXPECTED_COMPILER_SCIENCE_ACTIVITY
        )
        for changed in (
            {},
            {**bridge.EXPECTED_COMPILER_SCIENCE_ACTIVITY, "unknown_activity": 0},
        ):
            with self.assertRaisesRegex(
                bridge.AnnotationMediaBridgeError, "exact zero schema"
            ):
                bridge._exact_compiler_science_activity(changed)

    def test_png_encoder_is_metadata_free_and_workflow_compatible(self) -> None:
        image = np.arange(10 * 12 * 3, dtype=np.uint8).reshape(10, 12, 3)
        path = self.root / "image.png"
        path.write_bytes(bridge.png_bytes(image))
        self.assertEqual(bridge.annotation._png_dimensions(path), (12, 10))
        payload = path.read_bytes()
        self.assertNotIn(b"tEXt", payload)
        self.assertNotIn(b"iTXt", payload)
        self.assertNotIn(b"zTXt", payload)

    def test_unknown_original_camera_transform_fails_closed(self) -> None:
        crop = {
            "original_camera_replay": {
                "source_camera_shape": [720, 1280, 3],
                "output_shape": [168, 320, 3],
                "transform_chain": [{"operation": "invented_resize"}],
            }
        }
        with self.assertRaisesRegex(
            bridge.AnnotationMediaBridgeError, "exact original-camera replay failed"
        ):
            bridge._apply_original_transform(
                np.zeros((720, 1280, 3), dtype=np.uint8), crop, "N3",
                replay_client=mock.Mock(
                    replay=mock.Mock(side_effect=RuntimeError("invented transform"))
                ),
            )

    def test_original_and_generated_pixels_use_exact_camera_witness_api(self) -> None:
        original = np.zeros((720, 1280, 3), dtype=np.uint8)
        expected_original = np.full((168, 320, 3), 7, dtype=np.uint8)
        generated = np.zeros((33, 528, 640, 3), dtype=np.uint8)
        expected_generated = np.full((33, 168, 320, 3), 9, dtype=np.uint8)
        chain = [{"operation": "signed_exact_chain"}]
        crop = {
            "original_camera_replay": {
                "output_shape": [168, 320, 3],
                "transform_chain": chain,
            },
            "generated_decoded_crop": {"output_shape": [33, 168, 320, 3]},
        }
        with (
            mock.patch.object(
                bridge.camera_replay, "validate_camera_crop_contract"
            ) as validate,
            mock.patch.object(
                bridge.camera_replay,
                "extract_generated_crop",
                return_value=expected_generated,
            ) as extract,
        ):
            replay_client = mock.Mock()
            replay_evidence = {
                "session_id": "a" * 64,
                "ordinal": 0,
                "exact_signed_python_subprocess": True,
            }
            replay_client.replay.return_value = (expected_original, replay_evidence)
            observed_original, observed_chain, observed_replay = bridge._apply_original_transform(
                original, crop, "N3", replay_client=replay_client
            )
            observed_generated = bridge._apply_generated_crop(generated, crop, "N3")
        self.assertTrue(np.array_equal(observed_original, expected_original))
        self.assertEqual(observed_chain, chain)
        self.assertEqual(observed_replay, replay_evidence)
        self.assertTrue(np.array_equal(observed_generated, expected_generated))
        self.assertEqual(validate.call_count, 2)
        self.assertTrue(np.array_equal(replay_client.replay.call_args.args[0], original))
        self.assertIs(extract.call_args.args[0], generated)
        self.assertIs(extract.call_args.args[1], crop)

    def test_worker_command_uses_signed_interpreter_not_current_python(self) -> None:
        executable = self.root / "signed-d1-python"
        executable.write_bytes(b"#!/bin/sh\nexit 99\n")
        executable.chmod(0o755)
        crop = {
            "payload_sha256": "d" * 64,
            "runtime_dependencies": {
                "python": bridge.descriptor(executable.resolve()),
            },
        }
        contract = write_json(self.root / "d1_crop.json", crop)
        command, identity = bridge._replay_worker_command(
            model="D1", crop=crop, contract_path=contract
        )
        self.assertEqual(command[0], str(executable.resolve()))
        self.assertEqual(identity, crop["runtime_dependencies"]["python"])
        self.assertNotEqual(command[0], str(Path(sys.executable).resolve()))

    def test_worker_rejects_wrong_active_interpreter_before_camera_replay(self) -> None:
        executable = self.root / "signed-d1-python"
        executable.write_bytes(b"not the active interpreter")
        executable.chmod(0o755)
        crop = {
            "payload_sha256": "e" * 64,
            "runtime_dependencies": {
                "python": bridge.descriptor(executable.resolve()),
            },
        }
        contract = write_json(self.root / "d1_crop.json", crop)
        args = mock.Mock(
            model="D1",
            crop_contract=contract,
            crop_contract_sha256=bridge.sha256_file(contract),
            crop_contract_bytes=contract.stat().st_size,
            bridge_sha256=bridge.sha256_file(Path(bridge.__file__).resolve()),
            camera_replay_sha256=bridge.sha256_file(
                bridge.CAMERA_REPLAY_PATH.resolve()
            ),
        )
        with (
            mock.patch.object(bridge, "_validate_crop_contract", return_value=crop),
            mock.patch.object(
                bridge.camera_replay, "replay_original_camera_frame"
            ) as replay,
            self.assertRaisesRegex(
                bridge.AnnotationMediaBridgeError, "wrong interpreter"
            ),
        ):
            bridge._run_camera_replay_worker(args)
        replay.assert_not_called()

    def test_ipc_body_corruption_cannot_be_accepted(self) -> None:
        stream = io.BytesIO()
        bridge._write_ipc_frame(stream, {
            "schema_version": bridge.REPLAY_IPC_SCHEMA,
            "message_type": "replay_response",
        }, b"exact pixels")
        value = bytearray(stream.getvalue())
        value[-1] ^= 1
        with self.assertRaisesRegex(
            bridge.AnnotationMediaBridgeError, "payload hash changed"
        ):
            bridge._read_ipc_frame(
                io.BytesIO(value), maximum_payload_bytes=100,
                timeout_s=None, label="corrupted child response",
            )

    def test_publish_tranches_deduplicate_and_never_overlap(self) -> None:
        media = self.root / "annotation_media"
        media.mkdir()
        (self.root / "source_extraction_lineage.json").write_text("{}\n")
        (self.root / "rendered_png_manifest.json").write_text("{}\n")
        image_rows = []
        lineage_rows = []
        colors = [1, 1, 2, 3]
        for ordinal, color in enumerate(colors):
            image_id = f"source_{ordinal}"
            relative = Path("annotation_media") / f"{image_id}.png"
            payload = bridge.png_bytes(
                np.full((1, 1, 3), color, dtype=np.uint8)
            )
            (self.root / relative).write_bytes(payload)
            digest = bridge.sha256_bytes(payload)
            image_rows.append({
                "source_image_id": image_id,
                "source_request_id": f"request_{ordinal}",
                "image_role": "current",
                "annotation_media_path": str(relative),
                "annotation_media_sha256": digest,
                "width_px": 1,
                "height_px": 1,
                "camera_id": "over_shoulder_left_camera",
                "camera_crop_id": bridge.EXPECTED_CROP_IDS["N3"],
                "camera_crop_sha256": "c" * 64,
                "alignment_receipt_id": "alignment",
                "alignment_receipt_sha256": "a" * 64,
                "render_receipt_id": f"render_{ordinal}",
                "render_receipt_sha256": hashlib.sha256(
                    f"render:{ordinal}".encode()
                ).hexdigest(),
            })
            lineage_rows.append({
                "source_image_id": image_id,
                "join_key": {
                    "model_id": "N3",
                    "cell_id": f"cell_{ordinal}",
                    "request_index": ordinal,
                },
            })
        with mock.patch.object(
            bridge, "PUBLISH_TRANCHE_ASSET_BUDGET_BYTES", 100
        ):
            first = bridge._build_publish_tranche_index(
                temp_root=self.root,
                image_rows=image_rows,
                lineage_rows=lineage_rows,
            )
            second = bridge._build_publish_tranche_index(
                temp_root=self.root,
                image_rows=image_rows,
                lineage_rows=lineage_rows,
            )
        self.assertEqual(first, second)
        self.assertEqual(first["unique_asset_count"], 3)
        self.assertEqual(first["render_alias_count"], 4)
        all_digests = [
            asset["asset_sha256"]
            for tranche in first["tranches"]
            for asset in tranche["assets"]
        ]
        self.assertEqual(len(all_digests), len(set(all_digests)))
        self.assertEqual(sum(
            len(asset["aliases"])
            for tranche in first["tranches"]
            for asset in tranche["assets"]
        ), 4)

    def test_pixel_review_checklist_is_unsigned_human_template(self) -> None:
        media = self.root / "annotation_media"
        media.mkdir()
        path = media / "source_deadbeef.png"
        payload = bridge.png_bytes(np.zeros((2, 3, 3), dtype=np.uint8))
        path.write_bytes(payload)
        digest = bridge.sha256_bytes(payload)
        checklist = bridge._build_pixel_review_checklist(
            temp_root=self.root,
            image_rows=[{
                "annotation_media_path": str(path.relative_to(self.root)),
                "annotation_media_sha256": digest,
                "width_px": 3,
                "height_px": 2,
            }],
            rendered_png_manifest_sha256="f" * 64,
        )
        bridge.verify_signed(checklist, "test checklist")
        template = checklist["review_receipt_template_unsigned"]
        self.assertEqual(
            checklist["status"], "awaiting_named_human_visual_inspection"
        )
        self.assertIsNone(template["reviewer_code"])
        self.assertIsNone(template["reviewed_at"])
        self.assertIsNone(template["payload_sha256"])
        self.assertTrue(all(
            value is None for value in template["content_exclusions"].values()
        ))
        with self.assertRaises(Exception):
            bridge.annotation._validate_pixel_blindness_receipt(
                template,
                receipt_path=path,
                stage="development",
                artifact_scope="annotation_media",
                expected_assets=[(digest, 3, 2)],
            )

    def test_descriptor_rejects_symlink_ancestor(self) -> None:
        raw = self.root / "raw"
        actual = raw / "actual"
        actual.mkdir(parents=True)
        path = actual / "receipt.json"
        path.write_text("{}\n", encoding="utf-8")
        link = raw / "link"
        link.symlink_to(actual, target_is_directory=True)
        value = {
            "path": str(link / path.name),
            "sha256": bridge.sha256_file(path),
            "bytes": path.stat().st_size,
        }
        with self.assertRaisesRegex(bridge.AnnotationMediaBridgeError, "symlink"):
            bridge._verify_descriptor(
                value, base=raw, raw_root=raw, label="adversarial descriptor"
            )

    def test_prepare_requires_canonical_raw_root_and_manifest_scope(self) -> None:
        raw = self.root / "raw"
        raw.mkdir()
        value = {
            "schema_version": bridge.INPUT_SCHEMA,
            "study_id": bridge.STUDY_ID,
            "mode": "formal_full_development",
            "raw_root": str(raw),
            "compiler": {},
            "models": [],
        }
        inside = write_json(raw / "input.json", value)
        with self.assertRaisesRegex(
            bridge.AnnotationMediaBridgeError, "canonical task PVC root"
        ):
            bridge.prepare(inside, bridge.sha256_file(inside), raw / "output")

        outside = write_json(self.root / "outside.json", value)
        with mock.patch.object(bridge, "CANONICAL_RAW_ROOT", raw.resolve()):
            with self.assertRaisesRegex(
                bridge.AnnotationMediaBridgeError, "input manifest escapes raw root"
            ):
                bridge.prepare(
                    outside, bridge.sha256_file(outside), raw / "output"
                )

    def test_pre_review_state_cannot_be_mistaken_for_image_inventory(self) -> None:
        value = bridge.sign_document({
            "schema_version": bridge.PRE_REVIEW_SCHEMA,
            "status": "pending_human_pixel_blindness_review",
            "safe_for_rater_distribution": False,
        })
        self.assertNotEqual(value["schema_version"], bridge.IMAGE_INVENTORY_SCHEMA)
        self.assertIs(value["safe_for_rater_distribution"], False)


if __name__ == "__main__":
    unittest.main()
