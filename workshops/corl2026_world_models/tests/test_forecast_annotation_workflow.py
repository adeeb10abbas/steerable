import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

from jsonschema import Draft202012Validator, FormatChecker


MODULE = Path(__file__).resolve().parents[1] / "analysis/forecast_annotation_workflow.py"
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "experiments/forecast_layout/annotation_workflow.schema.json"
SPEC = importlib.util.spec_from_file_location("forecast_annotation_workflow", MODULE)
workflow = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(workflow)


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def png_bytes(width, height, rgb):
    def chunk(kind, payload):
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    rows = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def png_with_text_metadata(payload):
    end = payload.rfind(b"IEND") - 4
    text = b"source_path\x00/data/N3/original_left.png"
    kind = b"tEXt"
    chunk = (
        struct.pack(">I", len(text))
        + kind
        + text
        + struct.pack(">I", zlib.crc32(kind + text) & 0xFFFFFFFF)
    )
    return payload[:end] + chunk + payload[end:]


class AnnotationWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(cls.schema)
        cls.schema_validator = Draft202012Validator(cls.schema, format_checker=FormatChecker())

    def assertSchemaValid(self, document, label):
        errors = sorted(self.schema_validator.iter_errors(document), key=lambda error: list(error.path))
        detail = "\n".join(f"{error.json_path}: {error.message}" for error in errors[:5])
        self.assertFalse(errors, f"{label} failed Draft 2020-12 validation:\n{detail}")

    def write_recording_artifacts(
        self,
        *,
        cell_id,
        recording_id,
        model_id,
        layout_pair_id,
        condition_id,
        recording_status,
        executed_action_count,
        censor_reason,
        source_video_id,
        source_video_sha256,
        stage="development",
    ):
        artifact_directory = self.root / "recording_artifacts"
        artifact_directory.mkdir(exist_ok=True)
        chunk = 32 if model_id == "N3" else 8
        actions = []
        for action_index in range(executed_action_count):
            actions.append(
                {
                    "action_index": action_index,
                    "request_index": action_index // chunk,
                    "executed_action_sha256": hashlib.sha256(f"{cell_id}:action:{action_index}".encode()).hexdigest(),
                    "control_timestamp": f"2026-09-13T00:00:00.{action_index:06d}Z",
                    "physics_step_id": f"{cell_id}:physics:{action_index}",
                    "camera_frame_id": f"{cell_id}:camera:{action_index}",
                }
            )
        manifest = workflow.sign_document(
            {
                "schema_version": workflow.ACTION_MANIFEST_SCHEMA,
                "study_id": workflow.STUDY_ID,
                "cell_id": cell_id,
                "recording_id": recording_id,
                "model_id": model_id,
                "executed_action_count": executed_action_count,
                "actions": actions,
            }
        )
        token = hashlib.sha256(cell_id.encode()).hexdigest()[:16]
        action_path = artifact_directory / f"actions_{token}.json"
        json_write(action_path, manifest)
        receipt = workflow.sign_document(
            {
                "schema_version": workflow.RECORDING_RECEIPT_SCHEMA,
                "study_id": workflow.STUDY_ID,
                "receipt_id": f"recording-receipt-{token}",
                "stage": stage,
                "cell_id": cell_id,
                "recording_id": recording_id,
                "model_id": model_id,
                "layout_pair_id": layout_pair_id,
                "condition_id": condition_id,
                "recording_status": recording_status,
                "executed_action_count": executed_action_count,
                "censor_reason": censor_reason,
                "source_video_id": source_video_id,
                "source_video_sha256": source_video_sha256,
                "action_manifest_path": str(action_path.resolve()),
                "action_manifest_sha256": file_sha(action_path),
            }
        )
        receipt_path = artifact_directory / f"recording_{token}.json"
        json_write(receipt_path, receipt)
        return {
            "recording_receipt_path": str(receipt_path.resolve()),
            "recording_receipt_sha256": file_sha(receipt_path),
            "action_manifest_path": str(action_path.resolve()),
            "action_manifest_sha256": file_sha(action_path),
        }

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.alignment_sha = "a" * 64
        self.video_sha = "b" * 64
        alignment_unsigned = {
            "contract_id": "n3-primary-alignment-v1",
            "model_id": "N3",
            "mapping_receipt_id": "n3-pilot-mapping-receipt",
            "mapping_receipt_sha256": "9" * 64,
            "primary_horizon_s": 8 / 15,
            "generated_frame_index": 3,
            "target_executed_action_offset": 8,
            "control_step_s": 1 / 15,
            "captured_frame_interval_s": 1 / 15,
            "timestamp_tolerance_s": 1 / 30,
            "camera_id": "over_shoulder_left_camera",
            "camera_crop_id": "primary-camera-crop-v1",
            "camera_crop_sha256": "8" * 64,
            "image_width_px": 12,
            "image_height_px": 10,
            "early_horizon": None,
        }
        self.alignment_contract = {
            **alignment_unsigned,
            "contract_sha256": workflow.sha256_bytes(workflow.canonical_bytes(alignment_unsigned)),
        }
        self.requests = []
        planned = workflow._planned_cells("development", "reduced_n3")
        valid_cells = list(sorted(planned.items()))[:5]
        self.roster = []
        valid_ids = {cell_id for cell_id, _ in valid_cells}
        for cell_id, (model_id, layout_pair_id, condition_id) in sorted(planned.items()):
            valid = cell_id in valid_ids
            recording_index = next(
                (index for index, (candidate, _) in enumerate(valid_cells) if candidate == cell_id),
                None,
            )
            roster_row = {
                    "cell_id": cell_id,
                    "recording_id": f"recording-{recording_index}" if valid else None,
                    "model_id": model_id,
                    "layout_pair_id": layout_pair_id,
                    "condition_id": condition_id,
                    "recording_status": "valid_complete" if valid else "not_run",
                    "executed_action_count": 450 if valid else None,
                    "censor_reason": None,
                    "source_video_id": f"video-{recording_index}" if valid else None,
                    "source_video_sha256": self.video_sha if valid else None,
                    "recording_receipt_path": None,
                    "recording_receipt_sha256": None,
                    "action_manifest_path": None,
                    "action_manifest_sha256": None,
                }
            if valid:
                roster_row.update(
                    self.write_recording_artifacts(
                        cell_id=cell_id,
                        recording_id=roster_row["recording_id"],
                        model_id=model_id,
                        layout_pair_id=layout_pair_id,
                        condition_id=condition_id,
                        recording_status="valid_complete",
                        executed_action_count=450,
                        censor_reason=None,
                        source_video_id=roster_row["source_video_id"],
                        source_video_sha256=self.video_sha,
                    )
                )
            self.roster.append(roster_row)
        for episode_index, (cell_id, (model_id, layout_pair_id, condition_id)) in enumerate(valid_cells):
            for request_index in range(15):
                self.requests.append(
                    {
                        "source_request_id": f"{cell_id}/request_{request_index}",
                        "cell_id": cell_id,
                        "source_video_id": f"video-{episode_index}",
                        "source_video_sha256": self.video_sha,
                        "model_id": model_id,
                        "layout_pair_id": layout_pair_id,
                        "condition_id": condition_id,
                        "episode_id": f"recording-{episode_index}",
                        "request_index": request_index,
                        "request_start_action_index": request_index * 32,
                        "action_manifest_sha256": self.roster[episode_index]["action_manifest_sha256"],
                        "camera_id": "over_shoulder_left_camera",
                        "camera_crop_id": self.alignment_contract["camera_crop_id"],
                        "camera_crop_sha256": self.alignment_contract["camera_crop_sha256"],
                        "alignment_contract_id": self.alignment_contract["contract_id"],
                        "alignment_contract_sha256": self.alignment_contract["contract_sha256"],
                        "technical_valid": True,
                        "technical_invalid_reason": None,
                        "camera_identity_match": True,
                        "target_within_executed_prefix": request_index < 14,
                        "executed_prefix_actions": 32 if request_index < 14 else 2,
                        "target_executed_action_offset": 8,
                        "generated_frame_index": 3,
                        "target_physical_time_s": 8 / 15,
                        "timestamp_error_s": 0.001 if request_index < 14 else None,
                        "timestamp_tolerance_s": self.alignment_contract["timestamp_tolerance_s"],
                        "history_mode": "persistence_at_initial_request" if request_index == 0 else "preceding_observation",
                        "early_horizon_supported": False,
                        "alignment_receipt_id": f"alignment-{episode_index}-{request_index}",
                        "alignment_receipt_sha256": self.alignment_sha,
                    }
                )
        self.inventory = {
            "schema_version": workflow.REQUEST_INVENTORY_SCHEMA,
            "study_id": workflow.STUDY_ID,
            "stage": "development",
            "cohort_branch": "reduced_n3",
            "inventory_complete": True,
            "inventory_finalized_at": "2026-09-13T00:30:00Z",
            "annotation_state": "not_started",
            "episode_roster": self.roster,
            "alignment_contracts": [self.alignment_contract],
            "requests": self.requests,
        }
        example_directory = self.root / "examples"
        example_directory.mkdir()
        self.example_image = example_directory / "example_001.png"
        self.example_image.write_bytes(png_bytes(12, 10, (30, 60, 90)))
        self.example_blindness = self.root / "illustrated_examples_blindness_receipt.json"
        example_blindness = workflow.sign_document(
            {
                "schema_version": workflow.PIXEL_BLINDNESS_RECEIPT_SCHEMA,
                "study_id": workflow.STUDY_ID,
                "stage": "development",
                "artifact_scope": "illustrated_examples",
                "status": "human_reviewed_source_blind",
                "reviewer_code": "pixel-reviewer",
                "reviewed_at": "2026-09-13T00:14:00Z",
                "review_method": "human_visual_inspection_of_rendered_pixels_and_presentation",
                "content_exclusions": {key: True for key in workflow.EXAMPLE_EXCLUSION_KEYS},
                "assets": [
                    {
                        "media_sha256": file_sha(self.example_image),
                        "width_px": 12,
                        "height_px": 10,
                    }
                ],
            }
        )
        json_write(self.example_blindness, example_blindness)
        self.examples = self.root / "illustrated_examples.json"
        example_manifest = workflow.sign_document(
            {
                "schema_version": workflow.EXAMPLE_MANIFEST_SCHEMA,
                "study_id": workflow.STUDY_ID,
                "status": "approved_source_free",
                "source_scope": "synthetic_or_model_blind_development_calibration",
                "reviewer_code": "example-reviewer",
                "reviewed_at": "2026-09-13T00:15:00Z",
                "content_exclusions": {key: True for key in workflow.EXAMPLE_EXCLUSION_KEYS},
                "pixel_blindness_receipt_path": self.example_blindness.name,
                "pixel_blindness_receipt_sha256": file_sha(self.example_blindness),
                "files": [
                    {
                        "file": "examples/example_001.png",
                        "sha256": file_sha(self.example_image),
                        "width_px": 12,
                        "height_px": 10,
                        "caption": "Visible projected centers and explicit unknown marking",
                        "purpose": "Landmark calibration",
                    }
                ],
            }
        )
        json_write(self.examples, example_manifest)
        self.rubric = {
            "schema_version": workflow.RUBRIC_SCHEMA,
            "study_id": workflow.STUDY_ID,
            "status": "frozen",
            "freeze_scope": "development_duplicate_validation_and_confirmation",
            "landmarks": {
                "cube_center": "center of the visible projected Rubik's-cube silhouette",
                "bowl_center": "center of the visible projected bowl silhouette",
                "bowl_width": "visible projected horizontal bowl width",
                "partial_occlusion": "apply the illustrated rule or mark unknown; never infer a hidden center",
                "identity_uncertainty": "mark unknown when object identity is uncertain",
            },
            "never_infer_hidden_centers": True,
            "ambiguity_codes": [
                {"code": "none", "description": "unambiguous and resolvable"},
                {"code": "object_missing", "description": "at least one object is not visible"},
                {"code": "identity_uncertain", "description": "at least one entity cannot be identified"},
            ],
            "source_guess_options": ["camera_observation", "generated_future", "unsure"],
            "illustrated_examples": {
                "manifest_path": self.examples.name,
                "manifest_sha256": file_sha(self.examples),
            },
        }
        self.rubric_path = self.root / "rubric.json"
        json_write(self.rubric_path, self.rubric)
        self.freeze = {
            "schema_version": workflow.FREEZE_SCHEMA,
            "study_id": workflow.STUDY_ID,
            "stage": "development",
            "status": "frozen_for_development_validation",
            "cohort_branch": "reduced_n3",
            "qualified_model_ids": ["N3"],
            "request_sampling": {
                "seed": workflow.REQUEST_SAMPLING_SEED,
                "cap_per_episode": workflow.REQUEST_SAMPLE_CAP,
                "algorithm": workflow.REQUEST_SAMPLING_ALGORITHM,
            },
            "packet_randomization": {
                "rater_seeds": {"rater_a": 81173, "rater_b": 91283},
                "adjudicator_seed": 101393,
                "distinct_rater_seed_values": [81173, 91283, 101393],
            },
            "rubric": {"path": self.rubric_path.name, "sha256": file_sha(self.rubric_path)},
            "qualified_alignment_contract_sha256_by_model": {
                "N3": self.alignment_contract["contract_sha256"]
            },
            "movement_resolution": {
                "status": "pending_duplicate_development_labels",
                "threshold_relative_image_diagonal": None,
                "development_summary_path": None,
                "development_summary_sha256": None,
            },
            "development_validation_decision": None,
            "development_final_consensus": None,
        }
        self.freeze_path = self.root / "development_freeze.json"
        json_write(self.freeze_path, self.freeze)

    def tearDown(self):
        self.temporary.cleanup()

    def make_selection(self):
        inventory_path = self.root / "request_inventory.json"
        json_write(inventory_path, self.inventory)
        selection = workflow.select_requests(
            self.inventory,
            inventory_sha256=file_sha(inventory_path),
        )
        selection_path = self.root / "selection.json"
        json_write(selection_path, selection)
        return selection, selection_path

    def make_images(self, selection, selection_path):
        stage = selection["stage"]
        media = self.root / f"annotation_media_{stage}"
        media.mkdir(exist_ok=True)
        raw = self.root / f"raw_images_{stage}"
        raw.mkdir(exist_ok=True)
        receipts = self.root / f"render_receipts_{stage}"
        receipts.mkdir(exist_ok=True)
        images = []
        color = 1
        for selected in (row for row in selection["requests"] if row["selected"]):
            request = selected["source"]
            roles = workflow._expected_roles(request)
            for role in sorted(roles):
                path = media / f"image_{color:04d}.png"
                path.write_bytes(png_bytes(12, 10, (color % 251, (color * 3) % 251, (color * 7) % 251)))
                digest = file_sha(path)
                source_path = raw / f"source_{color:04d}.png"
                source_path.write_bytes(path.read_bytes())
                row = {
                        "source_image_id": f"source-image-{color}",
                        "source_request_id": request["source_request_id"],
                        "source_video_id": request["source_video_id"],
                        "source_video_sha256": request["source_video_sha256"],
                        "image_role": role,
                        "source_image_path": str(source_path.relative_to(self.root)),
                        "source_image_sha256": file_sha(source_path),
                        "annotation_media_path": str(path.relative_to(self.root)),
                        "annotation_media_sha256": digest,
                        "width_px": 12,
                        "height_px": 10,
                        "camera_id": request["camera_id"],
                        "camera_crop_id": request["camera_crop_id"],
                        "camera_crop_sha256": request["camera_crop_sha256"],
                        "alignment_receipt_id": request["alignment_receipt_id"],
                        "alignment_receipt_sha256": request["alignment_receipt_sha256"],
                        "render_receipt_id": f"render-{color}",
                        "presentation_sanitized": True,
                        "contains_overlay": False,
                    }
                receipt = workflow.sign_document(
                    {
                        "schema_version": workflow.RENDER_RECEIPT_SCHEMA,
                        "study_id": workflow.STUDY_ID,
                        "receipt_id": row["render_receipt_id"],
                        "source_image_id": row["source_image_id"],
                        "source_request_id": row["source_request_id"],
                        "source_video_id": row["source_video_id"],
                        "source_video_sha256": row["source_video_sha256"],
                        "image_role": row["image_role"],
                        "source_image_path": str(source_path.resolve()),
                        "source_image_sha256": row["source_image_sha256"],
                        "annotation_media_path": str(path.resolve()),
                        "annotation_media_sha256": row["annotation_media_sha256"],
                        "width_px": row["width_px"],
                        "height_px": row["height_px"],
                        "camera_id": row["camera_id"],
                        "camera_crop_id": row["camera_crop_id"],
                        "camera_crop_sha256": row["camera_crop_sha256"],
                        "alignment_receipt_id": row["alignment_receipt_id"],
                        "alignment_receipt_sha256": row["alignment_receipt_sha256"],
                        "presentation_sanitized": True,
                        "contains_overlay": False,
                    }
                )
                receipt_path = receipts / f"render_{color:04d}.json"
                json_write(receipt_path, receipt)
                row["render_receipt_path"] = str(receipt_path.relative_to(self.root))
                row["render_receipt_sha256"] = file_sha(receipt_path)
                images.append(row)
                color += 1
        blindness_path = self.root / f"annotation_media_blindness_receipt_{stage}.json"
        unique_assets = sorted(
            {
                (row["annotation_media_sha256"], row["width_px"], row["height_px"])
                for row in images
            }
        )
        blindness = workflow.sign_document(
            {
                "schema_version": workflow.PIXEL_BLINDNESS_RECEIPT_SCHEMA,
                "study_id": workflow.STUDY_ID,
                "stage": stage,
                "artifact_scope": "annotation_media",
                "status": "human_reviewed_source_blind",
                "reviewer_code": "media-reviewer",
                "reviewed_at": "2026-09-13T00:45:00Z",
                "review_method": "human_visual_inspection_of_rendered_pixels_and_presentation",
                "content_exclusions": {key: True for key in workflow.EXAMPLE_EXCLUSION_KEYS},
                "assets": [
                    {"media_sha256": digest, "width_px": width, "height_px": height}
                    for digest, width, height in unique_assets
                ],
            }
        )
        json_write(blindness_path, blindness)
        image_inventory = {
            "schema_version": workflow.IMAGE_INVENTORY_SCHEMA,
            "study_id": workflow.STUDY_ID,
            "stage": stage,
            "selection_manifest_sha256": file_sha(selection_path),
            "pixel_blindness_receipt_path": str(blindness_path.relative_to(self.root)),
            "pixel_blindness_receipt_sha256": file_sha(blindness_path),
            "images": images,
        }
        path = self.root / "image_inventory.json"
        json_write(path, image_inventory)
        return image_inventory, path

    def make_packages(self):
        selection, selection_path = self.make_selection()
        _, image_path = self.make_images(selection, selection_path)
        packet_root = self.root / "rater_packets"
        restricted = self.root / "restricted" / "identity_map.json"
        result = workflow.package_packets(
            selection_path=selection_path,
            image_inventory_path=image_path,
            freeze_path=self.freeze_path,
            packet_root=packet_root,
            restricted_map_path=restricted,
        )
        return result, packet_root, restricted

    def complete_response(self, slot_directory, *, rater_code, offset):
        response_directory = self.root / f"responses_{slot_directory.parent.name}_{slot_directory.name}_{rater_code}"
        response_directory.mkdir()
        template_by_packet = {
            json.loads(path.read_text())["packet_id"]: path
            for path in slot_directory.glob("batch_*/response_template.json")
        }
        response_slot = json.loads(next(iter(template_by_packet.values())).read_text())["rater_slot"]
        expected_packet_root = (
            slot_directory if response_slot == "adjudicator" else slot_directory.parent
        )
        packet_order = None
        for mapping_path in (self.root / "restricted").glob("*.json"):
            candidate = json.loads(mapping_path.read_text())
            stream = candidate.get("packets", {}).get(response_slot)
            if (
                isinstance(stream, dict)
                and Path(candidate.get("packet_root", "")).resolve() == expected_packet_root.resolve()
            ):
                packet_order = [batch["packet_id"] for batch in stream["batches"]]
                break
        self.assertIsNotNone(packet_order, "could not recover frozen packet order for response fixture")
        for batch_index, packet_id in enumerate(packet_order):
            template_path = template_by_packet[packet_id]
            response = json.loads(template_path.read_text())
            response["rater_code"] = rater_code
            required_attestations = (
                workflow.ADJUDICATOR_ATTESTATIONS
                if response["rater_slot"] == "adjudicator"
                else workflow.RESPONSE_ATTESTATIONS
            )
            response["attestations"] = {key: True for key in required_attestations}
            start_hour = 1 + 2 * batch_index
            response["started_at"] = f"2026-09-13T{start_hour:02d}:00:00Z"
            response["completed_at"] = f"2026-09-13T{start_hour + 1:02d}:00:00Z"
            response["locked"] = True
            response["locked_at"] = f"2026-09-13T{start_hour + 1:02d}:00:01Z"
            for index, annotation in enumerate(response["annotations"]):
                annotation.update(
                    {
                        "cube_resolvability": "resolvable",
                        "bowl_resolvability": "resolvable",
                        "cube_identity": "rubiks_cube",
                        "bowl_identity": "bowl",
                        "cube_center_px": [8.0 + offset * ((index + batch_index) % 3), 6.0],
                        "bowl_center_px": [3.0, 4.0],
                        "bowl_width_px": 5.0,
                        "ambiguity_codes": ["none"],
                        "source_guess": "unsure",
                        "annotation_seconds": 7.5,
                        "notes": "",
                    }
                )
            json_write(response_directory / f"{response['packet_id']}.json", response)
        return response_directory

    def make_final_consensus(
        self,
        restricted_path,
        rater_a,
        rater_b,
        *,
        adjudicator_code="human-c",
        namespace="development",
    ):
        adjudication_packets = self.root / f"adjudication_packets_{namespace}"
        adjudication_map = self.root / "restricted" / f"adjudication_map_{namespace}.json"
        workflow.package_adjudication(
            restricted_map_path=restricted_path,
            rater_a_response_path=rater_a,
            rater_b_response_path=rater_b,
            packet_root=adjudication_packets,
            adjudication_map_path=adjudication_map,
        )
        adjudicator_response = self.complete_response(
            adjudication_packets,
            rater_code=adjudicator_code,
            offset=0.125,
        )
        consensus = workflow.merge_adjudication(
            adjudication_map_path=adjudication_map,
            adjudicator_response_path=adjudicator_response,
        )
        path = self.root / f"{namespace}_final_consensus.json"
        json_write(path, consensus)
        return consensus, path, adjudication_packets, adjudication_map, adjudicator_response

    def make_confirmation_inventory(self):
        planned = workflow._planned_cells("confirmation", "reduced_n3")
        valid_cell_id = sorted(planned)[0]
        roster = []
        valid_row = None
        for cell_id, (model_id, layout_pair_id, condition_id) in sorted(planned.items()):
            valid = cell_id == valid_cell_id
            row = {
                "cell_id": cell_id,
                "recording_id": "confirmation-recording-0" if valid else None,
                "model_id": model_id,
                "layout_pair_id": layout_pair_id,
                "condition_id": condition_id,
                "recording_status": "valid_complete" if valid else "not_run",
                "executed_action_count": 450 if valid else None,
                "censor_reason": None,
                "recording_receipt_path": None,
                "recording_receipt_sha256": None,
                "action_manifest_path": None,
                "action_manifest_sha256": None,
                "source_video_id": "confirmation-video-0" if valid else None,
                "source_video_sha256": self.video_sha if valid else None,
            }
            if valid:
                row.update(
                    self.write_recording_artifacts(
                        cell_id=cell_id,
                        recording_id=row["recording_id"],
                        model_id=model_id,
                        layout_pair_id=layout_pair_id,
                        condition_id=condition_id,
                        recording_status="valid_complete",
                        executed_action_count=450,
                        censor_reason=None,
                        source_video_id=row["source_video_id"],
                        source_video_sha256=row["source_video_sha256"],
                        stage="confirmation",
                    )
                )
                valid_row = row
            roster.append(row)
        assert valid_row is not None
        requests = []
        for request_index in range(15):
            request = copy.deepcopy(self.requests[0])
            request.update(
                {
                    "source_request_id": f"{valid_cell_id}/request_{request_index}",
                    "cell_id": valid_cell_id,
                    "source_video_id": valid_row["source_video_id"],
                    "source_video_sha256": valid_row["source_video_sha256"],
                    "model_id": valid_row["model_id"],
                    "layout_pair_id": valid_row["layout_pair_id"],
                    "condition_id": valid_row["condition_id"],
                    "episode_id": valid_row["recording_id"],
                    "request_index": request_index,
                    "request_start_action_index": request_index * 32,
                    "action_manifest_sha256": valid_row["action_manifest_sha256"],
                    "target_within_executed_prefix": request_index < 14,
                    "executed_prefix_actions": 32 if request_index < 14 else 2,
                    "timestamp_error_s": 0.001 if request_index < 14 else None,
                    "history_mode": (
                        "persistence_at_initial_request"
                        if request_index == 0
                        else "preceding_observation"
                    ),
                    "alignment_receipt_id": f"confirmation-alignment-{request_index}",
                }
            )
            requests.append(request)
        return {
            "schema_version": workflow.REQUEST_INVENTORY_SCHEMA,
            "study_id": workflow.STUDY_ID,
            "stage": "confirmation",
            "cohort_branch": "reduced_n3",
            "inventory_complete": True,
            "inventory_finalized_at": "2026-09-13T04:00:00Z",
            "annotation_state": "not_started",
            "episode_roster": roster,
            "alignment_contracts": [copy.deepcopy(self.alignment_contract)],
            "requests": requests,
        }

    def test_metadata_only_uniform_selection_is_stable_and_retains_zero_eligible(self):
        selection = workflow.select_requests(self.inventory)
        reversed_inventory = dict(self.inventory, requests=list(reversed(self.requests)))
        self.assertEqual(selection, workflow.select_requests(reversed_inventory))
        self.assertEqual(selection["counts"]["selected"], 20)
        self.assertEqual({row["selected"] for row in selection["requests"]}, {True, False})
        self.assertEqual(sum(row["selected_count"] == 4 for row in selection["episodes"]), 5)
        self.assertEqual(selection["counts"]["zero_eligible_episodes"], 11)

        zero = copy.deepcopy(self.inventory)
        for row in zero["requests"][:15]:
            row["camera_identity_match"] = False
        result = workflow.select_requests(zero)
        self.assertEqual(result["counts"]["zero_eligible_episodes"], 12)
        first = next(row for row in result["episodes"] if row["episode_id"] == "recording-0")
        self.assertIsNone(first["eligible_request_inclusion_probability"])

    def test_timestamp_residual_exists_exactly_when_forecast_timing_is_available(self):
        selection = workflow.select_requests(self.inventory)
        terminal = next(
            row for row in selection["requests"]
            if row["source"]["request_index"] == 14
        )
        self.assertIsNone(terminal["source"]["timestamp_error_s"])
        self.assertEqual(terminal["eligibility_reasons"], ["target_outside_executed_prefix"])
        self.assertFalse(terminal["timing_camera_action_eligible"])
        self.assertSchemaValid(self.inventory, "null residual outside executed prefix")

        invented = copy.deepcopy(self.inventory)
        outside = next(
            row for row in invented["requests"] if row["request_index"] == 14
        )
        outside["timestamp_error_s"] = 0.001
        with self.assertRaisesRegex(
            workflow.ContractError, "must be null when forecast timing is unavailable"
        ):
            workflow.select_requests(invented)
        self.assertFalse(self.schema_validator.is_valid(invented))

        missing = copy.deepcopy(self.inventory)
        within = next(
            row for row in missing["requests"] if row["request_index"] == 0
        )
        within["timestamp_error_s"] = None
        with self.assertRaisesRegex(workflow.ContractError, "timestamp_error_s must be finite"):
            workflow.select_requests(missing)
        self.assertFalse(self.schema_validator.is_valid(missing))

        timing_unavailable = copy.deepcopy(self.inventory)
        invalid_within = next(
            row for row in timing_unavailable["requests"] if row["request_index"] == 1
        )
        invalid_within.update(
            technical_valid=False,
            technical_invalid_reason="forecast_timing_unavailable",
            timestamp_error_s=None,
        )
        unavailable_selection = workflow.select_requests(timing_unavailable)
        unavailable = next(
            row for row in unavailable_selection["requests"]
            if row["source"]["source_request_id"] == invalid_within["source_request_id"]
        )
        self.assertTrue(unavailable["source"]["target_within_executed_prefix"])
        self.assertEqual(unavailable["eligibility_reasons"], ["technical_invalid"])
        self.assertFalse(unavailable["timing_camera_action_eligible"])
        self.assertSchemaValid(
            timing_unavailable,
            "null residual for a within-prefix timing-unavailable request",
        )

        invented_unavailable = copy.deepcopy(timing_unavailable)
        invalid_within_invented = next(
            row for row in invented_unavailable["requests"]
            if row["source_request_id"] == invalid_within["source_request_id"]
        )
        invalid_within_invented["timestamp_error_s"] = 0.001
        with self.assertRaisesRegex(
            workflow.ContractError, "must be null when forecast timing is unavailable"
        ):
            workflow.select_requests(invented_unavailable)
        self.assertFalse(self.schema_validator.is_valid(invented_unavailable))

    def test_selection_rejects_quality_or_visibility_fields_and_wrong_seed(self):
        inventory = copy.deepcopy(self.inventory)
        inventory["requests"][0]["object_visibility"] = True
        with self.assertRaisesRegex(workflow.ContractError, "disallowed keys"):
            workflow.select_requests(inventory)
        with self.assertRaisesRegex(workflow.ContractError, "seed"):
            workflow.select_requests(self.inventory, seed=7)

    def test_complete_and_censored_action_counts_and_request_bounds(self):
        bad_complete = copy.deepcopy(self.inventory)
        bad_complete["episode_roster"][0]["executed_action_count"] = 449
        with self.assertRaisesRegex(workflow.ContractError, "exactly 450"):
            workflow.select_requests(bad_complete)

        censored = copy.deepcopy(self.inventory)
        cell_id = self.roster[0]["cell_id"]
        roster_row = next(row for row in censored["episode_roster"] if row["cell_id"] == cell_id)
        roster_row.update(recording_status="valid_censored", executed_action_count=65, censor_reason="safety_abort")
        roster_row.update(
            self.write_recording_artifacts(
                cell_id=cell_id,
                recording_id=roster_row["recording_id"],
                model_id=roster_row["model_id"],
                layout_pair_id=roster_row["layout_pair_id"],
                condition_id=roster_row["condition_id"],
                recording_status="valid_censored",
                executed_action_count=65,
                censor_reason="safety_abort",
                source_video_id=roster_row["source_video_id"],
                source_video_sha256=roster_row["source_video_sha256"],
            )
        )
        original = [row for row in censored["requests"] if row["cell_id"] == cell_id]
        replacement = []
        for index, prefix in enumerate((32, 32, 1)):
            row = copy.deepcopy(original[index])
            row["request_index"] = index
            row["source_request_id"] = f"{cell_id}/censored_{index}"
            row["executed_prefix_actions"] = prefix
            row["request_start_action_index"] = sum((32, 32, 1)[:index])
            row["action_manifest_sha256"] = roster_row["action_manifest_sha256"]
            row["target_within_executed_prefix"] = prefix >= row["target_executed_action_offset"]
            row["timestamp_error_s"] = (
                0.001 if row["target_within_executed_prefix"] else None
            )
            replacement.append(row)
        censored["requests"] = [row for row in censored["requests"] if row["cell_id"] != cell_id] + replacement
        workflow.select_requests(censored)
        censored["episode_roster"][0]["executed_action_count"] = 64
        censored["episode_roster"][0].update(
            self.write_recording_artifacts(
                cell_id=cell_id,
                recording_id=roster_row["recording_id"],
                model_id="N3",
                layout_pair_id=roster_row["layout_pair_id"],
                condition_id=roster_row["condition_id"],
                recording_status="valid_censored",
                executed_action_count=64,
                censor_reason="safety_abort",
                source_video_id=roster_row["source_video_id"],
                source_video_sha256=roster_row["source_video_sha256"],
            )
        )
        new_action_hash = censored["episode_roster"][0]["action_manifest_sha256"]
        for row in censored["requests"]:
            if row["cell_id"] == cell_id:
                row["action_manifest_sha256"] = new_action_hash
        with self.assertRaisesRegex(workflow.ContractError, "requests, expected 2"):
            workflow.select_requests(censored)

    def test_recording_receipt_and_action_manifest_are_mandatory_and_cross_bound(self):
        missing_receipt = copy.deepcopy(self.inventory)
        missing_receipt["episode_roster"][0]["recording_receipt_sha256"] = "0" * 64
        with self.assertRaisesRegex(workflow.ContractError, "recording receipt file hash mismatch"):
            workflow.select_requests(missing_receipt)

        rebound = copy.deepcopy(self.inventory)
        roster_row = rebound["episode_roster"][0]
        action_path = Path(roster_row["action_manifest_path"])
        action_manifest = json.loads(action_path.read_text())
        action_manifest["actions"][0]["request_index"] = 1
        json_write(action_path, workflow.sign_document(action_manifest))
        action_sha = file_sha(action_path)

        receipt_path = Path(roster_row["recording_receipt_path"])
        receipt = json.loads(receipt_path.read_text())
        receipt["action_manifest_sha256"] = action_sha
        json_write(receipt_path, workflow.sign_document(receipt))
        roster_row["action_manifest_sha256"] = action_sha
        roster_row["recording_receipt_sha256"] = file_sha(receipt_path)
        for request in rebound["requests"]:
            if request["cell_id"] == roster_row["cell_id"]:
                request["action_manifest_sha256"] = action_sha
        with self.assertRaisesRegex(workflow.ContractError, "action manifest request ownership"):
            workflow.select_requests(rebound)

    def test_full_two_model_inventory_accepts_n3_and_d1_exact_request_counts(self):
        inventory = copy.deepcopy(self.inventory)
        inventory["cohort_branch"] = "full_two_model"
        planned = workflow._planned_cells("development", "full_two_model")
        chosen = {
            "N3": next(cell for cell, identity in sorted(planned.items()) if identity[0] == "N3"),
            "D1": next(cell for cell, identity in sorted(planned.items()) if identity[0] == "D1"),
        }
        d1_unsigned = {
            **{key: value for key, value in self.alignment_contract.items() if key != "contract_sha256"},
            "contract_id": "d1-primary-alignment-v1",
            "model_id": "D1",
            "mapping_receipt_id": "d1-pilot-mapping-receipt",
            "mapping_receipt_sha256": "7" * 64,
        }
        d1_contract = {**d1_unsigned, "contract_sha256": workflow.sha256_bytes(workflow.canonical_bytes(d1_unsigned))}
        contracts = {"N3": self.alignment_contract, "D1": d1_contract}
        inventory["alignment_contracts"] = [self.alignment_contract, d1_contract]
        inventory["episode_roster"] = []
        inventory["requests"] = []
        for cell_id, (model, layout, condition) in sorted(planned.items()):
            valid = cell_id == chosen[model]
            roster_row = {
                    "cell_id": cell_id,
                    "recording_id": f"full-{model}" if valid else None,
                    "model_id": model,
                    "layout_pair_id": layout,
                    "condition_id": condition,
                    "recording_status": "valid_complete" if valid else "not_run",
                    "executed_action_count": 450 if valid else None,
                    "censor_reason": None,
                    "source_video_id": f"full-video-{model}" if valid else None,
                    "source_video_sha256": self.video_sha if valid else None,
                    "recording_receipt_path": None,
                    "recording_receipt_sha256": None,
                    "action_manifest_path": None,
                    "action_manifest_sha256": None,
                }
            if valid:
                roster_row.update(
                    self.write_recording_artifacts(
                        cell_id=cell_id,
                        recording_id=roster_row["recording_id"],
                        model_id=model,
                        layout_pair_id=layout,
                        condition_id=condition,
                        recording_status="valid_complete",
                        executed_action_count=450,
                        censor_reason=None,
                        source_video_id=roster_row["source_video_id"],
                        source_video_sha256=self.video_sha,
                    )
                )
            inventory["episode_roster"].append(roster_row)
            if not valid:
                continue
            chunk = 32 if model == "N3" else 8
            count = 15 if model == "N3" else 57
            contract = contracts[model]
            for index in range(count):
                prefix = chunk if index < count - 1 else 450 - chunk * (count - 1)
                inventory["requests"].append(
                    {
                        **copy.deepcopy(self.requests[0]),
                        "source_request_id": f"{cell_id}/request_{index}",
                        "cell_id": cell_id,
                        "source_video_id": f"full-video-{model}",
                        "model_id": model,
                        "layout_pair_id": layout,
                        "condition_id": condition,
                        "episode_id": f"full-{model}",
                        "request_index": index,
                        "request_start_action_index": index * chunk,
                        "action_manifest_sha256": roster_row["action_manifest_sha256"],
                        "alignment_contract_id": contract["contract_id"],
                        "alignment_contract_sha256": contract["contract_sha256"],
                        "executed_prefix_actions": prefix,
                        "target_within_executed_prefix": prefix >= contract["target_executed_action_offset"],
                        "timestamp_error_s": (
                            0.001
                            if prefix >= contract["target_executed_action_offset"]
                            else None
                        ),
                        "history_mode": "persistence_at_initial_request" if index == 0 else "preceding_observation",
                    }
                )
        result = workflow.select_requests(inventory)
        self.assertEqual(result["counts"]["selected"], 8)
        self.assertEqual({row["source"]["model_id"] for row in result["requests"] if row["selected"]}, {"N3", "D1"})

    def test_development_freeze_requires_null_empirical_threshold(self):
        info = workflow.validate_freeze(self.freeze, freeze_path=self.freeze_path, stage="development")
        self.assertEqual(info["rubric_sha256"], file_sha(self.rubric_path))
        invented = copy.deepcopy(self.freeze)
        invented["movement_resolution"]["threshold_relative_image_diagonal"] = 0.01
        with self.assertRaisesRegex(workflow.ContractError, "must not invent"):
            workflow.validate_freeze(invented, freeze_path=self.freeze_path, stage="development")
        reused_seed = copy.deepcopy(self.freeze)
        reused_seed["packet_randomization"]["rater_seeds"]["rater_b"] = 81173
        reused_seed["packet_randomization"]["distinct_rater_seed_values"] = [81173, 81173, 101393]
        with self.assertRaisesRegex(workflow.ContractError, "seeds must be distinct"):
            workflow.validate_freeze(reused_seed, freeze_path=self.freeze_path, stage="development")

    def test_illustrated_example_manifest_rejects_outcome_claims_and_png_metadata(self):
        manifest = json.loads(self.examples.read_text())
        manifest["files"][0]["caption"] = "successful model output"
        json_write(self.examples, workflow.sign_document(manifest))
        self.rubric["illustrated_examples"]["manifest_sha256"] = file_sha(self.examples)
        json_write(self.rubric_path, self.rubric)
        self.freeze["rubric"]["sha256"] = file_sha(self.rubric_path)
        json_write(self.freeze_path, self.freeze)
        with self.assertRaisesRegex(workflow.ContractError, "model or condition identity"):
            workflow.validate_freeze(self.freeze, freeze_path=self.freeze_path, stage="development")

        self.example_image.write_bytes(png_with_text_metadata(png_bytes(12, 10, (30, 60, 90))))
        manifest["files"][0]["caption"] = "Visible projected centers"
        manifest["files"][0]["sha256"] = file_sha(self.example_image)
        json_write(self.examples, workflow.sign_document(manifest))
        self.rubric["illustrated_examples"]["manifest_sha256"] = file_sha(self.examples)
        json_write(self.rubric_path, self.rubric)
        self.freeze["rubric"]["sha256"] = file_sha(self.rubric_path)
        json_write(self.freeze_path, self.freeze)
        with self.assertRaisesRegex(workflow.ContractError, "disallowed metadata chunk"):
            workflow.validate_freeze(self.freeze, freeze_path=self.freeze_path, stage="development")

    def test_human_pixel_blindness_receipt_is_separate_hash_bound_and_fail_closed(self):
        selection, selection_path = self.make_selection()
        image_inventory, image_path = self.make_images(selection, selection_path)
        blindness_path = self.root / image_inventory["pixel_blindness_receipt_path"]
        blindness = json.loads(blindness_path.read_text())
        blindness["assets"][0]["width_px"] = float(blindness["assets"][0]["width_px"])
        json_write(blindness_path, workflow.sign_document(blindness))
        image_inventory["pixel_blindness_receipt_sha256"] = file_sha(blindness_path)
        json_write(image_path, image_inventory)
        with self.assertRaisesRegex(workflow.ContractError, "width_px must be a positive integer"):
            workflow.package_packets(
                selection_path=selection_path,
                image_inventory_path=image_path,
                freeze_path=self.freeze_path,
                packet_root=self.root / "float-dimension-packets",
                restricted_map_path=self.root / "float-dimension-restricted" / "map.json",
            )

        blindness["assets"][0]["width_px"] = int(blindness["assets"][0]["width_px"])
        blindness["content_exclusions"]["model_identity_absent"] = False
        json_write(blindness_path, workflow.sign_document(blindness))
        image_inventory["pixel_blindness_receipt_sha256"] = file_sha(blindness_path)
        json_write(image_path, image_inventory)
        with self.assertRaisesRegex(workflow.ContractError, "human pixel-blindness review found prohibited content"):
            workflow.package_packets(
                selection_path=selection_path,
                image_inventory_path=image_path,
                freeze_path=self.freeze_path,
                packet_root=self.root / "unreviewed-packets",
                restricted_map_path=self.root / "unreviewed-restricted" / "map.json",
            )

        example_manifest = json.loads(self.examples.read_text())
        example_manifest["pixel_blindness_receipt_sha256"] = "0" * 64
        json_write(self.examples, workflow.sign_document(example_manifest))
        self.rubric["illustrated_examples"]["manifest_sha256"] = file_sha(self.examples)
        json_write(self.rubric_path, self.rubric)
        self.freeze["rubric"]["sha256"] = file_sha(self.rubric_path)
        json_write(self.freeze_path, self.freeze)
        with self.assertRaisesRegex(workflow.ContractError, "pixel-blindness receipt hash mismatch"):
            workflow.validate_freeze(self.freeze, freeze_path=self.freeze_path, stage="development")

    def test_packets_hide_source_identity_and_separate_counterparts(self):
        result, packet_root, restricted_path = self.make_packages()
        self.assertEqual(result["rater_packets"], 2)
        restricted = json.loads(restricted_path.read_text())
        serialized_map = restricted_path.read_text()
        self.assertIn("source_video_id", serialized_map)
        self.assertIn("condition_id", serialized_map)
        self.assertEqual(os.stat(restricted_path).st_mode & 0o777, 0o600)

        reverse = {}
        for asset in restricted["assets"]:
            for slot, opaque in asset["rater_opaque_ids"].items():
                reverse[(slot, opaque)] = {row["source_request_id"] for row in asset["source_records"]}
        source_orders = []
        for slot in ("rater_a", "rater_b"):
            order = []
            for packet_path in sorted((packet_root / slot).glob("batch_*/packet.json")):
                packet = json.loads(packet_path.read_text())
                serialized = packet_path.read_text()
                for forbidden in ("model_id", "condition_id", "source_request", "source_video", "episode_id", "original_left", "N3"):
                    self.assertNotIn(forbidden, serialized)
                packet_rubric = json.loads((packet_path.parent / "rubric.json").read_text())
                self.assertEqual(packet_rubric["illustrated_examples"]["manifest_path"], "illustrated_examples.json")
                self.assertEqual(packet["rubric_sha256"], file_sha(packet_path.parent / "rubric.json"))
                ids = [item["opaque_image_id"] for item in packet["items"]]
                sources = [reverse[(slot, opaque)] for opaque in ids]
                flattened = [value for values in sources for value in values]
                self.assertEqual(len(flattened), len(set(flattened)))
                self.assertTrue(all(item["media_sha256"] == file_sha(packet_path.parent / item["media_file"]) for item in packet["items"]))
                order.extend(ids)
                self.assertTrue((packet_path.parent / "response_template.json").is_file())
            source_orders.append(order)
        self.assertNotEqual(source_orders[0], source_orders[1])

    def test_adjudicator_packets_are_source_free_media_bound_and_mechanically_required(self):
        _, packet_root, restricted_path = self.make_packages()
        rater_a = self.complete_response(packet_root / "rater_a", rater_code="human-a", offset=0.0)
        rater_b = self.complete_response(packet_root / "rater_b", rater_code="human-b", offset=0.25)
        consensus, _, adjudication_packets, adjudication_map, adjudicator_response = self.make_final_consensus(
            restricted_path,
            rater_a,
            rater_b,
            namespace="source-free",
        )
        self.assertGreater(consensus["counts"]["independently_adjudicated"], 0)
        for packet_path in sorted(adjudication_packets.glob("batch_*/packet.json")):
            packet_text = packet_path.read_text()
            for forbidden in (
                "model_id",
                "condition_id",
                "source_request_id",
                "source_video_id",
                "episode_id",
                "original_left",
                "reflected_right",
                "first_pass",
                "human-a",
                "human-b",
            ):
                self.assertNotIn(forbidden, packet_text)
            packet = json.loads(packet_text)
            self.assertEqual(packet["rater_slot"], "adjudicator")
            self.assertTrue(
                all(
                    item["media_sha256"] == file_sha(packet_path.parent / item["media_file"])
                    for item in packet["items"]
                )
            )

        first_packet = sorted(adjudication_packets.glob("batch_*/packet.json"))[0]
        first_item = json.loads(first_packet.read_text())["items"][0]
        media_path = first_packet.parent / first_item["media_file"]
        media_path.write_bytes(png_bytes(12, 10, (250, 1, 1)))
        with self.assertRaisesRegex(workflow.ContractError, "packet media hash mismatch"):
            workflow.merge_adjudication(
                adjudication_map_path=adjudication_map,
                adjudicator_response_path=adjudicator_response,
            )

    def test_development_summary_derives_q95_and_confirmation_binds_exact_receipt(self):
        _, packet_root, restricted_path = self.make_packages()
        rater_a = self.complete_response(packet_root / "rater_a", rater_code="human-a", offset=0.0)
        rater_b = self.complete_response(packet_root / "rater_b", rater_code="human-b", offset=0.25)
        summary = workflow.summarize_development_labels(
            restricted_map_path=restricted_path,
            rater_a_response_path=rater_a,
            rater_b_response_path=rater_b,
        )
        self.assertGreater(summary["eligible_duplicate_count"], 0)
        self.assertGreater(summary["movement_resolution_threshold_relative_image_diagonal"], 0)
        self.assertTrue(summary["adequacy_is_not_inferred_by_code"])
        self.assertIn("predicted", summary["by_image_role"])
        self.assertGreater(summary["by_image_role"]["executed"]["eligible_duplicate_count"], 0)
        summary_path = self.root / "development_summary.json"
        json_write(summary_path, summary)
        (
            _,
            consensus_path,
            adjudication_packets,
            adjudication_map,
            adjudicator_response,
        ) = self.make_final_consensus(restricted_path, rater_a, rater_b)

        confirmation = copy.deepcopy(self.freeze)
        confirmation["stage"] = "confirmation"
        confirmation["status"] = "frozen_for_confirmation"
        confirmation["movement_resolution"] = {
            "status": "frozen_from_duplicate_development_labels",
            "threshold_relative_image_diagonal": summary["movement_resolution_threshold_relative_image_diagonal"],
            "development_summary_path": summary_path.name,
            "development_summary_sha256": file_sha(summary_path),
        }
        confirmation["development_validation_decision"] = {
            "measurement_usable": True,
            "decided_by": "authorized-scientific-reviewer",
            "decided_at": "2026-09-13T03:00:00Z",
            "basis": "Development coverage and movement-to-noise scale were reviewed.",
            "development_summary_sha256": file_sha(summary_path),
        }
        confirmation["development_final_consensus"] = {
            "path": consensus_path.name,
            "sha256": file_sha(consensus_path),
        }
        confirmation_path = self.root / "confirmation_freeze.json"
        json_write(confirmation_path, confirmation)
        workflow.validate_freeze(confirmation, freeze_path=confirmation_path, stage="confirmation")

        missing_adjudication = copy.deepcopy(confirmation)
        missing_adjudication["development_final_consensus"] = None
        with self.assertRaisesRegex(workflow.ContractError, "lacks final adjudicated"):
            workflow.validate_freeze(missing_adjudication, freeze_path=confirmation_path, stage="confirmation")

        same_adjudicator_response = self.complete_response(
            adjudication_packets,
            rater_code="human-a",
            offset=0.125,
        )
        with self.assertRaisesRegex(workflow.ContractError, "adjudicator is not independent"):
            workflow.merge_adjudication(
                adjudication_map_path=adjudication_map,
                adjudicator_response_path=same_adjudicator_response,
            )

        incomplete_consensus = json.loads(consensus_path.read_text())
        incomplete_consensus["labels"].pop()
        incomplete_consensus = workflow.sign_document(incomplete_consensus)
        incomplete_path = self.root / "incomplete_consensus.json"
        json_write(incomplete_path, incomplete_consensus)
        incomplete_freeze = copy.deepcopy(confirmation)
        incomplete_freeze["development_final_consensus"] = {
            "path": incomplete_path.name,
            "sha256": file_sha(incomplete_path),
        }
        with self.assertRaisesRegex(workflow.ContractError, "does not reproduce"):
            workflow.validate_freeze(incomplete_freeze, freeze_path=confirmation_path, stage="confirmation")

        cross_cohort = copy.deepcopy(confirmation)
        cross_cohort["cohort_branch"] = "full_two_model"
        cross_cohort["qualified_model_ids"] = ["N3", "D1"]
        cross_cohort["qualified_alignment_contract_sha256_by_model"]["D1"] = "d" * 64
        with self.assertRaisesRegex(workflow.ContractError, "cohort branch differs from development"):
            workflow.validate_freeze(cross_cohort, freeze_path=confirmation_path, stage="confirmation")

        changed = copy.deepcopy(confirmation)
        changed["movement_resolution"]["threshold_relative_image_diagonal"] += 0.001
        with self.assertRaisesRegex(workflow.ContractError, "differs"):
            workflow.validate_freeze(changed, freeze_path=confirmation_path, stage="confirmation")
        unapproved = copy.deepcopy(confirmation)
        unapproved["development_validation_decision"]["measurement_usable"] = False
        with self.assertRaisesRegex(workflow.ContractError, "not approved"):
            workflow.validate_freeze(unapproved, freeze_path=confirmation_path, stage="confirmation")

    def test_adjudication_packet_response_and_merge_are_stage_generic_for_confirmation(self):
        _, development_packets, development_map = self.make_packages()
        development_a = self.complete_response(
            development_packets / "rater_a", rater_code="development-human-a", offset=0.0
        )
        development_b = self.complete_response(
            development_packets / "rater_b", rater_code="development-human-b", offset=0.25
        )
        summary = workflow.summarize_development_labels(
            restricted_map_path=development_map,
            rater_a_response_path=development_a,
            rater_b_response_path=development_b,
        )
        summary_path = self.root / "confirmation_prerequisite_summary.json"
        json_write(summary_path, summary)
        _, development_consensus_path, _, _, _ = self.make_final_consensus(
            development_map,
            development_a,
            development_b,
            adjudicator_code="development-human-c",
            namespace="confirmation-prerequisite",
        )
        confirmation_freeze = copy.deepcopy(self.freeze)
        confirmation_freeze.update(stage="confirmation", status="frozen_for_confirmation")
        confirmation_freeze["movement_resolution"] = {
            "status": "frozen_from_duplicate_development_labels",
            "threshold_relative_image_diagonal": summary[
                "movement_resolution_threshold_relative_image_diagonal"
            ],
            "development_summary_path": str(summary_path.resolve()),
            "development_summary_sha256": file_sha(summary_path),
        }
        confirmation_freeze["development_validation_decision"] = {
            "measurement_usable": True,
            "decided_by": "authorized-scientific-reviewer",
            "decided_at": "2026-09-13T03:30:00Z",
            "basis": "Development duplicate-label coverage and movement scale were reviewed.",
            "development_summary_sha256": file_sha(summary_path),
        }
        confirmation_freeze["development_final_consensus"] = {
            "path": str(development_consensus_path.resolve()),
            "sha256": file_sha(development_consensus_path),
        }
        confirmation_freeze_path = self.root / "confirmation_freeze_for_packets.json"
        json_write(confirmation_freeze_path, confirmation_freeze)

        confirmation_inventory = self.make_confirmation_inventory()
        confirmation_inventory_path = self.root / "confirmation_request_inventory.json"
        json_write(confirmation_inventory_path, confirmation_inventory)
        confirmation_selection = workflow.select_requests(
            confirmation_inventory,
            inventory_path=confirmation_inventory_path,
            inventory_sha256=file_sha(confirmation_inventory_path),
        )
        confirmation_selection_path = self.root / "confirmation_selection.json"
        json_write(confirmation_selection_path, confirmation_selection)
        _, confirmation_images_path = self.make_images(
            confirmation_selection,
            confirmation_selection_path,
        )
        confirmation_packets = self.root / "confirmation_packets"
        confirmation_map = self.root / "restricted" / "confirmation_identity_map.json"
        workflow.package_packets(
            selection_path=confirmation_selection_path,
            image_inventory_path=confirmation_images_path,
            freeze_path=confirmation_freeze_path,
            packet_root=confirmation_packets,
            restricted_map_path=confirmation_map,
        )
        confirmation_a = self.complete_response(
            confirmation_packets / "rater_a", rater_code="confirmation-human-a", offset=0.0
        )
        confirmation_b = self.complete_response(
            confirmation_packets / "rater_b", rater_code="confirmation-human-b", offset=0.25
        )
        adjudication_packets = self.root / "confirmation_adjudication_packets"
        adjudication_map = self.root / "restricted" / "confirmation_adjudication_map.json"
        package_result = workflow.package_adjudication(
            restricted_map_path=confirmation_map,
            rater_a_response_path=confirmation_a,
            rater_b_response_path=confirmation_b,
            packet_root=adjudication_packets,
            adjudication_map_path=adjudication_map,
        )
        adjudicator_response = self.complete_response(
            adjudication_packets,
            rater_code="confirmation-human-c",
            offset=0.125,
        )
        consensus = workflow.merge_adjudication(
            adjudication_map_path=adjudication_map,
            adjudicator_response_path=adjudicator_response,
        )
        self.assertEqual(package_result["stage"], "confirmation")
        self.assertEqual(consensus["stage"], "confirmation")
        self.assertEqual(consensus["counts"]["images"], package_result["first_pass_image_count"])
        self.assertSchemaValid(json.loads(adjudication_map.read_text()), "confirmation adjudication map")
        self.assertSchemaValid(consensus, "confirmation final consensus")

    def test_summary_rejects_same_rater_and_invented_unknown_coordinates(self):
        _, packet_root, restricted_path = self.make_packages()
        rater_a = self.complete_response(packet_root / "rater_a", rater_code="same-human", offset=0.0)
        rater_b = self.complete_response(packet_root / "rater_b", rater_code="same-human", offset=0.1)
        with self.assertRaisesRegex(workflow.ContractError, "same rater"):
            workflow.summarize_development_labels(
                restricted_map_path=restricted_path,
                rater_a_response_path=rater_a,
                rater_b_response_path=rater_b,
            )

        response_path = sorted(rater_b.glob("*.json"))[0]
        response = json.loads(response_path.read_text())
        response["rater_code"] = "Different Human"
        response["annotations"][0].update(
            {
                "cube_resolvability": "resolvable",
                "bowl_resolvability": "resolvable",
                "cube_identity": "rubiks_cube",
                "bowl_identity": "bowl",
                "ambiguity_codes": ["none"],
                "cube_center_px": [2.0, 3.0],
                "bowl_center_px": [2.0, 3.0],
                "bowl_width_px": 4.0,
            }
        )
        json_write(response_path, response)
        with self.assertRaisesRegex(workflow.ContractError, "canonical lowercase"):
            workflow.summarize_development_labels(
                restricted_map_path=restricted_path,
                rater_a_response_path=rater_a,
                rater_b_response_path=rater_b,
            )

        response = json.loads(response_path.read_text())
        response["rater_code"] = "different-human"
        response["started_at"] = "2026-09-13T03:00:00Z"
        response["completed_at"] = "2026-09-13T02:00:00Z"
        json_write(response_path, response)
        with self.assertRaisesRegex(workflow.ContractError, "timestamps are out of order"):
            workflow.summarize_development_labels(
                restricted_map_path=restricted_path,
                rater_a_response_path=rater_a,
                rater_b_response_path=rater_b,
            )
        response["started_at"] = "2026-09-13T01:00:00Z"
        response["completed_at"] = "2026-09-13T02:00:00Z"
        response["annotations"][0].update(
            {
                "cube_resolvability": "unknown",
                "bowl_resolvability": "resolvable",
                "cube_identity": "unknown",
                "bowl_identity": "bowl",
                "ambiguity_codes": ["object_missing"],
                "cube_center_px": [2.0, 3.0],
                "bowl_center_px": [2.0, 3.0],
                "bowl_width_px": 4.0,
            }
        )
        json_write(response_path, response)
        with self.assertRaisesRegex(workflow.ContractError, "unresolved cube must not have coordinates"):
            workflow.summarize_development_labels(
                restricted_map_path=restricted_path,
                rater_a_response_path=rater_a,
                rater_b_response_path=rater_b,
            )

    def test_response_sets_reject_impossible_time_and_same_rater_batch_overlap(self):
        _, packet_root, restricted_path = self.make_packages()
        rater_a = self.complete_response(
            packet_root / "rater_a", rater_code="timing-human-a", offset=0.0
        )
        rater_b = self.complete_response(
            packet_root / "rater_b", rater_code="timing-human-b", offset=0.25
        )
        mapping = json.loads(restricted_path.read_text())
        batches = mapping["packets"]["rater_a"]["batches"]
        first_path = rater_a / f"{batches[0]['packet_id']}.json"
        first = json.loads(first_path.read_text())
        first["annotations"][0]["annotation_seconds"] = 100000.0
        json_write(first_path, first)
        with self.assertRaisesRegex(workflow.ContractError, "session duration"):
            workflow.summarize_development_labels(
                restricted_map_path=restricted_path,
                rater_a_response_path=rater_a,
                rater_b_response_path=rater_b,
            )

        first["annotations"][0]["annotation_seconds"] = 7.5
        json_write(first_path, first)
        self.assertGreaterEqual(len(batches), 2)
        second_path = rater_a / f"{batches[1]['packet_id']}.json"
        second = json.loads(second_path.read_text())
        second["started_at"] = first["started_at"]
        second["completed_at"] = first["completed_at"]
        second["locked_at"] = first["locked_at"]
        json_write(second_path, second)
        with self.assertRaisesRegex(workflow.ContractError, "overlap or are out of packet order"):
            workflow.summarize_development_labels(
                restricted_map_path=restricted_path,
                rater_a_response_path=rater_a,
                rater_b_response_path=rater_b,
            )

    def test_exact_png_hash_dimensions_and_required_roles_fail_closed(self):
        selection, selection_path = self.make_selection()
        inventory, image_path = self.make_images(selection, selection_path)
        inventory["images"][0]["width_px"] = 11
        json_write(image_path, inventory)
        with self.assertRaisesRegex(workflow.ContractError, "dimensions"):
            workflow.package_packets(
                selection_path=selection_path,
                image_inventory_path=image_path,
                freeze_path=self.freeze_path,
                packet_root=self.root / "bad-packets",
                restricted_map_path=self.root / "bad-restricted" / "map.json",
            )

        inventory, image_path = self.make_images(selection, selection_path)
        inventory["images"].pop()
        json_write(image_path, inventory)
        with self.assertRaisesRegex(workflow.ContractError, "image roles"):
            workflow.package_packets(
                selection_path=selection_path,
                image_inventory_path=image_path,
                freeze_path=self.freeze_path,
                packet_root=self.root / "missing-role-packets",
                restricted_map_path=self.root / "missing-role-restricted" / "map.json",
            )

        inventory, image_path = self.make_images(selection, selection_path)
        media_path = self.root / inventory["images"][0]["annotation_media_path"]
        media_path.write_bytes(png_with_text_metadata(media_path.read_bytes()))
        inventory["images"][0]["annotation_media_sha256"] = file_sha(media_path)
        json_write(image_path, inventory)
        with self.assertRaisesRegex(workflow.ContractError, "disallowed metadata chunk"):
            workflow.package_packets(
                selection_path=selection_path,
                image_inventory_path=image_path,
                freeze_path=self.freeze_path,
                packet_root=self.root / "metadata-packets",
                restricted_map_path=self.root / "metadata-restricted" / "map.json",
            )

        inventory, image_path = self.make_images(selection, selection_path)
        inventory["images"][0]["source_image_path"] = inventory["images"][1]["source_image_path"]
        inventory["images"][0]["source_image_sha256"] = inventory["images"][1]["source_image_sha256"]
        json_write(image_path, inventory)
        with self.assertRaisesRegex(workflow.ContractError, "render receipt source"):
            workflow.package_packets(
                selection_path=selection_path,
                image_inventory_path=image_path,
                freeze_path=self.freeze_path,
                packet_root=self.root / "source-swap-packets",
                restricted_map_path=self.root / "source-swap-restricted" / "map.json",
            )

    def test_quantile_type7(self):
        self.assertEqual(workflow.quantile_type7([1.0], 0.95), 1.0)
        self.assertAlmostEqual(workflow.quantile_type7([0, 1, 2, 3, 4], 0.95), 3.8)

    def test_draft_2020_12_schema_validates_real_artifacts_and_rejects_validator_adversaries(self):
        self.assertSchemaValid(self.inventory, "request inventory")
        self.assertSchemaValid(self.freeze, "development freeze")
        self.assertSchemaValid(self.rubric, "rubric")
        self.assertSchemaValid(json.loads(self.examples.read_text()), "illustrated-example manifest")
        self.assertSchemaValid(json.loads(self.example_blindness.read_text()), "example blindness receipt")
        first_valid = next(row for row in self.roster if row["recording_status"] == "valid_complete")
        self.assertSchemaValid(
            json.loads(Path(first_valid["recording_receipt_path"]).read_text()),
            "recording receipt",
        )
        self.assertSchemaValid(
            json.loads(Path(first_valid["action_manifest_path"]).read_text()),
            "action manifest",
        )

        selection, selection_path = self.make_selection()
        self.assertSchemaValid(selection, "request selection")
        image_inventory, image_path = self.make_images(selection, selection_path)
        self.assertSchemaValid(image_inventory, "image inventory")
        blindness_path = self.root / image_inventory["pixel_blindness_receipt_path"]
        self.assertSchemaValid(json.loads(blindness_path.read_text()), "annotation-media blindness receipt")
        render_path = self.root / image_inventory["images"][0]["render_receipt_path"]
        self.assertSchemaValid(json.loads(render_path.read_text()), "render receipt")

        packet_root = self.root / "schema_packets"
        restricted_path = self.root / "restricted" / "schema_identity_map.json"
        workflow.package_packets(
            selection_path=selection_path,
            image_inventory_path=image_path,
            freeze_path=self.freeze_path,
            packet_root=packet_root,
            restricted_map_path=restricted_path,
        )
        self.assertSchemaValid(json.loads(restricted_path.read_text()), "restricted identity map")
        first_packet_path = sorted((packet_root / "rater_a").glob("batch_*/packet.json"))[0]
        self.assertSchemaValid(json.loads(first_packet_path.read_text()), "rater packet")
        rater_a = self.complete_response(packet_root / "rater_a", rater_code="schema-human-a", offset=0.0)
        rater_b = self.complete_response(packet_root / "rater_b", rater_code="schema-human-b", offset=0.25)
        first_response = json.loads(sorted(rater_a.glob("*.json"))[0].read_text())
        self.assertSchemaValid(first_response, "locked first-pass response")
        development_summary = workflow.summarize_development_labels(
            restricted_map_path=restricted_path,
            rater_a_response_path=rater_a,
            rater_b_response_path=rater_b,
        )
        self.assertSchemaValid(development_summary, "development label-noise summary")
        consensus, _, _, adjudication_map, adjudicator_response = self.make_final_consensus(
            restricted_path,
            rater_a,
            rater_b,
            adjudicator_code="schema-human-c",
            namespace="schema",
        )
        self.assertSchemaValid(json.loads(adjudication_map.read_text()), "adjudication map")
        self.assertSchemaValid(
            json.loads(sorted(adjudicator_response.glob("*.json"))[0].read_text()),
            "locked adjudicator response",
        )
        self.assertSchemaValid(consensus, "mechanically merged final consensus")

        invalid_reason = copy.deepcopy(self.inventory)
        invalid_reason["requests"][0].update(technical_valid=False, technical_invalid_reason=None)
        self.assertFalse(self.schema_validator.is_valid(invalid_reason))
        invalid_unknown = copy.deepcopy(first_response)
        invalid_unknown["annotations"][0].update(
            cube_resolvability="unknown",
            cube_identity="unknown",
            cube_center_px=[1.0, 1.0],
            ambiguity_codes=["object_missing"],
        )
        self.assertFalse(self.schema_validator.is_valid(invalid_unknown))
        invalid_seeds = copy.deepcopy(self.freeze)
        invalid_seeds["packet_randomization"]["distinct_rater_seed_values"] = [81173, 81173, 101393]
        self.assertFalse(self.schema_validator.is_valid(invalid_seeds))
        invalid_technical_roster = copy.deepcopy(self.inventory)
        invalid_technical_roster["episode_roster"][0].update(
            recording_status="technical_invalid",
            executed_action_count=0,
            censor_reason="request_transport_failure",
            source_video_id="known-video",
            source_video_sha256=None,
        )
        self.assertFalse(self.schema_validator.is_valid(invalid_technical_roster))
        invalid_selection = copy.deepcopy(selection)
        invalid_selection["selection_uses_object_visibility_or_forecast_quality"] = True
        self.assertFalse(self.schema_validator.is_valid(invalid_selection))
        invalid_restricted = json.loads(restricted_path.read_text())
        invalid_restricted["packets"]["rater_a"]["delivery_rule"] = "ALL_BATCHES_AT_ONCE"
        self.assertFalse(self.schema_validator.is_valid(invalid_restricted))
        invalid_summary = copy.deepcopy(development_summary)
        invalid_summary["adequacy_is_not_inferred_by_code"] = False
        self.assertFalse(self.schema_validator.is_valid(invalid_summary))

    def test_checked_in_schema_tracks_frozen_sampling_constants(self):
        schema = json.loads(SCHEMA_PATH.read_text())
        sampling = schema["$defs"]["requestSampling"]["properties"]
        self.assertEqual(sampling["seed"]["const"], workflow.REQUEST_SAMPLING_SEED)
        self.assertEqual(sampling["cap_per_episode"]["const"], workflow.REQUEST_SAMPLE_CAP)
        self.assertEqual(sampling["algorithm"]["const"], workflow.REQUEST_SAMPLING_ALGORITHM)
        seed_values = schema["$defs"]["packetRandomization"]["properties"]["distinct_rater_seed_values"]
        self.assertTrue(seed_values["uniqueItems"])
        self.assertIn("pattern", schema["$defs"]["rfc3339Utc"])
        self.assertIn("allOf", schema["$defs"]["request"])
        self.assertIn("allOf", schema["$defs"]["annotation"])
        self.assertEqual(
            schema["$defs"]["raterResponse"]["properties"]["rater_code"]["$ref"],
            "#/$defs/canonicalCode",
        )


if __name__ == "__main__":
    unittest.main()
