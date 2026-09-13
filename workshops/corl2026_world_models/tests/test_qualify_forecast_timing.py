from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


MODULE = Path(__file__).resolve().parents[1] / "analysis/qualify_forecast_timing.py"
SPEC = importlib.util.spec_from_file_location("qualify_forecast_timing", MODULE)
timing = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(timing)

FREEZE_MODULE = Path(__file__).resolve().parents[1] / "analysis/freeze_development_release.py"
FREEZE_SPEC = importlib.util.spec_from_file_location("freeze_development_release_timing_integration", FREEZE_MODULE)
freeze = importlib.util.module_from_spec(FREEZE_SPEC)
assert FREEZE_SPEC.loader is not None
FREEZE_SPEC.loader.exec_module(freeze)

WORKSHOP = Path(__file__).resolve().parents[1]
CONTRACT = WORKSHOP / "experiments/forecast_layout/forecast_timing_lineage_contract.json"
CONTRACT_SHA = timing.sha256_file(CONTRACT)
SHA = "4" * 64


def write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(timing.pretty_bytes(value))
    return path


def descriptor(path: Path) -> dict:
    return timing.file_descriptor(path)


def sign_to(path: Path, value: dict) -> Path:
    return write_json(path, timing.sign_document(value))


def frozen(value: object) -> object:
    if isinstance(value, dict):
        return {"__type__": "mapping", "items": {key: frozen(child) for key, child in value.items()}}
    if isinstance(value, list):
        return {"__type__": "list", "items": [frozen(child) for child in value]}
    return value


def journal_event(sequence: int, previous: str | None, kind: str, payload: dict) -> dict:
    base = {
        "sequence": sequence,
        "kind": kind,
        "wall_time_ns": 2_000_000_000 + sequence,
        "monotonic_ns": 3_000_000_000 + sequence,
        "previous_event_sha256": previous,
        "payload": payload,
    }
    return {**base, "event_sha256": timing.sha256_bytes(timing.canonical_bytes(base, ensure_ascii=True))}


class EvidenceFixture:
    def __init__(self, root: Path, *, camera_period_s: float = 0.1, executed_identity_matches: bool = True):
        self.root = root
        self.contract, _ = timing.load_contract(CONTRACT, CONTRACT_SHA)
        self.capture_paths = {model: self._capture(model) for model in timing.EXPECTED_MODELS}
        self.qualification_paths = {
            "N3": self._n3_qualification(),
            "D1": self._d1_qualification(),
        }
        self.recorder_path = self._recorder(camera_period_s, executed_identity_matches)

    def _capture(self, model: str) -> Path:
        root = self.root / model.lower() / "capture"
        fixture = root / "fixture.npz"
        fixture.parent.mkdir(parents=True, exist_ok=True)
        if model == "N3":
            np.savez(
                fixture,
                **{
                    "observation/image": np.zeros((540, 640, 3), dtype=np.uint8),
                    "observation/joint_position": np.zeros((7,), dtype=np.float32),
                    "observation/gripper_position": np.zeros((1,), dtype=np.float32),
                },
            )
        else:
            np.savez(fixture, image=np.zeros((4, 4, 3), dtype=np.uint8))
        capture = {
            "schema_version": timing.FIXED_CAPTURE_SCHEMA,
            "status": "passed",
            "capture_id": f"capture-{model.lower()}",
            "settled_reset_identity": f"reset-{model.lower()}",
            "model_request_count": 0,
            "behavioral_action_count": 0,
            "source_capture": {
                "simulator_observation_id": f"sim-observation-{model.lower()}",
                "camera_frame_ids": {camera: f"{camera}:0" for camera in timing.CAMERAS},
                "camera_capture_time_ns": {camera: 1_000_000_000 for camera in timing.CAMERAS},
                "camera_timestamp_source": {camera: "native_sensor_timestamp" for camera in timing.CAMERAS},
            },
            "artifacts": {},
            "model_fixtures": {model: {"fixture": descriptor(fixture)}},
        }
        return write_json(root / "capture.json", capture)

    def _n3_nested(self, root: Path, role: str, shape: list[int]) -> dict:
        root.mkdir(parents=True, exist_ok=True)
        payload = root / f"{role}.npy"
        payload.write_bytes((role + "-payload").encode())
        structure = {
            "__type__": "numpy",
            "artifact": {"path": payload.name, "sha256": timing.sha256_file(payload), "bytes": payload.stat().st_size},
            "kind": "numpy",
            "dtype": "<f4",
            "shape": shape,
            "value_sha256": hashlib.sha256(role.encode()).hexdigest(),
        }
        logical = hashlib.sha256((role + "-logical").encode()).hexdigest()
        manifest = write_json(root / f"{role}.manifest.json", {
            "schema_version": "wmf-lossless-nested-payload-v1",
            "role": role,
            "structure": structure,
            "logical_sha256": logical,
        })
        return {
            "manifest_path": str(manifest.resolve()),
            "manifest_sha256": timing.sha256_file(manifest),
            "logical_sha256": logical,
        }

    def _n3_qualification(self) -> Path:
        root = self.root / "n3" / "qualification"
        artifacts = root / "artifacts"
        requests = []
        for index in range(6):
            decoded = index < 3
            row = {"request_id": f"n3-{index}", "request_index": index, "decode_requested": decoded}
            if decoded and index == 0:
                row.update({
                    "decode_mode": "offline_exact_retained_latent_after_action_capture",
                    "official_joint_generation_calls": 1,
                    "official_infer_decode_calls": 0,
                    "offline_decode_calls": 1,
                    "action_captured_before_offline_decode": True,
                    "returned_action_identity": {"shape": [32, 8], "value_sha256": "1" * 64},
                    "retained_latent_identity": {"shape": [1, 16, 9, 10], "value_sha256": "2" * 64},
                    "decoded_future_identity": {"shape": [33, 8, 8, 3], "value_sha256": "3" * 64},
                    "raw_generated_action_artifact": self._n3_nested(artifacts, "raw_action", [33, 8]),
                    "returned_action_artifact": self._n3_nested(artifacts, "returned_action", [32, 8]),
                    "retained_latent_artifact": self._n3_nested(artifacts, "latent", [1, 16, 9, 10]),
                    "decoded_future_artifact": self._n3_nested(artifacts, "decoded", [33, 8, 8, 3]),
                })
            requests.append(row)
        expected = timing.EXPECTED_MODELS["N3"]
        capture = json.loads(self.capture_paths["N3"].read_text())
        selected_camera = "over_shoulder_left_camera"
        qualification = {
            "schema_version": timing.N3_QUALIFICATION_SCHEMA,
            "status": "passed",
            "qualified": True,
            "generation_request_count": 6,
            "robot_episode_count": 0,
            "physical_time_mapping_qualified": False,
            "source": {"commit": expected["commit"], "git_tree": expected["tree"]},
            "checkpoint": {
                "revision": expected["checkpoint_revision"],
                "payload_aggregate_sha256": expected["checkpoint_aggregate"],
                "full_payload_rehash_performed": True,
            },
            "requests": requests,
            "fixed_observation": {
                "wire_observation_sha256": timing._n3_fixture_wire_sha256(
                    Path(capture["model_fixtures"]["N3"]["fixture"]["path"])
                ),
                "source_capture": {
                    "provenance_kind": "live_fixed_observation_zero_policy_input",
                    "capture_id": capture["capture_id"],
                    "settled_reset_identity": capture["settled_reset_identity"],
                    "selected_original_camera_id": selected_camera,
                    "camera_frame_ids": {
                        "observation/image": capture["source_capture"]["camera_frame_ids"][selected_camera]
                    },
                    "camera_capture_time_ns": {
                        "observation/image": capture["source_capture"]["camera_capture_time_ns"][selected_camera]
                    },
                    "camera_timestamp_source": {
                        "observation/image": capture["source_capture"]["camera_timestamp_source"][selected_camera]
                    },
                    "capture_receipt": descriptor(self.capture_paths["N3"]),
                },
            },
        }
        return write_json(root / "qualification.json", qualification)

    def _artifact(self, root: Path, name: str, value: bytes = b"tensor") -> dict:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value + name.encode())
        return {"path": str(path.resolve()), "file_sha256": timing.sha256_file(path), "bytes": path.stat().st_size}

    def _d1_qualification(self) -> Path:
        root = self.root / "d1" / "qualification"
        request = {
            "schema_version": timing.D1_REQUEST_SCHEMA,
            "request_index": 0,
            "official_action_path": "GrootSimPolicy.lazy_joint_forward_causal",
            "custom_s2_used": False,
            "patched_s1_used": False,
            "official_returned_action": {"shape": [24, 8], "data_sha256": "5" * 64,
                                           **self._artifact(root, "action.bin")},
            "latent_video": {"shape": [1, 16, 3, 10, 20], "data_sha256": "6" * 64,
                             **self._artifact(root, "latent.bin")},
            "offline_decode": {
                "requested": True,
                "performed": True,
                "latent_data_sha256_before": "6" * 64,
                "latent_data_sha256_after": "6" * 64,
                "decoded_rgb": {"shape": [9, 8, 8, 3], "data_sha256": "7" * 64,
                                **self._artifact(root, "decoded_rgb.bin")},
                "decoded_tensor": {"shape": [1, 3, 9, 8, 8], "data_sha256": "8" * 64,
                                   **self._artifact(root, "decoded_tensor.bin")},
            },
        }
        manifest = write_json(root / "episode.json", {
            "schema_version": "wmf-d1-episode-manifest-v1",
            "status": "complete",
            "request_count": 1,
            "requests": [request],
        })
        records = {
            f"d1-{index}": {
                "offline_decode_performed": index < 3,
                "episode_manifest": str(manifest.resolve()),
                "episode_manifest_sha256": timing.sha256_file(manifest),
            }
            for index in range(6)
        }
        report = write_json(root / "report.json", {
            "schema_version": timing.D1_REPORT_SCHEMA,
            "status": "passed",
            "passed": True,
            "generation_request_count": 6,
            "behavioral_episode_count": 0,
            "records": records,
        })
        expected = timing.EXPECTED_MODELS["D1"]
        job = {
            "schema_version": timing.D1_JOB_SCHEMA,
            "status": "finished",
            "decision": "qualified",
            "exit_code": 0,
            "generation_request_count": 6,
            "behavioral_episode_count": 0,
            "server_contract": {
                "official_repository_commit": expected["commit"],
                "official_repository_tree": expected["tree"],
                "checkpoint_aggregate_sha256": expected["checkpoint_aggregate"],
                "official_action_path": "GrootSimPolicy.lazy_joint_forward_causal",
                "custom_s2_used": False,
                "patched_s1_used": False,
            },
            "probe": {"passed": True, "status": "passed", "artifact": descriptor(report)},
            "capture_manifest": {
                "schema_version": "wmf-d1-capture-manifest-validation-v1",
                "status": "passed",
                "capture_id": "capture-d1",
                "manifest": descriptor(self.capture_paths["D1"]),
            },
        }
        return write_json(root / "job.json", job)

    def _observation_payload(self, root: Path, step: int, clock: dict) -> dict:
        payload_root = root / "payloads"
        payload_root.mkdir(parents=True, exist_ok=True)
        arrays = {
            "array_0000": np.zeros((4, 4, 3), dtype=np.uint8),
            "array_0001": np.ones((4, 4, 3), dtype=np.uint8),
            "array_0002": np.full((4, 4, 3), 2, dtype=np.uint8),
            "array_0003": np.zeros((7,), dtype=np.float32),
            "array_0004": np.zeros((1,), dtype=np.float32),
        }
        path = payload_root / f"{step:07d}_observation.npz"
        np.savez(path, **arrays)
        nodes = {
            key: {"__type__": "ndarray", "key": key, "shape": list(value.shape), "dtype": value.dtype.str}
            for key, value in arrays.items()
        }
        structure = frozen({
            "observation_id": f"obs_{step:06d}",
            "phase": "settled_reset" if step == 0 else "post_action",
            "clock": clock,
            "state": {},
            "host_observation_received_monotonic_ns": 9_000_000_000 + step,
        })
        assert isinstance(structure, dict)
        structure["items"]["image_obs"] = {
            "__type__": "mapping",
            "items": dict(zip(timing.CAMERAS, (nodes["array_0000"], nodes["array_0001"], nodes["array_0002"]))),
        }
        structure["items"]["proprio_obs"] = {
            "__type__": "mapping",
            "items": {"joint_pos": nodes["array_0003"], "gripper_pos": nodes["array_0004"]},
        }
        result = {
            "role": "observation",
            "structure": structure,
            "array_count": len(arrays),
            "artifact": {
                "path": str(path.relative_to(root)),
                "sha256": timing.sha256_file(path),
                "bytes": path.stat().st_size,
                "encoding": "numpy_npz_zip_stored",
            },
        }
        result["payload_sha256"] = timing.sha256_bytes(timing.canonical_bytes(result, ensure_ascii=True))
        return result

    def _recorder(self, camera_period_s: float, executed_identity_matches: bool) -> Path:
        root = self.root / "recorder" / "adapter_attempt"
        events: list[dict] = []
        previous = None

        def append(kind: str, payload: dict) -> None:
            nonlocal previous
            row = journal_event(len(events), previous, kind, payload)
            events.append(row)
            previous = row["event_sha256"]

        def clock(step: int) -> dict:
            capture = 1_000_000_000 + round(step * camera_period_s * 1e9)
            return {
                "control_step": step,
                "physics_step": step * 4,
                "physics_time_s": step * 0.1,
                "cameras": {
                    camera: {
                        "frame_id": f"{camera}:{step}",
                        "capture_time_ns": capture,
                        "timestamp_source": "native_sensor_timestamp",
                    }
                    for camera in timing.CAMERAS
                },
            }

        initial_clock = clock(0)
        append("observation_captured", {
            "observation_id": "obs_000000",
            "control_step": 0,
            "phase": "settled_reset",
            "clock": initial_clock,
            "artifact": self._observation_payload(root, 0, initial_clock),
        })
        identity = {"dtype": "<f8", "shape": [8], "order": "C", "value_sha256": "9" * 64}
        wrong_identity = {**identity, "value_sha256": "a" * 64}
        for step in range(1, 451):
            append("recording_qualification_action_proposed", {
                "action_step": step,
                "request_index": None,
                "chunk_offset": None,
                "action_identity": identity,
                "action_source": {
                    "kind": "joint_position_hold",
                    "policy_model": None,
                    "model_request": False,
                    "scientific_claim": "recorder_qualification_only_nonbehavioral",
                    "source_observation_id": f"obs_{step - 1:06d}",
                    "action_step": step,
                },
            })
            begin = 10_000_000_000 + step * 1000
            append("environment_step_started", {
                "action_step": step,
                "request_index": None,
                "chunk_offset": None,
                "env_step_start_monotonic_ns": begin,
                "executed_action_identity": identity if executed_identity_matches or step != 1 else wrong_identity,
            })
            append("environment_step_completed", {
                "action_step": step,
                "request_index": None,
                "chunk_offset": None,
                "env_step_start_monotonic_ns": begin,
                "env_step_end_monotonic_ns": begin + 500,
                "success_predicates": {"left": False, "right": False, "released": False},
                "terminated": False,
                "truncated": step == 450,
            })
            current_clock = clock(step)
            append("observation_captured", {
                "observation_id": f"obs_{step:06d}",
                "control_step": step,
                "phase": "post_action",
                "clock": current_clock,
                "artifact": self._observation_payload(root, step, current_clock) if step <= 32 else None,
            })
        append("attempt_finalized", {"stop_reason": "action_cap"})
        journal = root / "events.partial.jsonl"
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal.write_bytes(b"".join(timing.canonical_bytes(row, ensure_ascii=True) + b"\n" for row in events))
        completion = write_json(root / "completion.json", {
            "schema_version": timing.RECORDER_ATTEMPT_SCHEMA,
            "recording_qualification_valid": True,
            "behavioral_result_valid": False,
            "model_attached": False,
            "stop_reason": "action_cap",
            "actions_executed": 450,
            "observation_count": 451,
            "request_count": 0,
            "event_count": len(events),
            "journal_tail_sha256": previous,
        })
        journal_desc = descriptor(journal)
        journal_desc.update(event_count=len(events), tail_sha256=previous)
        child = {
            "schema_version": timing.RECORDER_CHILD_SCHEMA,
            "status": "passed",
            "actions_executed": 450,
            "observation_count": 451,
            "model_request_count": 0,
            "behavioral_episode_count": 0,
            "recording_qualification_count": 1,
            "model_attached": False,
            "action_source": "joint_position_hold_from_each_preceding_original_proprioception",
            "adapter_completion": descriptor(completion),
            "adapter_journal": journal_desc,
        }
        return write_json(self.root / "recorder" / "child.json", child)

    def source_audit(self, model: str) -> Path:
        expected = timing.EXPECTED_MODELS[model]
        mapping = [
            {"generated_frame_index": frame, "executed_control_boundary": boundary}
            for frame, boundary in zip(expected["decoded_frame_indices"], expected["control_boundaries"])
        ]
        return sign_to(self.root / model.lower() / "source_audit.json", {
            "schema_version": timing.SOURCE_AUDIT_SCHEMA,
            "study_id": timing.STUDY_ID,
            "model_id": model,
            "status": "source_mapping_passed_live_probe_required",
            "contract": descriptor(CONTRACT),
            "source": {"commit": expected["commit"], "git_tree": expected["tree"]},
            "checkpoint_expected": self.contract["models"][model]["checkpoint"],
            "source_mapping": mapping,
            "source_mapping_sha256": timing.sha256_bytes(timing.canonical_bytes(mapping)),
            "physical_time_qualified": False,
            "prohibited_inferences_used": [],
        })

    def generation_probe(self, model: str) -> Path:
        result = timing.normalize_generation_probe(
            model_id=model,
            qualification_path=self.qualification_paths[model],
            qualification_sha256=timing.sha256_file(self.qualification_paths[model]),
            contract_path=CONTRACT,
            contract_sha256=CONTRACT_SHA,
        )
        return write_json(self.root / model.lower() / "generation_probe.json", result)

    def authority(self, model: str) -> Path:
        source = self.source_audit(model)
        generation = self.generation_probe(model)
        result = timing.qualify_timing(
            model_id=model,
            contract_path=CONTRACT,
            contract_sha256=CONTRACT_SHA,
            source_audit_path=source,
            source_audit_sha256=timing.sha256_file(source),
            generation_probe_path=generation,
            generation_probe_sha256=timing.sha256_file(generation),
            recorder_receipt_path=self.recorder_path,
            recorder_receipt_sha256=timing.sha256_file(self.recorder_path),
            camera_id="over_shoulder_left_camera",
        )
        return write_json(self.root / model.lower() / "authority.json", result)

    def development_entry(self, model: str) -> dict:
        root = self.root / model.lower() / "development"
        cell_id = f"synthetic-development-{model}"
        events: list[dict] = []
        previous = None

        def append(kind: str, payload: dict) -> None:
            nonlocal previous
            row = journal_event(len(events), previous, kind, payload)
            events.append(row)
            previous = row["event_sha256"]

        for step in range(451):
            capture = 1_000_000_000 + step * 100_000_000
            append("observation_captured", {
                "observation_id": f"obs_{step:06d}",
                "control_step": step,
                "clock": {
                    "control_step": step,
                    "physics_step": step * 4,
                    "physics_time_s": step * 0.1,
                    "cameras": {
                        camera: {
                            "frame_id": f"{camera}:{step}",
                            "capture_time_ns": capture,
                            "timestamp_source": "native_sensor_timestamp",
                        }
                        for camera in timing.CAMERAS
                    },
                },
            })
        append("model_request_packed", {
            "request_index": 0,
            "action_step_start": 0,
            "current_observation_id": "obs_000000",
        })
        if model == "D1":
            request_value = {
                "schema_version": timing.D1_REQUEST_SCHEMA,
                "configuration_id": "D1",
                "episode_id": "synthetic-d1-episode",
                "request_index": 0,
                "official_action_path": "GrootSimPolicy.lazy_joint_forward_causal",
                "custom_s2_used": False,
                "patched_s1_used": False,
                "offline_decode": {
                    "performed": True,
                    "decoded_rgb": {"shape": [9, 8, 8, 3]},
                    "decoded_tensor": {"shape": [1, 3, 9, 8, 8]},
                },
            }
        else:
            request_value = {
                "schema_version": timing.N3_REQUEST_SCHEMA,
                "study_id": timing.STUDY_ID,
                "cell_id": cell_id,
                "request_index": 0,
                "action_step_start": 0,
                "behavioral_model_request": True,
                "generation_qualification_request": False,
                "decoded_future_shape": [33, 8, 8, 3],
            }
        request = write_json(root / "request.json", request_value)
        raw_response = {
            "wmf_server_request_receipt": descriptor(request),
            "wmf_request_index": 0,
        }
        if model == "D1":
            raw_response["wmf_episode_context_id"] = request_value["episode_id"]
        else:
            raw_response.update({
                "wmf_cell_id": cell_id,
                "wmf_action_step_start": 0,
                "wmf_server_context_id": "synthetic-n3-context",
            })
        response_artifact = {
            "role": "model_response",
            "structure": frozen({"raw_response": raw_response, "future_evidence": {}}),
            "array_count": 0,
        }
        response_artifact["payload_sha256"] = timing.sha256_bytes(
            timing.canonical_bytes(response_artifact, ensure_ascii=True)
        )
        append("model_response_received", {
            "request_index": 0,
            "response_artifact": response_artifact,
        })
        append("attempt_finalized", {"stop_reason": "action_cap"})
        journal = root / "events.partial.jsonl"
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal.write_bytes(b"".join(timing.canonical_bytes(row, ensure_ascii=True) + b"\n" for row in events))
        completion = write_json(root / "completion.json", {
            "schema_version": timing.RECORDER_ATTEMPT_SCHEMA,
            "study_id": timing.STUDY_ID,
            "identity": {"cell_id": cell_id, "model_config": model, "stage": "development"},
            "behavioral_result_valid": True,
            "technical_invalid": False,
            "request_execution": [{
                "request_index": 0,
                "action_step_start": 0,
                "executed_actions": timing.EXPECTED_MODELS[model]["prefix"],
            }],
            "event_count": len(events),
            "journal_tail_sha256": previous,
        })
        journal_desc = descriptor(journal)
        journal_desc.update(event_count=len(events), tail_sha256=previous)
        return {
            "cell_id": cell_id,
            "request_index": 0,
            "request_receipt": descriptor(request),
            "adapter_completion": descriptor(completion),
            "adapter_journal": journal_desc,
        }


class ForecastTimingQualificationTests(unittest.TestCase):
    def test_contract_identity_survives_staged_prefix_and_rejects_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relative = timing.CONTRACT_REPOSITORY_RELATIVE
            old = root / "sources" / ("a" * 40) / relative
            current = root / "sources" / ("b" * 40) / relative
            old.parent.mkdir(parents=True)
            current.parent.mkdir(parents=True)
            old.write_bytes(CONTRACT.read_bytes())
            current.write_bytes(CONTRACT.read_bytes())
            timing.require_contract_descriptor_matches(
                descriptor(old), current, base=root, label="staged contract"
            )

            wrong_path = root / "same-bytes-wrong-relative-name.json"
            wrong_path.write_bytes(CONTRACT.read_bytes())
            with self.assertRaisesRegex(
                timing.TimingQualificationError, "repository-relative identity"
            ):
                timing.require_contract_descriptor_matches(
                    descriptor(wrong_path), current, base=root, label="staged contract"
                )

            wrong_content = root / "sources" / ("c" * 40) / relative
            wrong_content.parent.mkdir(parents=True)
            wrong_content.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(timing.TimingQualificationError, "identity changed"):
                timing.require_contract_descriptor_matches(
                    descriptor(wrong_content), current, base=root, label="staged contract"
                )

    def test_qualification_accepts_identical_contract_from_fresh_staged_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = EvidenceFixture(root)
            staged = root / "sources" / ("d" * 40) / timing.CONTRACT_REPOSITORY_RELATIVE
            staged.parent.mkdir(parents=True)
            staged.write_bytes(CONTRACT.read_bytes())
            source = fixture.source_audit("D1")
            generation = fixture.generation_probe("D1")
            authority = timing.qualify_timing(
                model_id="D1",
                contract_path=staged,
                contract_sha256=timing.sha256_file(staged),
                source_audit_path=source,
                source_audit_sha256=timing.sha256_file(source),
                generation_probe_path=generation,
                generation_probe_sha256=timing.sha256_file(generation),
                recorder_receipt_path=fixture.recorder_path,
                recorder_receipt_sha256=timing.sha256_file(fixture.recorder_path),
                camera_id="over_shoulder_left_camera",
            )
            self.assertEqual(authority["status"], "qualified_from_source_lineage_and_native_zero_policy_clocks")
            self.assertEqual(len(authority["generated_targets"]), 2)

    def test_production_contract_is_source_only_and_prohibits_shortcuts(self) -> None:
        contract, _ = timing.load_contract(CONTRACT, CONTRACT_SHA)
        self.assertEqual(contract["status"], "source_mapping_frozen_live_clock_probe_required")
        self.assertTrue(all(contract["prohibited_inferences"].values()))
        self.assertEqual(contract["models"]["D1"]["lineage"]["executed_control_boundaries"],
                         [0, 3, 6, 9, 12, 15, 18, 21, 24])
        self.assertNotIn("presentation_fps", json.dumps(contract))

    def test_n3_live_input_preparation_is_zero_policy_and_capture_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = EvidenceFixture(root)
            prepared = timing.prepare_n3_live_input(
                capture_path=fixture.capture_paths["N3"],
                capture_sha256=timing.sha256_file(fixture.capture_paths["N3"]),
                camera_id="over_shoulder_left_camera",
                output_dir=root / "prepared_n3",
            )
            self.assertEqual(prepared["schema_version"], timing.N3_LIVE_INPUT_SCHEMA)
            self.assertEqual(prepared["model_request_count"], 0)
            self.assertEqual(prepared["behavioral_action_count"], 0)
            manifest = json.loads(Path(prepared["observation_manifest"]["path"]).read_text())
            source = manifest["source_capture"]
            self.assertEqual(source["provenance_kind"], "live_fixed_observation_zero_policy_input")
            self.assertEqual(source["selected_original_camera_id"], "over_shoulder_left_camera")
            self.assertEqual(set(source["camera_capture_time_ns"]), {"observation/image"})

    def test_both_models_close_only_with_source_generation_and_native_clocks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EvidenceFixture(Path(temporary))
            expected = {"N3": (32, 1, 32), "D1": (2, 3, 6)}
            for model, (count, first_boundary, last_boundary) in expected.items():
                authority_path = fixture.authority(model)
                authority = timing.validate_timing_authority(
                    authority_path, timing.sha256_file(authority_path), expected_model=model
                )
                self.assertEqual(len(authority["generated_targets"]), count)
                self.assertEqual(authority["qualified_mapping_rows"][0]["executed_control_boundary"], first_boundary)
                self.assertEqual(authority["qualified_mapping_rows"][-1]["executed_control_boundary"], last_boundary)
                self.assertFalse(authority["presentation_video_fps_used"])
                self.assertFalse(authority["dreamzero_action_block_ratio_used_as_mapping"])

    def test_generation_probe_is_rederived_from_raw_qualification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EvidenceFixture(Path(temporary))
            probe_path = fixture.generation_probe("N3")
            probe = json.loads(probe_path.read_text())
            probe["selected_request"]["decoded_value_sha256"] = "b" * 64
            probe.pop("payload_sha256")
            forged = sign_to(Path(temporary) / "forged_generation.json", probe)
            with self.assertRaisesRegex(timing.TimingQualificationError, "underlying live qualification"):
                timing.validate_generation_probe(forged, timing.sha256_file(forged), expected_model="N3")

    def test_exact_hold_execution_identity_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EvidenceFixture(Path(temporary), executed_identity_matches=False)
            source = fixture.source_audit("D1")
            generation = fixture.generation_probe("D1")
            with self.assertRaisesRegex(timing.TimingQualificationError, "executed-action identity"):
                timing.qualify_timing(
                    model_id="D1", contract_path=CONTRACT, contract_sha256=CONTRACT_SHA,
                    source_audit_path=source, source_audit_sha256=timing.sha256_file(source),
                    generation_probe_path=generation, generation_probe_sha256=timing.sha256_file(generation),
                    recorder_receipt_path=fixture.recorder_path,
                    recorder_receipt_sha256=timing.sha256_file(fixture.recorder_path),
                    camera_id="over_shoulder_left_camera",
                )

    def test_camera_physics_disagreement_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EvidenceFixture(Path(temporary), camera_period_s=0.2)
            source = fixture.source_audit("D1")
            generation = fixture.generation_probe("D1")
            with self.assertRaisesRegex(timing.TimingQualificationError, "residual exceeds tolerance"):
                timing.qualify_timing(
                    model_id="D1", contract_path=CONTRACT, contract_sha256=CONTRACT_SHA,
                    source_audit_path=source, source_audit_sha256=timing.sha256_file(source),
                    generation_probe_path=generation, generation_probe_sha256=timing.sha256_file(generation),
                    recorder_receipt_path=fixture.recorder_path,
                    recorder_receipt_sha256=timing.sha256_file(fixture.recorder_path),
                    camera_id="over_shoulder_left_camera",
                )

    def test_resigned_authority_tamper_is_rejected_against_raw_clocks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EvidenceFixture(Path(temporary))
            authority_path = fixture.authority("D1")
            authority = json.loads(authority_path.read_text())
            authority["qualified_mapping_rows"][0]["camera_elapsed_s"] = 99.0
            authority["generated_targets"][0]["target_physical_time_s"] = 99.0
            authority.pop("payload_sha256")
            forged = sign_to(Path(temporary) / "forged_authority.json", authority)
            with self.assertRaisesRegex(timing.TimingQualificationError, "native clock trace"):
                timing.validate_timing_authority(forged, timing.sha256_file(forged), expected_model="D1")

    def test_bind_development_requires_exact_authority_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = EvidenceFixture(root)
            authority_path = fixture.authority("D1")
            authority = json.loads(authority_path.read_text())
            entry = fixture.development_entry("D1")
            inventory = write_json(root / "inventory.json", {
                "schema_version": timing.REQUEST_INVENTORY_SCHEMA,
                "study_id": timing.STUDY_ID,
                "model_id": "D1",
                "request_receipts": [entry],
            })
            bound = timing.bind_development_requests(
                authority_path=authority_path,
                authority_sha256=timing.sha256_file(authority_path),
                inventory_path=inventory,
                inventory_sha256=timing.sha256_file(inventory),
            )
            self.assertEqual(bound["schema_version"], timing.DEVELOPMENT_TIMING_SCHEMA)
            self.assertEqual(len(bound["source_request_receipt_sha256s"]), 1)
            self.assertEqual(bound["binding_mode"], "immutable_request_receipt_native_clock_sidecar")
            sidecar_path = write_json(root / "timing_sidecar.json", bound)
            timing.validate_development_timing(
                sidecar_path,
                timing.sha256_file(sidecar_path),
                expected_model="D1",
                expected_request_hashes=bound["source_request_receipt_sha256s"],
            )
            request_path = Path(entry["request_receipt"]["path"])
            validated, validated_path = freeze._validate_generated_timing(
                {"generated_target_timing_receipt": descriptor(sidecar_path)},
                model="D1",
                base=root,
                request_hashes=[timing.sha256_file(request_path)],
                request_receipts=[json.loads(request_path.read_text())],
            )
            self.assertEqual(validated["binding_mode"], "immutable_request_receipt_native_clock_sidecar")
            self.assertEqual(validated_path, sidecar_path.resolve())
            forged_value = copy.deepcopy(bound)
            forged_value["request_timing_bindings"][0]["target_bindings"][0]["camera_elapsed_s"] += 0.01
            forged_value.pop("payload_sha256")
            forged_sidecar = sign_to(root / "forged_timing_sidecar.json", forged_value)
            with self.assertRaisesRegex(timing.TimingQualificationError, "immutable receipts/clocks"):
                timing.validate_development_timing(
                    forged_sidecar,
                    timing.sha256_file(forged_sidecar),
                    expected_model="D1",
                )
            wrong_entry = copy.deepcopy(entry)
            wrong_entry["request_index"] = 1
            wrong_inventory = write_json(root / "wrong_inventory.json", {
                **json.loads(inventory.read_text()),
                "request_receipts": [wrong_entry],
            })
            with self.assertRaisesRegex(timing.TimingQualificationError, "request index changed"):
                timing.bind_development_requests(
                    authority_path=authority_path,
                    authority_sha256=timing.sha256_file(authority_path),
                    inventory_path=wrong_inventory,
                    inventory_sha256=timing.sha256_file(wrong_inventory),
                )
            copied_request = root / "same-bytes-other-attempt-request.json"
            copied_request.write_bytes(request_path.read_bytes())
            substituted_entry = copy.deepcopy(entry)
            substituted_entry["request_receipt"] = descriptor(copied_request)
            substituted_inventory = write_json(root / "substituted_inventory.json", {
                **json.loads(inventory.read_text()),
                "request_receipts": [substituted_entry],
            })
            with self.assertRaisesRegex(timing.TimingQualificationError, "points to another file"):
                timing.bind_development_requests(
                    authority_path=authority_path,
                    authority_sha256=timing.sha256_file(authority_path),
                    inventory_path=substituted_inventory,
                    inventory_sha256=timing.sha256_file(substituted_inventory),
                )

    def test_symlink_and_duplicate_json_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = EvidenceFixture(root)
            source = fixture.source_audit("N3")
            link = root / "source-link.json"
            link.symlink_to(source)
            with self.assertRaisesRegex(timing.TimingQualificationError, "symlink"):
                timing.validate_source_audit(link, timing.sha256_file(source))
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"schema_version":"x","schema_version":"y"}\n', encoding="utf-8")
            with self.assertRaisesRegex(timing.TimingQualificationError, "duplicate JSON key"):
                timing.load_json(duplicate, "duplicate fixture")


if __name__ == "__main__":
    unittest.main()
