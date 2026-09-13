from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np


WORKSHOP = Path(__file__).resolve().parents[1]
FORECAST = WORKSHOP / "experiments/forecast_layout"
sys.path.insert(0, str(FORECAST))
import fixed_observation_job as fixed  # noqa: E402


def write_json(path: Path, value: dict) -> dict:
    payload = fixed.canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": str(path.resolve()), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def accepted_chain(root: Path) -> tuple[Path, str, Path, str]:
    pool = json.loads((FORECAST / "layout_candidate_pool.json").read_text())
    candidate = next(row for row in pool["candidates"] if row["candidate_id"] == "P00__candidate_00")
    candidate_id = candidate["candidate_id"]
    candidate_sha = candidate["candidate_payload_sha256"]
    attempt = {
        "schema_version": fixed.GATE_ATTEMPT_SCHEMA,
        "candidate_id": candidate_id,
        "candidate_payload_sha256": candidate_sha,
        "candidate_pool_sha256": fixed.CANDIDATE_POOL_SHA256,
        "attempt_number": 0,
        "decision": "accepted",
        "passed": True,
        "model_request_count": 0,
        "behavioral_action_count": 0,
        "evaluation": {"passed": True, "decision": "accepted"},
        "error": None,
    }
    attempt_descriptor = write_json(root / "attempt/gate_attempt_receipt.json", attempt)
    record = {
        "schema_version": fixed.GATE_RECORD_SCHEMA,
        "study_namespace": fixed.NAMESPACE,
        "sequence": 0,
        "previous_record_sha256": None,
        "recorded_at_utc": "2026-09-13T00:00:00Z",
        "layout_pair_id": fixed.LAYOUT_PAIR_ID,
        "candidate_id": candidate_id,
        "candidate_rank": 0,
        "candidate_payload_sha256": candidate_sha,
        "candidate_pool_sha256": fixed.CANDIDATE_POOL_SHA256,
        "decision": "accepted",
        "passed": True,
        "model_request_count": 0,
        "behavioral_action_count": 0,
        "attempt_number": 0,
        "attempt_receipt": attempt_descriptor,
        "failure_count": 0,
        "failures": [],
    }
    record["record_sha256"] = fixed.sha256_bytes(fixed.canonical_bytes(record))
    ledger_path = root / "gate_ledger.jsonl"
    ledger_path.write_bytes(
        json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True).encode() + b"\n"
    )
    ledger_descriptor = fixed.file_identity(ledger_path)
    pose = {
        "schema_version": fixed.POSE_MANIFEST_SCHEMA,
        "study_namespace": fixed.NAMESPACE,
        "status": "LIVE_GATE_QUALIFIED_POSES_FROZEN_MODEL_EXECUTION_NOT_RELEASED",
        "physical_layout_gate_passed": True,
        "released_for_model_inference": False,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "candidate_pool_sha256": fixed.CANDIDATE_POOL_SHA256,
        "gate_ledger_sha256": ledger_descriptor["sha256"],
        "gate_ledger_last_record_sha256": record["record_sha256"],
        "qualified_layout_count": 1,
        "layout_pair_ids": [fixed.LAYOUT_PAIR_ID],
        "layout_pairs": {
            fixed.LAYOUT_PAIR_ID: {
                "layout_pair_id": fixed.LAYOUT_PAIR_ID,
                "candidate_id": candidate_id,
                "candidate_rank": 0,
                "candidate_payload_sha256": candidate_sha,
                "accepted_gate_record_sha256": record["record_sha256"],
                "accepted_gate_attempt_receipt": attempt_descriptor,
                "layouts": candidate["layouts"],
                "object_asset_provenance": candidate["object_asset_provenance"],
            }
        },
        "task_contract": {
            "action_cap": 450,
            "termination_terms": ["time_out"],
            "success_is_measurement_only": True,
            "success_termination_present": False,
        },
    }
    pose_path = root / "p00_pose_manifest.json"
    pose_descriptor = write_json(pose_path, pose)
    gate = {
        "schema_version": fixed.GATE_RECEIPT_SCHEMA,
        "study_namespace": fixed.NAMESPACE,
        "status": "finished",
        "job_id": "fixture-p00",
        "layout_pair_id": fixed.LAYOUT_PAIR_ID,
        "candidate_id": candidate_id,
        "candidate_payload_sha256": candidate_sha,
        "decision": "accepted",
        "exit_code": 0,
        "study_commit": "a" * 40,
        "source_contract_sha256": fixed.SOURCE_CONTRACT_SHA256,
        "candidate_pool_sha256": fixed.CANDIDATE_POOL_SHA256,
        "model_request_count": 0,
        "behavioral_action_count": 0,
        "gate_ledger": {**ledger_descriptor, "record_count": 1, "last_record_sha256": record["record_sha256"]},
        "gate_evidence": {
            "gate_record_sha256": record["record_sha256"],
            "gate_attempt_receipt": attempt_descriptor,
            "failure_count": 0,
            "raw_attempt_directory": str((root / "attempt").resolve()),
        },
    }
    gate_path = root / "fixture_gate_receipt.json"
    gate_descriptor = write_json(gate_path, gate)
    return gate_path, gate_descriptor["sha256"], pose_path, pose_descriptor["sha256"]


class FakeN3Client:
    def _extract_observation(self, observation, env_id=0):
        assert observation == {"raw": "observation"}
        assert env_id == 0
        return {"raw_image": np.full((3, 4, 3), 7, dtype=np.uint8)}

    def _pack_request(self, extracted, prompt):
        assert prompt == fixed.PROMPT
        assert extracted["raw_image"].shape == (3, 4, 3)
        return {
            "observation/image": np.full((540, 640, 3), 13, dtype=np.uint8),
            "observation/joint_position": np.arange(7, dtype=np.float32),
            "observation/gripper_position": np.asarray([0.4], dtype=np.float32),
            "prompt": prompt,
        }


class FakeOfficialD1:
    pass


class FakeD1Client(FakeOfficialD1):
    def _extract_observation(self, observation, env_id=0):
        assert observation == {"raw": "observation"}
        assert env_id == 0
        return {"left": np.full((4, 5, 3), 21, dtype=np.uint8)}

    def _pack_request(self, extracted, prompt):
        assert self.cam2_source == "right"
        assert self.resize == "pad"
        assert (self.image_height, self.image_width) == (180, 320)
        image = np.full((180, 320, 3), 23, dtype=np.uint8)
        return {
            "observation/exterior_image_0_left": image,
            "observation/exterior_image_1_left": image.copy(),
            "observation/wrist_image_left": image.copy(),
            "observation/joint_position": np.arange(7, dtype=np.float64),
            "observation/cartesian_position": np.zeros(6, dtype=np.float64),
            "observation/gripper_position": np.asarray([0.4], dtype=np.float64),
            "prompt": prompt,
        }


class FixedObservationTests(unittest.TestCase):
    def test_gate_and_pose_manifest_chain_is_fully_hash_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate_path, gate_sha, pose_path, pose_sha = accepted_chain(root)
            result = fixed.verify_gate_and_pose_manifest(
                gate_receipt_path=gate_path,
                gate_receipt_sha256=gate_sha,
                pose_manifest_path=pose_path,
                pose_manifest_sha256=pose_sha,
            )
            self.assertEqual(result["candidate_id"], "P00__candidate_00")
            self.assertEqual(result["accepted_gate_record_sha256"], json.loads((root / "gate_ledger.jsonl").read_text())["record_sha256"])
            pose = json.loads(pose_path.read_text())
            pose["task_contract"]["termination_terms"] = ["success", "time_out"]
            pose_path.write_bytes(fixed.canonical_bytes(pose))
            with self.assertRaisesRegex(fixed.FixedObservationError, "argument hash mismatch"):
                fixed.verify_gate_and_pose_manifest(
                    gate_receipt_path=gate_path,
                    gate_receipt_sha256=gate_sha,
                    pose_manifest_path=pose_path,
                    pose_manifest_sha256=pose_sha,
                )

    def test_missing_p00_manifest_is_frozen_once_from_accepted_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate_path, gate_sha, _pose_path, _pose_sha = accepted_chain(root)
            target = root / "released/p00_pose_manifest.json"
            path, digest = fixed.freeze_p00_pose_manifest(
                source_root=WORKSHOP.parents[1],
                gate_receipt_path=gate_path,
                gate_receipt_sha256=gate_sha,
                pose_manifest_path=target,
            )
            self.assertEqual(path, target.resolve())
            self.assertEqual(digest, fixed.sha256_file(target))
            manifest = json.loads(target.read_text())
            self.assertEqual(manifest["layout_pair_ids"], [fixed.LAYOUT_PAIR_ID])
            self.assertFalse(manifest["released_for_model_inference"])
            path2, digest2 = fixed.freeze_p00_pose_manifest(
                source_root=WORKSHOP.parents[1],
                gate_receipt_path=gate_path,
                gate_receipt_sha256=gate_sha,
                pose_manifest_path=target,
                expected_pose_manifest_sha256=digest,
            )
            self.assertEqual((path2, digest2), (path, digest))

    def test_exact_preprocessing_invokes_both_client_paths_without_model(self):
        n3, d1, preprocessing = fixed.extract_exact_model_inputs(
            {"raw": "observation"},
            study_root=Path(__file__).resolve().parents[3],
            n3_client_class=FakeN3Client,
            d1_client_class=FakeD1Client,
        )
        self.assertEqual(n3["observation/image"].shape, (540, 640, 3))
        self.assertEqual(n3["observation/joint_position"].dtype, np.float32)
        self.assertEqual(d1["observation/exterior_image_1_left"].shape, (180, 320, 3))
        self.assertEqual(d1["observation/cartesian_position"].dtype, np.float64)
        self.assertEqual(preprocessing["provenance"]["model_request_count"], 0)
        self.assertIn("N3/raw_image", preprocessing["arrays"])
        self.assertIn("D1/left", preprocessing["arrays"])

    def test_artifacts_retain_raw_arrays_and_both_strict_wire_fixtures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate_path, gate_sha, pose_path, pose_sha = accepted_chain(root)
            release = fixed.verify_gate_and_pose_manifest(
                gate_receipt_path=gate_path,
                gate_receipt_sha256=gate_sha,
                pose_manifest_path=pose_path,
                pose_manifest_sha256=pose_sha,
            )
            raw = {
                "image_obs": {
                    name: np.full((8, 10, 3), index, dtype=np.uint8)
                    for index, name in enumerate(fixed.RAW_CAMERAS)
                },
                "proprio_obs": {
                    "arm_joint_pos": np.arange(7, dtype=np.float32),
                    "gripper_pos": np.asarray([0.3], dtype=np.float32),
                },
            }
            cameras = {
                name: {
                    "frame_id": 75,
                    "native_capture_time_source": "_timestamp",
                    "capture_time_ns": 5_000_000_000,
                }
                for name in fixed.RAW_CAMERAS
            }
            clock = {"camera_counters": cameras, "physics_step": 600, "physics_time_s": 5.0}
            reset = {
                "reset_identity": "reset-1",
                "settle_evidence": {"settle_steps": 60, "stability_window_steps": 15},
            }
            n3 = {
                "observation/image": np.zeros((540, 640, 3), dtype=np.uint8),
                "observation/joint_position": np.zeros(7, dtype=np.float32),
                "observation/gripper_position": np.zeros(1, dtype=np.float32),
            }
            d1_image = np.zeros((180, 320, 3), dtype=np.uint8)
            d1 = {
                "observation/exterior_image_0_left": d1_image,
                "observation/exterior_image_1_left": d1_image,
                "observation/wrist_image_left": d1_image,
                "observation/joint_position": np.zeros(7, dtype=np.float64),
                "observation/cartesian_position": np.zeros(6, dtype=np.float64),
                "observation/gripper_position": np.zeros(1, dtype=np.float64),
            }
            receipt = fixed.write_capture_artifacts(
                output_dir=root / "capture",
                raw_observation=raw,
                physical_state={"arrays": {"objects/cube/position": np.zeros(3)}},
                native_clock=clock,
                settled_reset_receipt=reset,
                fresh_physical_checks={"collision": {"passed": True}, "visibility": {"passed": True}},
                release=release,
                n3_arrays=n3,
                d1_arrays=d1,
                preprocessing={"arrays": {"N3/raw": np.zeros(1)}, "provenance": {"model_request_count": 0}},
                study_commit="a" * 40,
                robolab_commit=fixed.ROBOLAB_COMMIT,
                environment_seed=2026091000,
                runtime_identity={"pod": "test"},
            )
            self.assertEqual(receipt["model_request_count"], 0)
            self.assertEqual(receipt["behavioral_action_count"], 0)
            with np.load(receipt["model_fixtures"]["N3"]["fixture"]["path"], allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), set(fixed.N3_ARRAY_KEYS))
            with np.load(receipt["model_fixtures"]["D1"]["fixture"]["path"], allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), set(fixed.D1_ARRAY_KEYS))
            with self.assertRaisesRegex(fixed.FixedObservationError, "overwrite capture directory"):
                fixed.write_capture_artifacts(
                    output_dir=root / "capture",
                    raw_observation=raw,
                    physical_state={"arrays": {"x": np.zeros(1)}},
                    native_clock=clock,
                    settled_reset_receipt=reset,
                    fresh_physical_checks={},
                    release=release,
                    n3_arrays=n3,
                    d1_arrays=d1,
                    preprocessing={"arrays": {"x": np.zeros(1)}, "provenance": {}},
                    study_commit="a" * 40,
                    robolab_commit=fixed.ROBOLAB_COMMIT,
                    environment_seed=2026091000,
                    runtime_identity={},
                )

    def test_native_clock_requires_real_camera_counter_and_timestamp(self):
        class Sensor:
            def __init__(self, include_timestamp=True):
                self._timestamp = np.asarray([2.3])
                if include_timestamp:
                    self._timestamp_last_update = np.asarray([2.25])

        class Sim:
            frame_count = 328
            current_time = 2.733333333333

        class Env:
            sim = Sim()
            common_step_counter = 41
            episode_length_buf = np.asarray([0])
            scene = {name: Sensor() for name in fixed.RAW_CAMERAS}

        cfg = types.SimpleNamespace(
            decimation=8,
            sim=types.SimpleNamespace(dt=1 / 120, render_interval=8),
        )
        observation = {
            "image_obs": {
                name: np.zeros((1, 8, 10, 3), dtype=np.uint8)
                for name in fixed.RAW_CAMERAS
            }
        }
        clock = fixed.capture_native_clock(Env(), cfg, observation)
        self.assertEqual(clock["physics_step"], 328)
        self.assertEqual(clock["camera_counters"]["wrist_cam"]["capture_time_ns"], 2_250_000_000)
        self.assertIsNone(clock["camera_counters"]["wrist_cam"]["native_frame_counter"])
        self.assertTrue(clock["camera_counters"]["wrist_cam"]["frame_id"].startswith("rgb-sha256:"))
        Env.scene = {name: Sensor(include_timestamp=name != "wrist_cam") for name in fixed.RAW_CAMERAS}
        with self.assertRaisesRegex(fixed.FixedObservationError, "wrist_cam capture timestamp"):
            fixed.capture_native_clock(Env(), cfg, observation)

    def test_queue_command_reexecutes_pinned_simulator_python(self):
        command = fixed.build_child_command(
            source_root=Path("/queue/sources") / ("a" * 40),
            output_dir=Path("/raw/capture"),
            pose_manifest_path=Path("/raw/p00.json"),
            pose_manifest_sha256="2" * 64,
            gate_receipt_path=Path("/queue/gate.json"),
            gate_receipt_sha256="3" * 64,
            study_commit="a" * 40,
            environment_seed=2026091000,
        )
        self.assertEqual(command[0], str(fixed.ROBOLAB_PYTHON))
        self.assertEqual(command[2], "capture")
        self.assertIn("--pose-manifest-sha256", command)
        self.assertNotIn("infer", command)


if __name__ == "__main__":
    unittest.main()
