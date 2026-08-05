from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

from experiments.v3.robotwin_wams.contract import (
    EMPTY_DIFF_SHA256,
    MODEL_SPECS,
    RELEASE_GATES,
    AdapterError,
    adapter_contract_sha256,
    canonical_sha256,
    load_authorized_pair,
    validate_measurement_transform,
    verify_runtime_identity,
)
from experiments.v3.robotwin_wams.launcher import (
    build_guard_command,
    build_native_command,
    compile_native_pair,
)


ROOT = Path(__file__).resolve().parents[1]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RobotwinWamV3AdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.temp = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def runtime_manifest(
        self,
        model_id: str,
        *,
        external: Path | None = None,
        simulator: Path | None = None,
    ) -> tuple[Path, dict]:
        external = (external or self.temp / "external").resolve()
        simulator = (simulator or self.temp / "robotwin").resolve()
        spec = MODEL_SPECS[model_id]
        generic_artifact = {
            "path": str(self.temp / "model_blind_evidence.json"),
            "sha256": "a" * 64,
            "bytes": 1,
        }
        transform = {
            "schema_version": "vla-wam-shared-v3-robotwin-frame-transform-v1",
            "source_frame_id": "sapien_world_xyz_m",
            "target_frame_id": "robot_base_object_minus_reference_xyz_m",
            "status": "passed_model_blind_before_behavior",
            "recorded_before_any_v3_behavioral_inference": True,
            "model_requests_during_validation": 0,
            "models_loaded_during_validation": 0,
            "rotation_source_to_target": [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
            "translation_source_to_target_m": [1.0, 2.0, 3.0],
            "fixture_validation_artifact": generic_artifact,
        }
        transform["transform_sha256"] = canonical_sha256(transform)
        runtime = {
            "schema_version": "vla-wam-shared-v3-robotwin-runtime-identity-v1",
            "study_id": "vla_wam_language_steerability_v3",
            "model_id": model_id,
            "status": "passed_all_registered_release_gates",
            "runtime_id": f"test-{model_id}-runtime",
            "phase_a_queue_sha256": (
                "8350b98f958424b56b66e67e8c70ec3951d27f4ae257476d6f08c0aaa873cb7c"
            ),
            "adapter_contract_sha256": adapter_contract_sha256(ROOT, model_id),
            "external_repository": {
                "path": str(external),
                "commit": spec["source_commit"],
                "diff_hash": EMPTY_DIFF_SHA256,
            },
            "simulator_repository": {
                "path": str(simulator),
                "commit": spec["simulator_commit"] or "f" * 40,
                "diff_hash": EMPTY_DIFF_SHA256,
            },
            "checkpoint": {
                "id": spec["checkpoint_id"],
                "revision": "exact-test-revision",
                "sha256": "b" * 64,
                "hash_gate_passed": True,
                "hash_manifest_artifact": generic_artifact,
            },
            "environment": {"lock_artifact": generic_artifact},
            "simulator_version": "RoboTwin pinned test runtime",
            "renderer_backend": "SAPIEN Vulkan headless",
            "adapter_files": {
                "wrapper": {
                    "path": str((external / spec["wrapper_path"]).resolve()),
                    "sha256": spec["wrapper_sha256"],
                },
                "runner": {
                    "path": str((external / spec["runner_path"]).resolve()),
                    "sha256": spec["runner_sha256"],
                },
            },
            "release_gates": {
                gate: {"status": "passed", "artifact": generic_artifact}
                for gate in RELEASE_GATES
            },
            "measurement_transform": transform,
        }
        runtime["runtime_identity_sha256"] = canonical_sha256(runtime)
        path = self.temp / f"{model_id}_runtime.json"
        path.write_text(json.dumps(runtime))
        return path, runtime

    def test_queue_resolves_exact_pair_and_rejects_r0(self) -> None:
        pair = load_authorized_pair(ROOT, "efficient_wam_rt_robotwin", 3, 1)
        self.assertEqual(pair.environment_seed, 4_300_003)
        self.assertEqual(pair.policy_seed, 8_503)
        self.assertEqual(pair.anchor_task, "place_a2b_right")
        self.assertEqual(
            pair.left["prompt"],
            "Put the small woodenblock to the left of the red playingcards box.",
        )
        self.assertEqual(
            pair.right["prompt"],
            "Put the small woodenblock to the right of the red playingcards box.",
        )
        with self.assertRaisesRegex(AdapterError, "r0 is immutable"):
            load_authorized_pair(ROOT, "efficient_wam_rt_robotwin", 3, 0)
        with self.assertRaisesRegex(AdapterError, "exactly r1..r9"):
            load_authorized_pair(ROOT, "fastwam_robotwin", 3, 10)

    def test_commands_reuse_exact_v2_entrypoints_without_resume(self) -> None:
        external = self.temp / "external"
        for model_id in MODEL_SPECS:
            pair = load_authorized_pair(ROOT, model_id, 6, 9)
            command = build_native_command(
                pair,
                study_root=ROOT,
                external_repository=external,
                native_output_dir=self.temp / model_id,
            )
            rendered = " ".join(command)
            self.assertNotIn("--resume", command)
            self.assertIn("--prompt-family direct_command", rendered)
            self.assertIn("--max-actions 400", rendered)
            self.assertIn("4300006", rendered)
            self.assertIn("9306", rendered)
            if model_id == "efficient_wam_rt_robotwin":
                self.assertIn("--predicted-video-max-chunks 1", rendered)
            elif model_id == "fastwam_robotwin":
                self.assertIn("--text-cfg-scale 2.0", rendered)
                self.assertIn("--action-horizon 32", rendered)
                self.assertNotIn("--contrastive-negative", command)
            else:
                self.assertIn("--condition correct --condition swapped", rendered)
                self.assertIn("--save-first-predicted-latent", command)
            guard = build_guard_command(
                pair,
                study_root=ROOT,
                attempt_dir=self.temp / "attempt",
                gpu_index=2,
                native_command=command,
            )
            self.assertIn("--launch", guard)
            self.assertEqual(guard.count("--requested-relation"), 2)
            self.assertIn(pair.pair_id, guard)

    def test_runtime_manifest_binds_transform_and_release_gates(self) -> None:
        external = self.temp / "external"
        simulator = self.temp / "robotwin"
        path, runtime = self.runtime_manifest(
            "fastwam_robotwin", external=external, simulator=simulator
        )
        observed = verify_runtime_identity(
            ROOT,
            "fastwam_robotwin",
            path,
            external_repository=external,
            simulator_repository=simulator,
            verify_live_files=False,
        )
        self.assertEqual(observed["runtime_identity_sha256"], runtime["runtime_identity_sha256"])
        tampered = copy.deepcopy(runtime)
        tampered["release_gates"]["fixed_observation_exact_repeat"]["status"] = "failed"
        tampered["runtime_identity_sha256"] = canonical_sha256(
            {key: value for key, value in tampered.items() if key != "runtime_identity_sha256"}
        )
        path.write_text(json.dumps(tampered))
        with self.assertRaisesRegex(AdapterError, "has not passed"):
            verify_runtime_identity(
                ROOT,
                "fastwam_robotwin",
                path,
                external_repository=external,
                simulator_repository=simulator,
                verify_live_files=False,
            )

    def test_transform_rejects_axis_relabeling(self) -> None:
        _, runtime = self.runtime_manifest("fastwam_robotwin")
        transform = copy.deepcopy(runtime["measurement_transform"])
        transform["rotation_source_to_target"] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        transform["transform_sha256"] = canonical_sha256(
            {key: value for key, value in transform.items() if key != "transform_sha256"}
        )
        with self.assertRaisesRegex(AdapterError, r"world -X to robot-base \+Y"):
            validate_measurement_transform(transform, verify_artifacts=False)

    @staticmethod
    def native_state(object_xyz: list[float], relation: str, *, open_: bool) -> dict:
        target_xyz = [0.0, 0.0, 0.0]
        dx = object_xyz[0]
        dy = object_xyz[1]
        distance = (dx * dx + dy * dy) ** 0.5
        side = dx < 0 if relation == "left" else dx > 0
        region = 0.08 < distance < 0.2 and side and abs(dy) < 0.05
        return {
            "success": bool(region and open_),
            "relation_region": bool(region),
            "object_xyz": object_xyz,
            "target_xyz": target_xyz,
            "object_minus_target_x": dx,
            "object_minus_target_y": dy,
            "distance_xy": distance,
            "grippers_open": open_,
        }

    def write_native_condition(
        self,
        pair,
        relation: str,
        native_root: Path,
        *,
        success: bool = True,
    ) -> None:
        condition = (
            native_root
            / pair.anchor_task
            / f"environment_seed_{pair.environment_seed}"
            / f"sampling_seed_{pair.policy_seed}"
            / f"direct_command__{relation}"
        )
        condition.mkdir(parents=True)
        final_x = -0.1 if relation == "left" else 0.1
        trajectory = [
            {"action_step": 0, **self.native_state([0.3, 0.0, 0.10], relation, open_=True)},
            {"action_step": 1, **self.native_state([0.3, 0.0, 0.14], relation, open_=False)},
            {"action_step": 2, **self.native_state([final_x, 0.0, 0.14], relation, open_=False)},
            {
                "action_step": 3,
                **self.native_state(
                    [final_x if success else 0.3, 0.0, 0.10], relation, open_=True
                ),
            },
        ]
        trajectory_path = condition / "trajectory.json"
        trajectory_path.write_text(json.dumps(trajectory))
        action_path = condition / "action_trace.npz"
        self.write_zero_npz(action_path, (3, 14))
        video = condition / "simulator.mp4"
        video.write_bytes(b"synthetic-video-evidence")
        result = {
            "task": pair.anchor_task,
            "environment_seed": pair.environment_seed,
            "sampling_seed": pair.policy_seed,
            "condition": f"direct_command__{relation}",
            "prompt_family": "direct_command",
            "requested_relation": relation,
            "prompt": pair.cell(relation)["prompt"],
            "negative_prompt": "",
            "object_name": "086_woodenblock",
            "target_name": "081_playingcards",
            "actions_executed": 3,
            "requested_success": trajectory[-1]["success"],
            "initial": {key: value for key, value in trajectory[0].items() if key != "action_step"},
            "final": {key: value for key, value in trajectory[-1].items() if key != "action_step"},
            "wall_seconds": 1.25,
            "simulator_video": str(video.resolve()),
            "trajectory_path": str(trajectory_path.resolve()),
            "action_trace": {
                "path": str(action_path.resolve()),
                "sha256": file_hash(action_path),
                "count": 3,
                "shape": [3, 14],
            },
        }
        if pair.model_id == "efficient_wam_rt_robotwin":
            predicted = condition / "predicted_futures"
            predicted.mkdir()
            (predicted / "chunk000.mp4").write_bytes(b"decoded-future")
            result["predicted_video_dir"] = str(predicted.resolve())
        elif pair.model_id == "lingbot_va_robotwin":
            latent = condition / "first_predicted_latent.pt"
            latent.write_bytes(b"latent-future")
            result["first_predicted_latent_path"] = str(latent.resolve())
        (condition / "result.json").write_text(json.dumps(result))

    @staticmethod
    def write_zero_npz(path: Path, shape: tuple[int, ...]) -> None:
        header = repr(
            {"descr": "<f4", "fortran_order": False, "shape": shape}
        ).encode("latin1")
        padding = (16 - ((10 + len(header) + 1) % 16)) % 16
        header = header + b" " * padding + b"\n"
        payload = (
            b"\x93NUMPY"
            + bytes((1, 0))
            + struct.pack("<H", len(header))
            + header
            + b"\x00" * (math.prod(shape) * 4)
        )
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("executed.npy", payload)

    def prepare_attempt(
        self,
        model_id: str = "fastwam_robotwin",
        *,
        directory: str = "attempt",
    ) -> tuple[Path, object, dict]:
        attempt = self.temp / directory
        attempt.mkdir()
        (attempt / "native.stdout.log").write_text("completed\n")
        (attempt / "native.stderr.log").write_text("")
        (attempt / "thermal_events.jsonl").write_text(
            json.dumps(
                {"event": "monitor_completed", "worker_exit_code": 0}
            )
            + "\n"
        )
        pair = load_authorized_pair(ROOT, model_id, 3, 1)
        _, runtime = self.runtime_manifest(model_id)
        return attempt, pair, runtime

    def test_compile_writes_two_matched_raw_behavioral_rows(self) -> None:
        attempt, pair, runtime = self.prepare_attempt()
        for relation in ("left", "right"):
            self.write_native_condition(pair, relation, attempt / "native")
        summary = compile_native_pair(
            pair,
            runtime=runtime,
            attempt_dir=attempt,
            attempt_id="test-fastwam-pair03-r01-a1",
            guard_return_code=0,
        )
        self.assertEqual(summary["behavioral_rows"], 2)
        self.assertEqual(summary["infrastructure_rows"], 0)
        rows = [
            json.loads(line)
            for line in (attempt / "behavioral_episodes.jsonl").read_text().splitlines()
        ]
        self.assertEqual({row["failure_taxonomy"] for row in rows}, {"correct"})
        self.assertEqual({row["frozen_failure_stage"] for row in rows}, {"success"})
        by_relation = {row["requested_relation"]: row for row in rows}
        self.assertAlmostEqual(
            by_relation["left"]["measurements"]["signed_final_lateral_offset_m"],
            0.1,
        )
        self.assertAlmostEqual(
            by_relation["right"]["measurements"]["signed_final_lateral_offset_m"],
            -0.1,
        )
        self.assertTrue(all(row["operational_wall_time_valid"] for row in rows))
        self.assertTrue((attempt / "behavioral_episodes.jsonl.manifest.json").is_file())
        with self.assertRaisesRegex(AdapterError, "refusing to overwrite"):
            compile_native_pair(
                pair,
                runtime=runtime,
                attempt_dir=attempt,
                attempt_id="test-fastwam-pair03-r01-a1",
                guard_return_code=0,
            )

    def test_missing_mate_is_separate_infrastructure_not_behavioral_zero(self) -> None:
        attempt, pair, runtime = self.prepare_attempt()
        self.write_native_condition(pair, "left", attempt / "native")
        summary = compile_native_pair(
            pair,
            runtime=runtime,
            attempt_dir=attempt,
            attempt_id="test-fastwam-pair03-r01-a2",
            guard_return_code=0,
        )
        self.assertEqual(summary["behavioral_rows"], 1)
        self.assertEqual(summary["infrastructure_rows"], 1)
        behavior = json.loads((attempt / "behavioral_episodes.jsonl").read_text())
        infrastructure = json.loads((attempt / "infrastructure_attempts.jsonl").read_text())
        self.assertEqual(behavior["requested_relation"], "left")
        self.assertEqual(infrastructure["record_type"], "infrastructure_attempt")
        self.assertNotIn("requested_success", infrastructure)
        self.assertNotIn("failure_taxonomy", infrastructure)

    def test_all_three_model_interfaces_retain_their_exposed_future_contract(self) -> None:
        expected = {
            "efficient_wam_rt_robotwin": ("decoded_future_video", 1),
            "fastwam_robotwin": ("action_only_not_applicable", 0),
            "lingbot_va_robotwin": ("latent_only_future_not_decodable", 1),
        }
        for index, (model_id, (interface, future_count)) in enumerate(expected.items()):
            attempt, pair, runtime = self.prepare_attempt(
                model_id, directory=f"attempt_{index}"
            )
            for relation in ("left", "right"):
                self.write_native_condition(pair, relation, attempt / "native")
            summary = compile_native_pair(
                pair,
                runtime=runtime,
                attempt_dir=attempt,
                attempt_id=f"test-{model_id}-pair03-r01",
                guard_return_code=0,
            )
            self.assertEqual(summary["behavioral_rows"], 2)
            rows = [
                json.loads(line)
                for line in (attempt / "behavioral_episodes.jsonl").read_text().splitlines()
            ]
            self.assertTrue(all(row["future_interface"] == interface for row in rows))
            self.assertTrue(all(len(row["future_evidence"]) == future_count for row in rows))

    def test_mismatched_reset_invalidates_both_pair_members(self) -> None:
        attempt, pair, runtime = self.prepare_attempt(directory="mismatched_attempt")
        for relation in ("left", "right"):
            self.write_native_condition(pair, relation, attempt / "native")
        right_dir = (
            attempt
            / "native"
            / pair.anchor_task
            / f"environment_seed_{pair.environment_seed}"
            / f"sampling_seed_{pair.policy_seed}"
            / "direct_command__right"
        )
        trajectory_path = right_dir / "trajectory.json"
        trajectory = json.loads(trajectory_path.read_text())
        trajectory[0] = {
            "action_step": 0,
            **self.native_state([0.31, 0.0, 0.10], "right", open_=True),
        }
        trajectory_path.write_text(json.dumps(trajectory))
        result_path = right_dir / "result.json"
        result = json.loads(result_path.read_text())
        result["initial"] = {
            key: value for key, value in trajectory[0].items() if key != "action_step"
        }
        result_path.write_text(json.dumps(result))
        summary = compile_native_pair(
            pair,
            runtime=runtime,
            attempt_dir=attempt,
            attempt_id="test-fastwam-mismatched-reset",
            guard_return_code=0,
        )
        self.assertEqual(summary["behavioral_rows"], 0)
        self.assertEqual(summary["infrastructure_rows"], 2)
        self.assertTrue(
            all("physical initial states" in message for message in summary["errors"].values())
        )


if __name__ == "__main__":
    unittest.main()
