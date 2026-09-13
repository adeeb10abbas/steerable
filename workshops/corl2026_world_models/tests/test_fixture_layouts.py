"""Focused contracts for model-blind fixture generation and qualification."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


WORKSHOP = Path(__file__).resolve().parents[1]
LAYOUT_ROOT = WORKSHOP / "experiments/forecast_layout"
sys.path.insert(0, str(LAYOUT_ROOT))
fixture = importlib.import_module("fixture_layouts")
gate = importlib.import_module("model_blind_fixture_gate")
tasks = importlib.import_module("fixture_tasks")


def load_source():
    path = LAYOUT_ROOT / "layout_source_contract.json"
    value, payload = fixture.load_json_file(path)
    return fixture.validate_source_contract(value), fixture.sha256_bytes(payload)


def build_pool():
    source, digest = load_source()
    return fixture.build_candidate_pool(source, digest), source, digest


def passing_capture(candidate, source, arm, command, repeat):
    seed = gate.environment_seed(candidate, repeat)
    marker = f"{arm}:{repeat}"
    hash_for = lambda role: gate.sha256_bytes(f"{marker}:{role}".encode())
    cameras = {}
    for camera in source["live_gate"]["required_cameras"]:
        cameras[camera] = {
            "shape_hwc": [720, 1280, 3],
            "dtype": "uint8",
            "pixel_range": 255,
            "rgb_sha256": hash_for(f"rgb:{camera}"),
            "visibility_method": "instance_segmentation",
            "visibility_source_sha256": hash_for(f"segmentation:{camera}"),
            "visible_object_pixels": {name: 100 for name in fixture.MOVABLE_OBJECTS},
        }
    configured = {
        name: {
            "position_robot_base_m": list(candidate["layouts"][arm]["positions_robot_base_m"][name]),
            "quaternion_wxyz": list(candidate["layouts"][arm]["quaternions_wxyz"][name]),
        }
        for name in fixture.MOVABLE_OBJECTS
    }
    settled = copy.deepcopy(configured)
    collision_pairs = {
        pair: {"clear": True, "evidence_sha256": hash_for(f"collision:{pair}")}
        for pair in source["live_gate"]["forbidden_collision_pairs"]
    }
    return {
        "schema_version": gate.CAPTURE_SCHEMA,
        "study_namespace": fixture.NAMESPACE,
        "candidate_id": candidate["candidate_id"],
        "candidate_payload_sha256": candidate["candidate_payload_sha256"],
        "layout_pair_id": candidate["layout_pair_id"],
        "layout_arm": arm,
        "command": command,
        "repeat_index": repeat,
        "environment_seed": seed,
        "model_request_count": 0,
        "behavioral_action_count": 0,
        "configured_poses": configured,
        "settled_poses": settled,
        "settle": {
            "settle_steps": source["live_gate"]["settle_steps"],
            "stability_window_steps": source["live_gate"]["stability_window_steps"],
            "terminated_during_settle": False,
            "truncated_during_settle": False,
            "maxima_by_object": {
                name: {"max_linear_speed_m_s": 0.001, "max_angular_speed_rad_s": 0.01}
                for name in fixture.MOVABLE_OBJECTS
            },
        },
        "reset_fingerprints": {
            "reset_state_sha256": hash_for("state"),
            "initial_observation_sha256": hash_for("observation"),
            "initial_camera_rgb_sha256": {
                camera: hash_for(f"rgb:{camera}")
                for camera in source["live_gate"]["required_cameras"]
            },
        },
        "cameras": cameras,
        "collision_checks": {"query_complete": True, "forbidden_pairs": collision_pairs},
        "success_predicates": {"left": False, "right": False},
    }


def passing_matrix(candidate, source):
    return [
        passing_capture(candidate, source, arm, command, repeat)
        for arm in fixture.LAYOUT_ARMS
        for command in fixture.COMMANDS
        for repeat in range(source["live_gate"]["repeat_resets_per_condition"])
    ]


class CandidatePoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool, cls.source, cls.source_sha = build_pool()

    def test_complete_pool_has_three_spares_and_is_explicitly_unreleased(self):
        self.assertEqual(self.pool["planned_layout_ids"], list(fixture.PLANNED_LAYOUT_IDS))
        self.assertEqual(self.pool["planned_layout_count"], 29)
        self.assertEqual(self.pool["candidate_count"], 116)
        self.assertEqual(self.pool["spares_per_layout"], 3)
        self.assertFalse(self.pool["released"])
        self.assertFalse(self.pool["launch_ready"])
        self.assertFalse(self.pool["generation_uses_model_outcomes"])
        self.assertEqual(self.pool["model_request_count"], 0)

    def test_every_object_is_exactly_y_reflected_with_source_quaternion(self):
        for candidate in self.pool["candidates"]:
            for name in fixture.MOVABLE_OBJECTS:
                original = candidate["layouts"]["original"]["positions_robot_base_m"][name]
                reflected = candidate["layouts"]["reflected"]["positions_robot_base_m"][name]
                self.assertEqual(reflected, [original[0], -original[1] if original[1] else 0.0, original[2]])
                self.assertEqual(
                    candidate["layouts"]["original"]["quaternions_wxyz"][name],
                    self.source["objects"][name]["quaternion_wxyz"],
                )
                self.assertEqual(
                    candidate["layouts"]["reflected"]["quaternions_wxyz"][name],
                    self.source["objects"][name]["quaternion_wxyz"],
                )
                self.assertRegex(candidate["object_asset_provenance"][name]["asset_lfs_oid_sha256"], r"^[0-9a-f]{64}$")

    def test_generation_is_byte_deterministic_and_p00_primary_is_historical(self):
        again = fixture.build_candidate_pool(self.source, self.source_sha)
        self.assertEqual(fixture.canonical_json_bytes(self.pool), fixture.canonical_json_bytes(again))
        p00 = next(row for row in self.pool["candidates"] if row["candidate_id"] == "P00__candidate_00")
        self.assertEqual(p00["derivation_attempt"], -1)
        self.assertEqual(
            p00["layouts"]["original"]["positions_robot_base_m"],
            self.source["p00_original_positions_robot_base_m"],
        )

    def test_tampering_or_pretending_release_fails_closed(self):
        candidate = copy.deepcopy(self.pool["candidates"][0])
        candidate["released"] = True
        with self.assertRaisesRegex(fixture.LayoutContractError, "bypassed"):
            fixture.validate_candidate(candidate)
        candidate = copy.deepcopy(self.pool["candidates"][0])
        candidate["layouts"]["reflected"]["positions_robot_base_m"]["banana"][1] += 0.001
        candidate["candidate_payload_sha256"] = fixture._candidate_payload_sha(candidate)
        with self.assertRaisesRegex(fixture.LayoutContractError, "exact y reflection"):
            fixture.validate_candidate(candidate)


class LiveGateTests(unittest.TestCase):
    def setUp(self):
        self.pool, self.source, self.source_sha = build_pool()
        self.candidate = self.pool["candidates"][4]
        self.pool_payload = fixture.canonical_json_bytes(self.pool)
        self.pool_sha = fixture.sha256_bytes(self.pool_payload)

    def test_full_capture_matrix_passes_and_hashes_match_left_right(self):
        result = gate.evaluate_candidate_captures(
            self.candidate, passing_matrix(self.candidate, self.source), self.source
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["decision"], "accepted")
        self.assertEqual(result["capture_count"], 8)
        self.assertTrue(all(result["matched_left_right_checks"].values()))

    def test_success_collision_and_left_right_hash_mismatch_are_rejections(self):
        captures = passing_matrix(self.candidate, self.source)
        captures[0]["success_predicates"]["left"] = True
        first_pair = self.source["live_gate"]["forbidden_collision_pairs"][0]
        captures[1]["collision_checks"]["forbidden_pairs"][first_pair]["clear"] = False
        captures[3]["reset_fingerprints"]["initial_observation_sha256"] = "f" * 64
        result = gate.evaluate_candidate_captures(self.candidate, captures, self.source)
        self.assertFalse(result["passed"])
        self.assertEqual(result["decision"], "physical_rejection")
        self.assertTrue(any("success predicate" in failure for failure in result["failures"]))
        self.assertTrue(any("forbidden reset collision" in failure for failure in result["failures"]))
        self.assertTrue(any("initial-observation hashes differ" in failure for failure in result["failures"]))

    def test_missing_camera_or_changed_configured_pose_is_technical_invalidity(self):
        captures = passing_matrix(self.candidate, self.source)
        captures[0]["cameras"].pop("wrist_cam")
        with self.assertRaisesRegex(gate.GateEvidenceError, "camera inventory"):
            gate.evaluate_candidate_captures(self.candidate, captures, self.source)
        captures = passing_matrix(self.candidate, self.source)
        captures[0]["configured_poses"]["banana"]["position_robot_base_m"][0] += 0.1
        with self.assertRaisesRegex(gate.GateEvidenceError, "differs from candidate"):
            gate.evaluate_candidate_captures(self.candidate, captures, self.source)

    def test_projection_visibility_fallback_is_evidence_bound(self):
        captures = passing_matrix(self.candidate, self.source)
        row = captures[0]["cameras"]["over_shoulder_left_camera"]
        row.pop("visible_object_pixels")
        row["visibility_method"] = "calibrated_projection_and_obb_occlusion"
        row["camera_geometry_source_sha256"] = "a" * 64
        row["projected_object_centers_uv"] = {name: [100.0, 200.0] for name in fixture.MOVABLE_OBJECTS}
        row["projected_unoccluded_by_object"] = {name: True for name in fixture.MOVABLE_OBJECTS}
        result = gate.evaluate_candidate_captures(self.candidate, captures, self.source)
        self.assertTrue(result["passed"])

    def test_append_only_ledger_chain_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rejections.jsonl"
            first = gate.append_ledger_record(
                path,
                {"decision": "physical_rejection", "candidate_id": "D01__candidate_00"},
                recorded_at_utc="2026-09-12T00:00:00Z",
            )
            second = gate.append_ledger_record(
                path,
                {"decision": "technical_invalid", "candidate_id": "D01__candidate_01"},
                recorded_at_utc="2026-09-12T00:01:00Z",
            )
            self.assertEqual(second["previous_record_sha256"], first["record_sha256"])
            self.assertEqual(len(gate.read_ledger(path)), 2)
            path.write_bytes(path.read_bytes().replace(b"technical_invalid", b"physical_rejectx", 1))
            with self.assertRaises(fixture.LayoutContractError):
                gate.read_ledger(path)

    def test_manifest_requires_exactly_one_accepted_record_and_stays_unreleased(self):
        accepted = {
            "layout_pair_id": self.candidate["layout_pair_id"],
            "candidate_id": self.candidate["candidate_id"],
            "candidate_payload_sha256": self.candidate["candidate_payload_sha256"],
            "candidate_pool_sha256": self.pool_sha,
            "decision": "accepted",
            "passed": True,
            "record_sha256": "a" * 64,
            "attempt_receipt": {"path": "/data/attempt.json", "sha256": "b" * 64, "bytes": 1},
        }
        manifest = gate.build_frozen_pose_manifest(
            pool=self.pool,
            candidate_pool_sha256=self.pool_sha,
            records=[accepted],
            gate_ledger_sha256="c" * 64,
            layout_pair_ids=[self.candidate["layout_pair_id"]],
        )
        self.assertTrue(manifest["physical_layout_gate_passed"])
        self.assertFalse(manifest["released_for_model_inference"])
        self.assertEqual(manifest["task_contract"]["termination_terms"], ["time_out"])
        with self.assertRaisesRegex(fixture.LayoutContractError, "exactly one"):
            gate.build_frozen_pose_manifest(
                pool=self.pool,
                candidate_pool_sha256=self.pool_sha,
                records=[],
                gate_ledger_sha256="c" * 64,
                layout_pair_ids=[self.candidate["layout_pair_id"]],
            )


class TimeoutTaskTests(unittest.TestCase):
    def fake_runtime_modules(self):
        modules = {}
        for name in (
            "isaaclab", "isaaclab.envs", "isaaclab.envs.mdp", "isaaclab.managers", "isaaclab.utils",
            "robolab", "robolab.core", "robolab.core.scenes", "robolab.core.scenes.utils",
            "robolab.core.task", "robolab.core.task.task",
        ):
            modules[name] = types.ModuleType(name)
        modules["isaaclab.envs.mdp"].time_out = lambda env: False

        class DoneTerm:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        modules["isaaclab.managers"].TerminationTermCfg = DoneTerm
        modules["isaaclab.utils"].configclass = lambda cls: cls

        class Init:
            pos = None
            rot = None
            lin_vel = None
            ang_vel = None

        class Asset:
            def __init__(self):
                self.init_state = Init()

        def import_scene(_name, names):
            return types.SimpleNamespace(**{name: Asset() for name in names})

        modules["robolab.core.scenes.utils"].import_scene = import_scene
        modules["robolab.core.task.task"].Task = object
        return modules

    def test_runtime_task_has_only_timeout_and_consumes_frozen_manifest_hash(self):
        pool, _, _ = build_pool()
        candidate = pool["candidates"][0]
        pool_sha = fixture.sha256_bytes(fixture.canonical_json_bytes(pool))
        accepted = {
            "layout_pair_id": "P00",
            "candidate_id": candidate["candidate_id"],
            "candidate_payload_sha256": candidate["candidate_payload_sha256"],
            "candidate_pool_sha256": pool_sha,
            "decision": "accepted",
            "passed": True,
            "record_sha256": "a" * 64,
            "attempt_receipt": {"path": "/data/receipt", "sha256": "b" * 64, "bytes": 1},
        }
        manifest = gate.build_frozen_pose_manifest(
            pool=pool,
            candidate_pool_sha256=pool_sha,
            records=[accepted],
            gate_ledger_sha256="c" * 64,
            layout_pair_ids=["P00"],
        )
        payload = fixture.canonical_json_bytes(manifest)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses.json"
            path.write_bytes(payload)
            with patch.dict(sys.modules, self.fake_runtime_modules()):
                task_class = tasks.build_timeout_only_task_class(
                    manifest_path=path,
                    manifest_sha256=fixture.sha256_bytes(payload),
                    layout_pair_id="P00",
                    layout_arm="reflected",
                    command="left",
                )
        self.assertTrue(hasattr(task_class.terminations, "time_out"))
        self.assertFalse(hasattr(task_class.terminations, "success"))
        self.assertEqual(task_class.wmf_action_cap, 450)
        self.assertFalse(task_class.wmf_stop_on_success)
        self.assertFalse(task_class.wmf_fixture_binding["released_for_model_inference"])
        self.assertEqual(task_class.scene.banana.init_state.pos, tuple(candidate["layouts"]["reflected"]["positions_robot_base_m"]["banana"]))

    def test_manifest_digest_tampering_fails_before_simulator_import(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text("{}\n")
            with self.assertRaisesRegex(fixture.LayoutContractError, "SHA-256 mismatch"):
                tasks.build_timeout_only_task_class(
                    manifest_path=path,
                    manifest_sha256="0" * 64,
                    layout_pair_id="P00",
                    layout_arm="original",
                    command="left",
                )


if __name__ == "__main__":
    unittest.main()
