"""Tests for V4 DROID fixture registry and instruction binding."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from experiments.online_correction_v4.droid_task_files.binding import (
    InstructionBindingError,
    load_bound_instruction,
    sha256_bytes,
    sha256_file,
)
from experiments.online_correction_v4.droid_task_files.constants import (
    EPISODE_LENGTH_S,
    HORIZONTAL_RELATIONS,
    SCENE_ASSET,
    SCENE_METADATA_SHA256,
)
from experiments.online_correction_v4.droid_task_files.registry import (
    FixtureRegistryError,
    blocked_fixture_ids,
    iter_horizontal_registrations,
    list_registered_horizontal_relations,
    resolve_fixture_registration,
    supported_fixture_ids,
)
from experiments.online_correction_v4.droid_task_files.reset_registry import (
    ResetRegistryError,
    load_reset_registry,
)


ROOT = Path(__file__).resolve().parents[1]


def _sample_queue_row(*, goal: str = "left", env_seed: int = 2100000045) -> dict:
    prompt_text = (
        "Place the the cube so that the the cube is left of the the bowl. "
        "Use the robot's fixed viewpoint for left, right, front, and behind."
    )
    if goal == "right":
        prompt_text = (
            "Place the the cube so that the the cube is right of the the bowl. "
            "Use the robot's fixed viewpoint for left, right, front, and behind."
        )
    elif goal == "front":
        prompt_text = (
            "Place the the cube so that the the cube is in front of the the bowl. "
            "Use the robot's fixed viewpoint for left, right, front, and behind."
        )
    elif goal == "behind":
        prompt_text = (
            "Place the the cube so that the the bowl is in front of the the cube. "
            "Use the robot's fixed viewpoint for left, right, front, and behind."
        )
    prompt_sha256 = sha256_bytes(prompt_text.encode("utf-8"))
    return {
        "campaign": "online_correction_v4",
        "episode_id": f"online_correction_v4-C1-b045-test-{goal}",
        "fixture": "horizontal",
        "env_seed": env_seed,
        "prompt_text": prompt_text,
        "prompt_sha256": prompt_sha256,
        "factors": {"goal": goal, "policy": "cosmos3_nano_droid", "wording": "direct"},
    }


def _sample_reset_registry(*, env_seed: int = 2100000045) -> dict:
    asset = lambda name: f"{SCENE_ASSET}::{name}@{SCENE_METADATA_SHA256}"
    return {
        "schema_version": "v4-droid-horizontal-reset-registry-v1",
        "fixture_id": "horizontal",
        "status": "model_blind_candidate_not_released_for_inference",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "scene_asset": SCENE_ASSET,
        "scene_metadata_sha256": SCENE_METADATA_SHA256,
        "contact_objects": ["rubiks_cube", "banana", "bowl", "table"],
        "object_roles": {
            "target": {"scene_object": "rubiks_cube", "asset_identity": asset("rubiks_cube")},
            "reference": {"scene_object": "bowl", "asset_identity": asset("bowl")},
            "distractor": {"scene_object": "banana", "asset_identity": asset("banana")},
        },
        "resets_by_env_seed": {
            str(env_seed): {
                "positions_robot_base_m": {
                    "rubiks_cube": [0.45, 0.0, 0.83],
                    "bowl": [0.55, 0.12, 0.82],
                    "banana": [0.40, -0.10, 0.81],
                }
            }
        },
    }


class FixtureRegistryTests(unittest.TestCase):
    def test_supported_fixture_is_horizontal_only(self) -> None:
        self.assertEqual(supported_fixture_ids(), ("horizontal",))

    def test_all_horizontal_relations_registered(self) -> None:
        self.assertEqual(list_registered_horizontal_relations(), HORIZONTAL_RELATIONS)
        registrations = list(iter_horizontal_registrations())
        self.assertEqual(len(registrations), 4)
        self.assertEqual({item.relation for item in registrations}, set(HORIZONTAL_RELATIONS))

    def test_horizontal_registration_is_timeout_only_with_external_scorer(self) -> None:
        reg = resolve_fixture_registration("horizontal", relation="left")
        self.assertEqual(reg.fixture_id, "horizontal")
        self.assertEqual(reg.relation, "left")
        self.assertEqual(reg.scene_asset, SCENE_ASSET)
        self.assertEqual(reg.scene_metadata_sha256, SCENE_METADATA_SHA256)
        self.assertEqual(reg.target_object, "rubiks_cube")
        self.assertEqual(reg.reference_object, "bowl")
        self.assertEqual(reg.episode_length_s, EPISODE_LENGTH_S)
        self.assertTrue(reg.timeout_only)
        self.assertTrue(reg.robolab_success_termination_forbidden)
        self.assertEqual(reg.external_scorer_mode, "external_v4_first_placement")
        self.assertIn("external_v4_scorer", reg.attributes)
        self.assertIn("timeout_only", reg.attributes)
        self.assertTrue(reg.task_module.endswith("horizontal_left.py"))
        self.assertEqual(reg.task_class, "V4HorizontalLeftTask")

    def test_each_relation_has_distinct_task_module(self) -> None:
        modules = {
            reg.relation: Path(reg.task_module).name
            for reg in iter_horizontal_registrations()
        }
        self.assertEqual(
            modules,
            {
                "left": "horizontal_left.py",
                "right": "horizontal_right.py",
                "front": "horizontal_front.py",
                "behind": "horizontal_behind.py",
            },
        )

    def test_unsupported_fixtures_fail_closed(self) -> None:
        for fixture_id, reason in blocked_fixture_ids().items():
            with self.subTest(fixture_id=fixture_id):
                with self.assertRaises(FixtureRegistryError) as ctx:
                    resolve_fixture_registration(fixture_id)
                self.assertIn(reason, str(ctx.exception))

    def test_unknown_fixture_rejected(self) -> None:
        with self.assertRaises(FixtureRegistryError):
            resolve_fixture_registration("unknown_fixture")

    def test_horizontal_requires_relation(self) -> None:
        with self.assertRaises(FixtureRegistryError):
            resolve_fixture_registration("horizontal")

    def test_invalid_relation_rejected(self) -> None:
        with self.assertRaises(FixtureRegistryError):
            resolve_fixture_registration("horizontal", relation="above")


class InstructionBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_backup = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)

    def test_load_bound_instruction_from_queue_row_file(self) -> None:
        row = _sample_queue_row(goal="left")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(row, handle)
            path = Path(handle.name)
        digest = sha256_file(path)
        bound = load_bound_instruction(
            expected_fixture="horizontal",
            expected_goal="left",
            queue_row_path=str(path),
            queue_row_sha256=digest,
        )
        self.assertEqual(bound.prompt_text, row["prompt_text"])
        self.assertEqual(bound.prompt_sha256, row["prompt_sha256"])
        self.assertEqual(bound.env_seed, row["env_seed"])
        self.assertEqual(bound.instruction, {"default": row["prompt_text"]})

    def test_instruction_binding_requires_exact_digest(self) -> None:
        row = _sample_queue_row(goal="right")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(row, handle)
            path = Path(handle.name)
        with self.assertRaises(InstructionBindingError):
            load_bound_instruction(
                expected_fixture="horizontal",
                expected_goal="right",
                queue_row_path=str(path),
                queue_row_sha256="0" * 64,
            )

    def test_goal_mismatch_rejected(self) -> None:
        row = _sample_queue_row(goal="left")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(row, handle)
            path = Path(handle.name)
        digest = sha256_file(path)
        with self.assertRaises(InstructionBindingError):
            load_bound_instruction(
                expected_fixture="horizontal",
                expected_goal="behind",
                queue_row_path=str(path),
                queue_row_sha256=digest,
            )

    def test_missing_env_vars_fail_closed(self) -> None:
        os.environ.pop("ONLINE_CORRECTION_V4_QUEUE_ROW", None)
        os.environ.pop("ONLINE_CORRECTION_V4_QUEUE_ROW_SHA256", None)
        with self.assertRaises(InstructionBindingError):
            load_bound_instruction(expected_fixture="horizontal", expected_goal="left")


class ResetRegistryTests(unittest.TestCase):
    def test_reset_registry_requires_exact_scene_object_mapping(self) -> None:
        payload = _sample_reset_registry()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(payload, handle)
            path = Path(handle.name)
        digest = sha256_file(path)
        registry = load_reset_registry(registry_path=str(path), registry_sha256=digest)
        self.assertEqual(registry.scene_asset, SCENE_ASSET)
        self.assertEqual(registry.object_roles["target"].scene_object, "rubiks_cube")
        self.assertEqual(registry.object_roles["reference"].scene_object, "bowl")
        self.assertIn(2100000045, registry.positions_by_env_seed)

    def test_reset_registry_digest_mismatch_rejected(self) -> None:
        payload = _sample_reset_registry()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(payload, handle)
            path = Path(handle.name)
        with self.assertRaises(ResetRegistryError):
            load_reset_registry(registry_path=str(path), registry_sha256="1" * 64)

    def test_reset_registry_rejects_wrong_scene_asset(self) -> None:
        payload = _sample_reset_registry()
        payload["scene_asset"] = "other_scene.usda"
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(payload, handle)
            path = Path(handle.name)
        digest = sha256_file(path)
        with self.assertRaises(ResetRegistryError):
            load_reset_registry(registry_path=str(path), registry_sha256=digest)


if __name__ == "__main__":
    unittest.main()
