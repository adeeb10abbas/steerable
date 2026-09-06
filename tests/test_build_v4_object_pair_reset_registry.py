"""Tests for the prospective V4 C7 sponge/tray reset registry builder."""

from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "tools/build_v4_object_pair_reset_registry.py"
    spec = importlib.util.spec_from_file_location("build_v4_object_pair_reset_registry", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


builder = _load_module()


class ObjectPairResetRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.campaign = ROOT / "docs/online_correction_v4/campaign.json"
        self.queue = ROOT / "artifacts/online_correction_v4/queue.jsonl"
        self.scene = (
            ROOT
            / "experiments/online_correction_v4/droid_task_files/scene_assets/"
            "sponge_tray_object_pair.usda"
        )

    def test_frozen_object_pair_registry_has_64_registered_seeds(self):
        payload = builder.build_registry(
            campaign_path=self.campaign,
            queue_path=self.queue,
            scene_path=self.scene,
        )
        self.assertEqual(payload["fixture_id"], "object_pair")
        self.assertEqual(payload["registered_env_seed_count"], 64)
        self.assertEqual(payload["registered_env_seed_min"], 2100040000)
        self.assertEqual(payload["registered_env_seed_max"], 2100040063)
        self.assertEqual(payload["model_request_count"], 0)
        self.assertEqual(payload["behavioral_episode_count"], 0)
        self.assertEqual(set(payload["object_roles"]), {"target", "reference"})
        self.assertEqual(
            payload["scene_metadata_sha256"],
            builder.sha256_file(self.scene),
        )

    def test_common_jitter_preserves_sponge_tray_relative_geometry(self):
        payload = builder.build_registry(
            campaign_path=self.campaign,
            queue_path=self.queue,
            scene_path=self.scene,
        )
        expected = [
            builder.OBJECT_SPECS["tray"]["base_position_robot_m"][axis]
            - builder.OBJECT_SPECS["sponge"]["base_position_robot_m"][axis]
            for axis in range(3)
        ]
        for reset in payload["resets_by_env_seed"].values():
            positions = reset["positions_robot_base_m"]
            observed = [
                positions["tray"][axis] - positions["sponge"][axis]
                for axis in range(3)
            ]
            for actual, wanted in zip(observed, expected):
                self.assertAlmostEqual(actual, wanted)

    def test_build_is_deterministic(self):
        first = builder.build_registry(
            campaign_path=self.campaign,
            queue_path=self.queue,
            scene_path=self.scene,
        )
        second = builder.build_registry(
            campaign_path=self.campaign,
            queue_path=self.queue,
            scene_path=self.scene,
        )
        self.assertEqual(
            builder.canonical_json_bytes(first),
            builder.canonical_json_bytes(second),
        )

    def test_queue_seed_mutation_is_rejected(self):
        rows = [
            json.loads(line)
            for line in self.queue.read_text(encoding="utf-8").splitlines()
            if line
        ]
        row = next(item for item in rows if item["fixture"] == "object_pair")
        row["env_seed"] += 1
        with tempfile.TemporaryDirectory() as tmp:
            queue = Path(tmp) / "queue.jsonl"
            queue.write_text(
                "".join(
                    json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
                    for item in rows
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                builder.ObjectPairRegistryBuildError,
                "exactly one environment seed|frozen fixture namespace",
            ):
                builder.build_registry(
                    campaign_path=self.campaign,
                    queue_path=queue,
                    scene_path=self.scene,
                )

    def test_missing_scene_is_rejected(self):
        with self.assertRaisesRegex(
            builder.ObjectPairRegistryBuildError,
            "scene asset does not exist",
        ):
            builder.build_registry(
                campaign_path=self.campaign,
                queue_path=self.queue,
                scene_path=ROOT / "missing-object-pair.usda",
            )

    def test_object_specs_require_distinct_roles(self):
        original = copy.deepcopy(builder.OBJECT_SPECS)
        try:
            builder.OBJECT_SPECS["tray"]["role"] = "target"
            with self.assertRaisesRegex(
                builder.ObjectPairRegistryBuildError,
                "roles must be target/reference",
            ):
                builder.build_registry(
                    campaign_path=self.campaign,
                    queue_path=self.queue,
                    scene_path=self.scene,
                )
        finally:
            builder.OBJECT_SPECS.clear()
            builder.OBJECT_SPECS.update(original)


if __name__ == "__main__":
    unittest.main()
