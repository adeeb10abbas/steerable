from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from experiments.online_correction_v4.droid_task_files.constants import (
    CONTAINMENT_SCENE_METADATA_SHA256,
    CONTAINMENT_SCENE_PATH,
    VERTICAL_SCENE_METADATA_SHA256,
    VERTICAL_SCENE_PATH,
)
from experiments.online_correction_v4.droid_task_files.registry import (
    FixtureRegistryError,
    resolve_active_registration,
)
from experiments.online_correction_v4.droid_task_files.reset_registry import (
    load_reset_registry,
)
from tools.build_v4_c5_c6_reset_registries import (
    build_registry,
    canonical_json_bytes,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "docs/online_correction_v4/campaign.json"
QUEUE = ROOT / "artifacts/online_correction_v4/queue.jsonl"
SETUP = ROOT / "artifacts/online_correction_v4/setup"


class C5C6ResetRegistryTests(unittest.TestCase):
    def test_committed_candidates_rebuild_exactly(self) -> None:
        expected = {
            "vertical": (
                Path(VERTICAL_SCENE_PATH),
                VERTICAL_SCENE_METADATA_SHA256,
                2100020000,
                2100020063,
            ),
            "containment": (
                Path(CONTAINMENT_SCENE_PATH),
                CONTAINMENT_SCENE_METADATA_SHA256,
                2100030000,
                2100030063,
            ),
        }
        for fixture_id, (scene, scene_sha, seed_min, seed_max) in expected.items():
            with self.subTest(fixture_id=fixture_id):
                payload = build_registry(
                    fixture_id=fixture_id,
                    campaign_path=CAMPAIGN,
                    queue_path=QUEUE,
                )
                committed = SETUP / f"{fixture_id}_reset_registry.candidate.json"
                self.assertEqual(canonical_json_bytes(payload), committed.read_bytes())
                self.assertEqual(sha256_file(scene), scene_sha)
                self.assertEqual(payload["registered_env_seed_count"], 64)
                self.assertEqual(payload["registered_env_seed_min"], seed_min)
                self.assertEqual(payload["registered_env_seed_max"], seed_max)

    def test_candidates_load_under_fixture_specific_schema(self) -> None:
        for fixture_id in ("vertical", "containment"):
            with self.subTest(fixture_id=fixture_id):
                path = SETUP / f"{fixture_id}_reset_registry.candidate.json"
                registry = load_reset_registry(
                    registry_path=str(path),
                    registry_sha256=sha256_file(path),
                    expected_fixture_id=fixture_id,
                )
                self.assertEqual(registry.fixture_id, fixture_id)
                self.assertEqual(set(registry.object_roles), {"target", "reference"})
                self.assertEqual(len(registry.positions_by_env_seed), 64)

    def test_active_registration_is_model_blind_only_before_g2_g3(self) -> None:
        for fixture_id in ("vertical", "containment"):
            with self.subTest(fixture_id=fixture_id):
                with self.assertRaises(FixtureRegistryError):
                    resolve_active_registration(fixture_id)
                registration = resolve_active_registration(
                    fixture_id,
                    allow_model_blind_candidate=True,
                )
                self.assertEqual(registration.fixture_id, fixture_id)
                self.assertIn("model_blind_candidate", registration.attributes)
                self.assertTrue(Path(registration.scene_asset).is_file())

    def test_registry_rejects_scene_digest_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "scene.usda"
            scene.write_text("#usda 1.0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scene digest differs"):
                build_registry(
                    fixture_id="vertical",
                    campaign_path=CAMPAIGN,
                    queue_path=QUEUE,
                    scene_path=scene,
                )


if __name__ == "__main__":
    unittest.main()
