from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from experiments.online_correction_v4.droid_task_files.constants import (
    REFERENCE_BINDING_SCENE_METADATA_SHA256,
)
from experiments.online_correction_v4.droid_task_files.reset_registry import (
    MODEL_BLIND_CANDIDATE_STATUS,
    load_reset_registry,
)
from experiments.online_correction_v4.droid_task_files.registry import (
    FixtureRegistryError,
    resolve_active_registration,
)
from tools.build_v4_reference_binding_reset_registry import (
    OUTPUT,
    SCENE,
    build_registry,
    canonical_json_bytes,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "docs/online_correction_v4/campaign.json"
QUEUE = ROOT / "artifacts/online_correction_v4/queue.jsonl"


class ReferenceBindingResetRegistryTests(unittest.TestCase):
    def test_committed_candidate_rebuilds_exactly(self) -> None:
        payload = build_registry(
            campaign_path=CAMPAIGN,
            queue_path=QUEUE,
        )
        self.assertEqual(canonical_json_bytes(payload), OUTPUT.read_bytes())
        self.assertEqual(
            sha256_file(SCENE),
            REFERENCE_BINDING_SCENE_METADATA_SHA256,
        )
        self.assertEqual(payload["registered_env_seed_count"], 128)
        self.assertEqual(payload["registered_env_seed_min"], 2100010000)
        self.assertEqual(payload["registered_env_seed_max"], 2100010127)

    def test_candidate_loads_under_fixture_schema(self) -> None:
        registry = load_reset_registry(
            registry_path=str(OUTPUT),
            registry_sha256=sha256_file(OUTPUT),
            required_status=MODEL_BLIND_CANDIDATE_STATUS,
            expected_fixture_id="reference_binding",
        )
        self.assertEqual(
            set(registry.positions_by_env_seed),
            set(range(2100010000, 2100010128)),
        )
        self.assertEqual(
            set(next(iter(registry.positions_by_env_seed.values()))),
            {"cube", "blue_bowl", "yellow_bowl"},
        )

    def test_active_registration_is_model_blind_only_before_gates(self) -> None:
        with self.assertRaises(FixtureRegistryError):
            resolve_active_registration("reference_binding")
        registration = resolve_active_registration(
            "reference_binding",
            allow_model_blind_candidate=True,
        )
        self.assertEqual(registration.target_object, "cube")
        self.assertEqual(registration.reference_object, "blue_bowl")
        self.assertEqual(
            registration.contact_object_list,
            ("cube", "blue_bowl", "yellow_bowl", "table"),
        )
        self.assertIn("model_blind_candidate", registration.attributes)

    def test_counterbalance_resolves_physical_a_without_geometry_drift(self) -> None:
        payload = json.loads(OUTPUT.read_text(encoding="utf-8"))
        rows = payload["resets_by_env_seed"].values()
        self.assertEqual({row["physical_A_color"] for row in rows}, {"blue", "yellow"})
        self.assertEqual(
            {row["physical_A_start_side"] for row in rows},
            {"left", "right"},
        )
        self.assertEqual(
            {tuple(row["physical_A_diagonal_signs"]) for row in rows},
            {(1, 1), (1, -1), (-1, 1), (-1, -1)},
        )
        for row in rows:
            positions = row["positions_robot_base_m"]
            a_name = row["physical_A_scene_object"]
            b_name = row["physical_B_scene_object"]
            self.assertAlmostEqual(
                abs(positions[a_name][1] - positions[b_name][1]),
                0.36,
            )
            expected_sign = 1 if row["physical_A_start_side"] == "left" else -1
            self.assertEqual(
                1 if positions[a_name][1] > positions[b_name][1] else -1,
                expected_sign,
            )

    def test_scene_digest_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            changed = Path(tmp) / SCENE.name
            changed.write_bytes(SCENE.read_bytes() + b"\n")
            self.assertNotEqual(
                sha256_file(changed),
                REFERENCE_BINDING_SCENE_METADATA_SHA256,
            )


if __name__ == "__main__":
    unittest.main()
