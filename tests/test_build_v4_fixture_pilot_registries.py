from __future__ import annotations

from pathlib import Path
import unittest

from tools.build_v4_fixture_pilot_registries import build_payloads


ROOT = Path(__file__).resolve().parents[1]


class FixturePilotRegistryTests(unittest.TestCase):
    def test_builds_containment_pilot_registries(self) -> None:
        reset, policy = build_payloads(
            fixture_id="containment",
            seed_manifest_path=ROOT / "artifacts/online_correction_v4/seed_manifest.json",
            base_reset_registry_path=(
                ROOT
                / "artifacts/online_correction_v4/setup/"
                "containment_reset_registry.candidate.json"
            ),
        )
        self.assertEqual(reset["registered_env_seed_count"], 24)
        self.assertEqual(reset["registered_env_seed_min"], 2110000600)
        self.assertEqual(reset["registered_env_seed_max"], 2110000623)
        self.assertEqual(len(policy["rows"]), 24)
        self.assertTrue(policy["confirmatory_env_seed_disjoint"])
        self.assertTrue(policy["confirmatory_policy_seed_disjoint"])


if __name__ == "__main__":
    unittest.main()
