from __future__ import annotations

from pathlib import Path
import unittest

from tools.build_v4_object_pair_pilot_registries import build_payloads


ROOT = Path(__file__).resolve().parents[1]


class ObjectPairPilotRegistryTests(unittest.TestCase):
    def test_builds_disjoint_24_seed_pilot_registries(self) -> None:
        reset, policy = build_payloads(
            seed_manifest_path=ROOT / "artifacts/online_correction_v4/seed_manifest.json",
            base_reset_registry_path=(
                ROOT
                / "artifacts/online_correction_v4/setup/"
                "object_pair_reset_registry.candidate.json"
            ),
        )
        self.assertEqual(reset["registered_env_seed_count"], 24)
        self.assertEqual(reset["registered_env_seed_min"], 2110000800)
        self.assertEqual(reset["registered_env_seed_max"], 2110000823)
        self.assertEqual(reset["qualification_scope"], "engineering_pilot")
        self.assertEqual(len(policy["rows"]), 24)
        self.assertEqual(policy["rows"][0]["policy_seed"], 1392709708)
        self.assertTrue(policy["confirmatory_env_seed_disjoint"])
        self.assertTrue(policy["confirmatory_policy_seed_disjoint"])


if __name__ == "__main__":
    unittest.main()
