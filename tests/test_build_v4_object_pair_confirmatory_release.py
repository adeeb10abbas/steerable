from __future__ import annotations

from pathlib import Path
import unittest

from tools.build_v4_object_pair_confirmatory_release import (
    ROOT,
    build_confirmatory_seed_registry,
)


class ObjectPairConfirmatoryReleaseTests(unittest.TestCase):
    def test_main_c7_seeds_are_complete_and_disjoint_from_pilot(self) -> None:
        registry = build_confirmatory_seed_registry(
            queue_path=ROOT / "artifacts/online_correction_v4/queue.jsonl",
            pilot_seed_registry_path=(
                ROOT
                / "artifacts/online_correction_v4/setup/"
                "object_pair_g7_nano_seed_registry.released.json"
            ),
        )
        seeds = registry["allowed_sampling_seeds"]
        self.assertEqual(len(seeds), 64)
        self.assertEqual(len(set(seeds)), 64)
        self.assertEqual(
            registry["pilot_collision_audit"]["collision_count"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
