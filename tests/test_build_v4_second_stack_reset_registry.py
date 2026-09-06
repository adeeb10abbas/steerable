from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.build_v4_second_stack_reset_registry import (
    BASE_POSITIONS_SCENE_XY_M,
    CHECKPOINT_REVISION,
    GR00T_COMMIT,
    REFERENCE_OBJECT,
    SIMPLER_ENV_COMMIT,
    SOURCE_OBJECT,
    build_registry,
    canonical_json_bytes,
)


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "docs/online_correction_v4/campaign.json"
QUEUE = ROOT / "artifacts/online_correction_v4/queue.jsonl"
CANDIDATE = (
    ROOT
    / "artifacts/online_correction_v4/setup"
    / "second_stack_reset_registry.candidate.json"
)


class SecondStackResetRegistryTests(unittest.TestCase):
    def test_builder_reproduces_committed_candidate(self) -> None:
        payload = build_registry(campaign_path=CAMPAIGN, queue_path=QUEUE)
        self.assertEqual(CANDIDATE.read_bytes(), canonical_json_bytes(payload))

    def test_candidate_binds_official_pinned_stack(self) -> None:
        payload = json.loads(CANDIDATE.read_text(encoding="utf-8"))
        identity = payload["external_stack_identity"]
        self.assertEqual(identity["gr00t_commit"], GR00T_COMMIT)
        self.assertEqual(identity["simpler_env_commit"], SIMPLER_ENV_COMMIT)
        self.assertEqual(identity["checkpoint_revision"], CHECKPOINT_REVISION)
        self.assertEqual(identity["embodiment_tag"], "SIMPLER_ENV_WIDOWX")
        self.assertEqual(
            payload["environment_name"],
            "simpler_env_widowx/widowx_stack_cube",
        )
        self.assertEqual(payload["native_control_dt_s"], 0.2)

    def test_resets_are_unique_balanced_and_preserve_pair_geometry(self) -> None:
        payload = json.loads(CANDIDATE.read_text(encoding="utf-8"))
        resets = payload["resets_by_env_seed"]
        self.assertEqual(len(resets), 64)
        self.assertEqual(
            sorted(map(int, resets)),
            list(range(2_100_050_000, 2_100_050_064)),
        )
        self.assertEqual(
            {
                sum(
                    row["physical_translation_sign"] == sign
                    for row in resets.values()
                )
                for sign in (-1, 1)
            },
            {32},
        )
        source_base = BASE_POSITIONS_SCENE_XY_M[SOURCE_OBJECT]
        reference_base = BASE_POSITIONS_SCENE_XY_M[REFERENCE_OBJECT]
        expected_delta = tuple(
            reference_base[index] - source_base[index] for index in range(2)
        )
        observed_jitter = set()
        for row in resets.values():
            positions = row["positions_scene_xy_m"]
            observed_jitter.add(tuple(row["jitter_scene_xy_m"]))
            delta = tuple(
                positions[REFERENCE_OBJECT][index]
                - positions[SOURCE_OBJECT][index]
                for index in range(2)
            )
            for actual, expected in zip(delta, expected_delta):
                self.assertAlmostEqual(actual, expected)
        self.assertEqual(len(observed_jitter), 64)

    def test_source_hashes_do_not_depend_on_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            alternate = build_registry(
                campaign_path=Path(tmp) / ".." / Path(CAMPAIGN),
                queue_path=Path(tmp) / ".." / Path(QUEUE),
            )
        committed = json.loads(CANDIDATE.read_text(encoding="utf-8"))
        self.assertEqual(
            alternate["external_stack_identity"],
            committed["external_stack_identity"],
        )


if __name__ == "__main__":
    unittest.main()
