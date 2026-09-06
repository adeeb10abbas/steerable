from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.build_v4_object_pair_g7_pilot_release import (
    build_nano_seed_registry,
    pilot_rows,
    release_pilot_resets,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "artifacts/online_correction_v4/setup"
QUALIFICATION = ROOT / "artifacts/online_correction_v4/qualification"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class ObjectPairG7PilotReleaseTests(unittest.TestCase):
    def test_builds_frozen_sixteen_static_eight_motion_manifest(self) -> None:
        source = SETUP / "object_pair_pilot_seed_registry.candidate.json"
        rows = pilot_rows(
            pilot_seed_registry=load(source),
            config_sha256=sha256_file(
                ROOT / "docs/online_correction_v4/campaign.json"
            ),
        )
        self.assertEqual(len(rows), 24)
        self.assertEqual(
            sum(
                row["counterbalance"]["pilot_kind"] == "stationary"
                for row in rows
            ),
            16,
        )
        self.assertEqual(
            sum(
                row["counterbalance"]["pilot_kind"] == "motion"
                for row in rows
            ),
            8,
        )
        self.assertEqual({row["cohort"] for row in rows}, {"engineering_pilot"})
        self.assertEqual({row["family"] for row in rows}, {"C7"})
        self.assertEqual(len({row["env_seed"] for row in rows}), 24)
        self.assertEqual(len({row["policy_seed"] for row in rows}), 24)
        execution_groups = {row["execution_group"] for row in rows}
        self.assertEqual(len(execution_groups), 8)
        self.assertEqual(
            {
                sum(row["execution_group"] == group_id for row in rows)
                for group_id in execution_groups
            },
            {3},
        )
        registry = build_nano_seed_registry(rows=rows, source_path=source)
        self.assertEqual(registry["scope"], "g7_engineering_pilot")
        self.assertEqual(len(registry["allowed_sampling_seeds"]), 24)

    def test_releases_only_complete_pilot_g2_g3_reset_set(self) -> None:
        candidate_path = SETUP / "object_pair_pilot_reset_registry.candidate.json"
        g2_path = (
            QUALIFICATION
            / "20260906_object_pair_pilot_g2_aggregate_g2c7pilot20260906b.json"
        )
        with tempfile.TemporaryDirectory() as tmp:
            g3_path = Path(tmp) / "pilot-g3.json"
            g3_path.write_text("{}", encoding="utf-8")
            released = release_pilot_resets(
                candidate=load(candidate_path),
                candidate_path=candidate_path,
                pilot_g2=load(g2_path),
                pilot_g2_path=g2_path,
                pilot_g3={
                    "fixture_id": "object_pair",
                    "status": "passed",
                    "passed": True,
                    "qualification_scope": "engineering_pilot",
                    "expected_scripted_check_count": 112,
                    "observed_scripted_check_count": 112,
                },
                pilot_g3_path=g3_path,
            )
        self.assertEqual(released["status"], "released_for_policy_inference")
        self.assertEqual(released["registered_env_seed_count"], 24)
        self.assertIn("pilot_g3", released["qualification_release_basis"])


if __name__ == "__main__":
    unittest.main()
