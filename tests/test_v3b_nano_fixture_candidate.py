from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.v3.cosmos_nano_phase_b import fixture_candidate as fixture


TASK = """
class Task:
    scene.rubiks_cube.init_state.pos = (0.30, 0.12, 0.08)
"""


class NanoPhaseBFixtureCandidateTest(unittest.TestCase):
    def build(self, root: Path) -> dict:
        metadata = root / "scene_metadata.json"
        left = root / "left.py"
        right = root / "right.py"
        metadata.write_text(json.dumps({
            fixture.SCENE_NAME: [
                {"name": "rubiks_cube", "position": [0.43, -0.10, 0.08], "rotation": [1, 0, 0, 0]},
                {"name": "bowl", "position": [0.44, 0.13, 0.07], "rotation": [1, 0, 0, 0]},
                {"name": "banana", "position": [0.54, -0.08, 0.07], "rotation": [1, 0, 0, 0]},
            ]
        }))
        left.write_text(TASK)
        right.write_text(TASK)
        return fixture.derive_candidate(
            scene_metadata_path=metadata,
            neutral_left_task_path=left,
            neutral_right_task_path=right,
            robolab_commit=fixture.ROBOLAB_COMMIT,
        )

    def test_positions_only_are_reflected_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            candidate = self.build(Path(temporary))
        control = candidate["layouts"]["control"]
        mirrored = candidate["layouts"]["position_mirrored"]
        for name in fixture.MOVABLE_OBJECTS:
            c = control["positions_robot_base_m"][name]
            m = mirrored["positions_robot_base_m"][name]
            self.assertEqual([c[0], -c[1], c[2]], m)
            self.assertEqual(
                control["quaternions_wxyz_unchanged"][name],
                mirrored["quaternions_wxyz_unchanged"][name],
            )
        self.assertEqual(candidate["model_request_count"], 0)
        self.assertEqual(candidate["behavioral_episode_count"], 0)
        self.assertIn("not a full geometric reflection", candidate["factor"]["claim_boundary"])

    def test_left_and_right_tasks_must_share_one_reset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "scene_metadata.json"
            left = root / "left.py"
            right = root / "right.py"
            metadata.write_text(json.dumps({
                fixture.SCENE_NAME: [
                    {"name": name, "position": [0.4, 0.1, 0.08], "rotation": [1, 0, 0, 0]}
                    for name in fixture.MOVABLE_OBJECTS
                ]
            }))
            left.write_text(TASK)
            right.write_text(TASK.replace("0.12", "0.11"))
            with self.assertRaisesRegex(fixture.FixtureCandidateError, "do not share"):
                fixture.derive_candidate(
                    scene_metadata_path=metadata,
                    neutral_left_task_path=left,
                    neutral_right_task_path=right,
                    robolab_commit=fixture.ROBOLAB_COMMIT,
                )


if __name__ == "__main__":
    unittest.main()
