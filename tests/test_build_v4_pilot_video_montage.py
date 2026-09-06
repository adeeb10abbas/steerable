from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.build_v4_pilot_video_montage import (
    discover_attempts,
    parse_selections,
)


class PilotVideoMontageTests(unittest.TestCase):
    def test_selection_parser_is_fail_closed(self) -> None:
        self.assertEqual(
            parse_selections(["g7c7p00=attempt0001"]),
            {"g7c7p00": ["attempt0001"]},
        )
        self.assertEqual(
            parse_selections(["v4r00=attempt0103"]),
            {"v4r00": ["attempt0103"]},
        )
        self.assertEqual(
            parse_selections(
                [
                    "g7c7p00=attempt0001",
                    "g7c7p00=attempt0002",
                ]
            ),
            {"g7c7p00": ["attempt0001", "attempt0002"]},
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_selections(["g7c7p00=attempt0001"] * 2)

    def test_discovery_requires_exact_queue_coverage(self) -> None:
        episode_id = "online_correction_v4-C7-pilot-00-example"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempt = (
                root
                / "uuid"
                / "lane-g7c7p00"
                / "episodes"
                / episode_id
                / "attempt0001"
            )
            attempt.mkdir(parents=True)
            (attempt / "COMPLETE.json").write_text(
                json.dumps(
                    {
                        "episode_id": episode_id,
                        "attempt_id": "attempt0001",
                        "status": "valid",
                    }
                ),
                encoding="utf-8",
            )
            discovered = discover_attempts(
                raw_root=root,
                queue={episode_id: {"episode_id": episode_id}},
                selections={"g7c7p00": ["attempt0001"]},
            )
            self.assertEqual(discovered, {episode_id: attempt})
            with self.assertRaisesRegex(ValueError, "missing"):
                discover_attempts(
                    raw_root=root,
                    queue={
                        episode_id: {"episode_id": episode_id},
                        "missing": {"episode_id": "missing"},
                    },
                    selections={"g7c7p00": ["attempt0001"]},
                )


if __name__ == "__main__":
    unittest.main()
