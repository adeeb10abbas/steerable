from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from experiments.online_correction_v4.droid_robolab import (
    LiveRoboLabBackend,
    write_queue_row,
)


class DroidRoboLabTests(unittest.TestCase):
    def test_queue_row_creates_fresh_episode_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "new" / "episode"
            path, digest = write_queue_row(
                output_dir=output,
                episode_id="episode-1",
                fixture_id="object_pair",
                prompt_text="Place the sponge.",
                prompt_sha256="a" * 64,
                env_seed=1,
                goal="left",
            )
            self.assertTrue(path.is_file())
            self.assertEqual(len(digest), 64)

    @unittest.skipIf(np is None, "numpy is owned by the cluster runtime")
    def test_viewport_capture_falls_back_to_observation_group(self) -> None:
        backend = object.__new__(LiveRoboLabBackend)
        backend.env = SimpleNamespace(viewport_recorder=None)
        backend._latest_raw_obs = {
            "viewport_cam": {
                "egocentric_mirrored_camera": np.array(
                    [[[[0, 1, 2], [3, 4, 5]]]], dtype=np.uint8
                )
            }
        }
        capture = backend.capture_viewport_frame()
        self.assertEqual(capture.format_kind, "raw_rgb24")
        self.assertEqual((capture.width, capture.height), (2, 1))


if __name__ == "__main__":
    unittest.main()
