from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from experiments.online_correction_v4.droid_robolab import write_queue_row


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


if __name__ == "__main__":
    unittest.main()
