import tempfile
import unittest
from pathlib import Path

from experiments.v3.pi05_stochastic_v3d001.sequential_supervisor import (
    _pair_paths,
    _process_state,
    _raw_paths,
    lane_progress,
)


class Cell:
    def __init__(self, cell_id: str, block_id: str) -> None:
        self.cell_id = cell_id
        self.block_id = block_id


class SequentialSupervisorTests(unittest.TestCase):
    def test_progress_requires_raw_and_pair_manifests(self) -> None:
        cells = [Cell("v3d001:pi05:env1:left:sample0", "v3d001:pi05:env1:sample0"),
                 Cell("v3d001:pi05:env1:right:sample0", "v3d001:pi05:env1:sample0")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in _raw_paths(root, cells, 4):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
                path.with_name(path.name + ".manifest.json").write_text("{}\n", encoding="utf-8")
            progress = lane_progress(root, cells, 4)
            self.assertEqual(progress["valid_cells"], 2)
            self.assertFalse(progress["complete"])
            pair = _pair_paths(root, cells, 4)[0]
            pair.parent.mkdir(parents=True, exist_ok=True)
            pair.write_text("{}\n", encoding="utf-8")
            pair.with_name(pair.name + ".manifest.json").write_text("{}\n", encoding="utf-8")
            self.assertTrue(lane_progress(root, cells, 4)["complete"])

    def test_current_process_is_visible(self) -> None:
        import os
        self.assertNotEqual(_process_state(os.getpid()), "absent")


if __name__ == "__main__":
    unittest.main()
