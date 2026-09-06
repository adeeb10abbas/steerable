from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.build_v4_object_pair_release_artifacts import build_scoring_geometry


class ObjectPairReleaseArtifactTests(unittest.TestCase):
    def test_builds_registered_eroded_workspace_and_diagonal(self) -> None:
        scene = {
            "table_world_aabb_m": {
                "min_xyz": [0.2, -0.48, -0.65],
                "max_xyz": [0.9, 0.52, 0.05],
            },
            "objects": {
                "sponge": {
                    "world_aabb_m": {
                        "min_xyz": [0.39, -0.1275, 0.053],
                        "max_xyz": [0.47, -0.0725, 0.088],
                    }
                },
                "tray": {
                    "world_aabb_m": {
                        "min_xyz": [0.37, 0.08, 0.053],
                        "max_xyz": [0.51, 0.18, 0.071],
                    }
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "initial_scene.json"
            g2 = root / "g2.json"
            g3 = root / "g3.json"
            source.write_text(json.dumps(scene), encoding="utf-8")
            g2.write_text("{}", encoding="utf-8")
            g3.write_text("{}", encoding="utf-8")
            payload = build_scoring_geometry(
                initial_scene=scene,
                initial_scene_path=source,
                g2_path=g2,
                g3_path=g3,
            )

        geometry = payload["geometry"]
        self.assertAlmostEqual(geometry["workspace"]["x_min"], -0.4475)
        self.assertAlmostEqual(geometry["workspace"]["x_max"], 0.4875)
        self.assertAlmostEqual(geometry["workspace"]["y_min"], -0.855)
        self.assertAlmostEqual(geometry["workspace"]["y_max"], -0.245)
        self.assertAlmostEqual(geometry["target_footprint"]["half_left"], 0.0275)
        self.assertAlmostEqual(geometry["target_footprint"]["half_front"], 0.04)
        self.assertGreater(geometry["d_cap_m"], 1.11)
        self.assertEqual(payload["task_frame"]["u_left_world"], [0.0, 1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
