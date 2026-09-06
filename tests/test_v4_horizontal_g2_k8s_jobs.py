"""Tests for simulator-only V4 horizontal G2 Kubernetes rendering."""

from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


renderer = _load(
    "render_v4_horizontal_g2_k8s_jobs",
    "tools/render_v4_horizontal_g2_k8s_jobs.py",
)
validator = _load(
    "validate_v4_horizontal_g2_k8s_jobs",
    "tools/validate_v4_horizontal_g2_k8s_jobs.py",
)
OBJECT_PAIR_SPEC = (
    ROOT / "deploy/k8s/v4_lane_bundle/g2-object-pair-spec.example.json"
)


@unittest.skipUnless(shutil.which("kubectl"), "kubectl is required")
class HorizontalG2K8sTests(unittest.TestCase):
    def test_default_spec_renders_complete_simulator_only_seed_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = renderer.render(renderer.DEFAULT_SPEC, Path(tmp))
            root = Path(report["bundle_root"])
            validated = validator.validate(root)
        self.assertTrue(validated["ok"])
        self.assertEqual(validated["job_count"], 128)
        self.assertEqual(validated["environment_seed_count"], 128)
        self.assertEqual(validated["model_request_count"], 0)
        self.assertEqual(validated["behavioral_episode_count"], 0)

    def test_renderer_rejects_partial_seed_limit(self) -> None:
        spec = json.loads(renderer.DEFAULT_SPEC.read_text(encoding="utf-8"))
        spec["max_seed_jobs"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            spec_path = Path(tmp) / "spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaises(renderer.G2RenderError):
                renderer.render(spec_path, Path(tmp) / "out")

    def test_object_pair_spec_renders_64_model_blind_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = renderer.render(OBJECT_PAIR_SPEC, Path(tmp))
            validated = validator.validate(Path(report["bundle_root"]))
        self.assertTrue(validated["ok"])
        self.assertEqual(validated["fixture_id"], "object_pair")
        self.assertEqual(validated["job_count"], 64)
        self.assertEqual(validated["environment_seed_count"], 64)
        self.assertEqual(validated["model_request_count"], 0)
        self.assertEqual(validated["behavioral_episode_count"], 0)


if __name__ == "__main__":
    unittest.main()
