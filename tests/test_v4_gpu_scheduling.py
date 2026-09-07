"""Tests for shared V4 GPU scheduling helpers."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "v4_gpu_scheduling", ROOT / "tools/v4_gpu_scheduling.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


gpu = _load()


class GpuSchedulingTests(unittest.TestCase):
    def test_default_model_blind_resolution(self) -> None:
        resolved = gpu.resolve_model_blind_scheduling({"gpu_product": "NVIDIA-A40"})
        self.assertEqual(resolved["gpu_product_allowlist"], ["NVIDIA-A40"])

    def test_allowlist_emits_affinity_yaml(self) -> None:
        rows = gpu.render_pod_gpu_scheduling_yaml(
            gpu_product="NVIDIA-A40",
            gpu_product_allowlist=[
                "NVIDIA-A40",
                "NVIDIA-A100-SXM4-80GB",
            ],
        )
        joined = "\n".join(rows)
        self.assertIn("nodeAffinity", joined)
        self.assertIn("NVIDIA-A100-SXM4-80GB", joined)

    def test_validate_single_product_selector(self) -> None:
        gpu.validate_pod_gpu_scheduling(
            {
                "nodeSelector": {
                    "node-role.kubernetes.io/worker-gpu": "",
                    "nvidia.com/gpu.product": "NVIDIA-A100-SXM4-80GB",
                }
            },
            gpu_product="NVIDIA-A100-SXM4-80GB",
            gpu_product_allowlist=["NVIDIA-A100-SXM4-80GB"],
        )


if __name__ == "__main__":
    unittest.main()
