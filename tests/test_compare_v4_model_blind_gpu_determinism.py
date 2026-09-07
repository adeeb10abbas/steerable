"""Tests for cross-GPU determinism comparison."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "compare_v4_model_blind_gpu_determinism",
        ROOT / "tools/compare_v4_model_blind_gpu_determinism.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


compare = _load()


def _receipt(gpu_name: str, *, banana: float = 0.001) -> dict:
    return {
        "environment_seed": 2100000000,
        "runtime_identity": {"gpu": {"name": gpu_name}},
        "reset_attestation": {
            "reset_registry_sha256": "392dcdfd797644f514b74cf82f65b3a69d7ba1120e959b55e0447df18d61ab13"
        },
        "position_max_error_m_by_object": {
            "banana": banana,
            "bowl": 0.001,
        },
    }


class CompareGpuDeterminismTests(unittest.TestCase):
    def test_passes_within_tolerance(self) -> None:
        result = compare.compare_receipts(
            baseline=_receipt("NVIDIA A40"),
            candidate=_receipt("NVIDIA A100-SXM4-80GB", banana=0.0015),
        )
        self.assertTrue(result["passed_within_tolerance"])

    def test_rejects_registry_mismatch(self) -> None:
        candidate = _receipt("NVIDIA A100-SXM4-80GB")
        candidate["reset_attestation"]["reset_registry_sha256"] = "deadbeef"
        with self.assertRaises(ValueError):
            compare.compare_receipts(
                baseline=_receipt("NVIDIA A40"),
                candidate=candidate,
            )


if __name__ == "__main__":
    unittest.main()
