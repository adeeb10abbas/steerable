"""Tests for complete horizontal G2 evidence aggregation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from experiments.online_correction_v4.model_blind_g2 import (
    AXIS_REVIEW_SCHEMA,
    SEED_RECEIPT_SCHEMA,
    canonical_json_bytes,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = (
    ROOT
    / "artifacts/online_correction_v4/setup/horizontal_reset_registry.candidate.json"
)


def _load_compiler():
    spec = importlib.util.spec_from_file_location(
        "compile_v4_horizontal_g2_aggregate",
        ROOT / "tools/compile_v4_horizontal_g2_aggregate.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


compiler = _load_compiler()


class CompileHorizontalG2Tests(unittest.TestCase):
    def test_complete_receipt_set_and_axis_review_pass(self) -> None:
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        seeds = sorted(int(seed) for seed in registry["resets_by_env_seed"])
        runtime = {
            "study_checkout": {"commit": "a" * 40},
            "robolab_checkout": {"commit": "b" * 40},
            "gpu": {"name": "NVIDIA A40", "driver_version": "580.95.05"},
            "gate_entrypoint_sha256": "1" * 64,
            "gate_core_sha256": "2" * 64,
            "droid_robolab_sha256": "3" * 64,
            "reset_registry_sha256": sha256_file(REGISTRY),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            overlay = root / "axis.png"
            overlay.write_bytes(b"axis-overlay")
            first_receipt = None
            for seed in seeds:
                directory = root / f"seed-{seed}"
                directory.mkdir()
                receipt = {
                    "schema_version": SEED_RECEIPT_SCHEMA,
                    "environment_seed": seed,
                    "passed_reset_and_camera": True,
                    "model_request_count": 0,
                    "behavioral_episode_count": 0,
                    "reset_registry_sha256": sha256_file(REGISTRY),
                    "runtime_identity": runtime,
                    "artifacts": {},
                }
                if first_receipt is None:
                    receipt["artifacts"] = {
                        "axis_overlay_images": {
                            "montage": {
                                "path": str(overlay),
                                "sha256": sha256_file(overlay),
                                "bytes": overlay.stat().st_size,
                            }
                        }
                    }
                path = directory / "g2_seed_receipt.json"
                path.write_bytes(canonical_json_bytes(receipt))
                if first_receipt is None:
                    first_receipt = path
            assert first_receipt is not None
            axis_review = {
                "schema_version": AXIS_REVIEW_SCHEMA,
                "campaign_id": "online_correction_v4",
                "fixture_id": "horizontal",
                "status": "passed",
                "passed": True,
                "rendered_left_front_up": True,
                "model_request_count": 0,
                "behavioral_episode_count": 0,
                "reviewer_identity": "unit-test",
                "reviewed_at_utc": "2026-09-05T00:00:00Z",
                "source_seed_receipt": {
                    "path": str(first_receipt),
                    "sha256": sha256_file(first_receipt),
                    "bytes": first_receipt.stat().st_size,
                    "environment_seed": seeds[0],
                },
                "source_axis_overlay": {
                    "path": str(overlay),
                    "sha256": sha256_file(overlay),
                    "bytes": overlay.stat().st_size,
                },
                "assertions": {
                    "left_axis_matches_fixed_robot_viewpoint": True,
                    "front_axis_points_toward_robot": True,
                    "up_axis_opposes_gravity": True,
                    "labels_and_arrow_origins_visible": True,
                },
            }
            axis_path = root / "axis-review.json"
            axis_path.write_bytes(canonical_json_bytes(axis_review))
            output = root / "aggregate.json"
            report = compiler.compile_receipts(
                registry_path=REGISTRY,
                registry_sha256=sha256_file(REGISTRY),
                receipts_root=root,
                axis_review_path=axis_path,
                output_path=output,
            )
            self.assertTrue(report["passed"])
            self.assertEqual(report["observed_seed_count"], 128)
            self.assertEqual(report["model_request_count"], 0)
            self.assertEqual(report["behavioral_episode_count"], 0)
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
