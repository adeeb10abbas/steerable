"""Tests for the prospective model-blind V4 horizontal reset registry."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from experiments.online_correction_v4.droid_task_files.binding import sha256_file
from experiments.online_correction_v4.droid_task_files.reset_registry import (
    load_reset_registry,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_v4_horizontal_reset_registry",
        ROOT / "tools/build_v4_horizontal_reset_registry.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


class HorizontalResetRegistryBuilderTests(unittest.TestCase):
    def build(self):
        return builder.build_registry(
            campaign_path=builder.DEFAULT_CAMPAIGN,
            queue_path=builder.DEFAULT_QUEUE,
            source_report_path=builder.DEFAULT_SOURCE,
        )

    def test_covers_all_registered_horizontal_blocks(self) -> None:
        payload = self.build()
        self.assertEqual(payload["registered_env_seed_count"], 128)
        self.assertEqual(payload["registered_env_seed_min"], 2100000000)
        self.assertEqual(payload["registered_env_seed_max"], 2100000127)
        self.assertEqual(len(payload["resets_by_env_seed"]), 128)
        self.assertEqual(payload["model_request_count"], 0)
        self.assertEqual(payload["behavioral_episode_count"], 0)

    def test_common_jitter_preserves_relative_geometry(self) -> None:
        payload = self.build()
        source = payload["source_identity"]["base_positions_robot_base_m"]
        for reset in payload["resets_by_env_seed"].values():
            positions = reset["positions_robot_base_m"]
            for name in ("bowl", "banana"):
                expected = [
                    source[name][axis] - source["rubiks_cube"][axis]
                    for axis in range(3)
                ]
                observed = [
                    positions[name][axis] - positions["rubiks_cube"][axis]
                    for axis in range(3)
                ]
                for actual, wanted in zip(observed, expected):
                    self.assertAlmostEqual(actual, wanted, places=12)

    def test_axis_jitter_is_deterministic_bounded_and_independent(self) -> None:
        payload_a = self.build()
        payload_b = self.build()
        self.assertEqual(
            builder.canonical_json_bytes(payload_a),
            builder.canonical_json_bytes(payload_b),
        )
        jitters = [
            tuple(row["jitter_robot_base_xy_m"])
            for row in payload_a["resets_by_env_seed"].values()
        ]
        self.assertEqual(len(set(jitters)), 128)
        self.assertTrue(
            all(abs(x) <= builder.JITTER_HALF_RANGE_X_M for x, _ in jitters)
        )
        self.assertTrue(
            all(abs(y) <= builder.JITTER_HALF_RANGE_Y_M for _, y in jitters)
        )
        self.assertTrue(any(abs(x - y) > 1e-9 for x, y in jitters))

    def test_output_is_accepted_by_runtime_registry_loader(self) -> None:
        payload = self.build()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            path.write_bytes(builder.canonical_json_bytes(payload))
            registry = load_reset_registry(
                registry_path=str(path),
                registry_sha256=sha256_file(path),
            )
        self.assertEqual(len(registry.positions_by_env_seed), 128)
        self.assertIn(2100000000, registry.positions_by_env_seed)

    def test_missing_queue_seed_fails_closed(self) -> None:
        rows = [
            json.loads(line)
            for line in builder.DEFAULT_QUEUE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows = [row for row in rows if row.get("env_seed") != 2100000127]
        with tempfile.TemporaryDirectory() as tmp:
            queue = Path(tmp) / "queue.jsonl"
            queue.write_text(
                "".join(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )
            with self.assertRaises(builder.ResetRegistryBuildError):
                builder.build_registry(
                    campaign_path=builder.DEFAULT_CAMPAIGN,
                    queue_path=queue,
                    source_report_path=builder.DEFAULT_SOURCE,
                )


if __name__ == "__main__":
    unittest.main()
