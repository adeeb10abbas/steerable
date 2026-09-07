from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.build_v4_object_pair_confirmatory_release import (
    ROOT,
    build_confirmatory_lane_spec,
    build_confirmatory_seed_registry,
    build_launch_matrix,
    build_runtime_lock,
)


class ObjectPairConfirmatoryReleaseTests(unittest.TestCase):
    def test_runtime_lock_rebinds_runner_hash(self) -> None:
        lock = build_runtime_lock(
            pilot_lock_path=ROOT
            / "artifacts/online_correction_v4/setup/object_pair_g7_runtime_lock.released.json",
            queue_path=ROOT / "artifacts/online_correction_v4/queue.jsonl",
            queue_manifest_path=ROOT
            / "artifacts/online_correction_v4/queue_manifest.json",
            seed_registry_path=ROOT
            / "artifacts/online_correction_v4/setup/"
            "object_pair_c7_confirmatory_nano_seed_registry.released.json",
            main_reset_registry_path=ROOT
            / "artifacts/online_correction_v4/setup/object_pair_reset_registry.released.json",
            main_g2_path=ROOT
            / "artifacts/online_correction_v4/qualification/"
            "20260906_object_pair_g2_aggregate_g2c7q20260905ap.json",
            main_g3_path=ROOT
            / "artifacts/online_correction_v4/qualification/"
            "20260906_object_pair_g3_aggregate_g3s7q20260906l.json",
            hardware_g4_path=ROOT
            / "artifacts/online_correction_v4/qualification/"
            "object_pair_g4_nano_a10080_a40_hardware_g4c7a100q20260906b.json",
            g7_path=ROOT
            / "artifacts/online_correction_v4/qualification/"
            "20260906_object_pair_g7_engineering_pilot_g7c7q20260906a.json",
            g8_path=ROOT
            / "artifacts/online_correction_v4/qualification/"
            "20260906_object_pair_g8_miniature_g8c7q20260906a.json",
            analysis_manifest_path=ROOT
            / "artifacts/online_correction_v4/frozen_analysis_manifest.json",
            source_commit="a" * 40,
            runtime_root="/runtime/c7-main",
        )

        import hashlib

        self.assertEqual(
            lock["runner"]["sha256"],
            hashlib.sha256(
                (ROOT / "tools/run_online_correction_v4.py").read_bytes()
            ).hexdigest(),
        )

    def test_main_c7_seeds_are_complete_and_disjoint_from_pilot(self) -> None:
        registry = build_confirmatory_seed_registry(
            queue_path=ROOT / "artifacts/online_correction_v4/queue.jsonl",
            pilot_seed_registry_path=(
                ROOT
                / "artifacts/online_correction_v4/setup/"
                "object_pair_g7_nano_seed_registry.released.json"
            ),
        )
        seeds = registry["allowed_sampling_seeds"]
        self.assertEqual(registry["scope"], "released_c7")
        self.assertEqual(len(seeds), 64)
        self.assertEqual(len(set(seeds)), 64)
        self.assertEqual(
            registry["pilot_collision_audit"]["collision_count"],
            0,
        )

    def test_lane_spec_and_matrix_bind_a100_stratum(self) -> None:
        pilot_seed = (
            ROOT
            / "artifacts/online_correction_v4/setup/"
            "object_pair_g7_nano_seed_registry.released.json"
        )
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            temp = Path(tmp)
            seed_path = temp / "object_pair_c7_confirmatory_nano_seed_registry.released.json"
            seed_path.write_text(
                json.dumps(
                    build_confirmatory_seed_registry(
                        queue_path=ROOT
                        / "artifacts/online_correction_v4/queue.jsonl",
                        pilot_seed_registry_path=pilot_seed,
                    )
                ),
                encoding="utf-8",
            )
            spec = build_confirmatory_lane_spec(
                pilot_lane_spec_path=ROOT
                / "deploy/k8s/v4_lane_bundle/"
                "g7-object-pair-pilot-spec.example.json",
                pilot_seed_registry_path=pilot_seed,
                seed_registry_path=seed_path,
                runtime_root="/runtime/c7-main",
                output_parent="/raw/c7-main",
            )
            self.assertEqual(
                spec["policy"]["gpu_product"],
                "NVIDIA-A100-SXM4-80GB",
            )
            port_index = spec["policy"]["experiment_argv"].index("--port")
            self.assertEqual(
                spec["policy"]["experiment_argv"][port_index + 1],
                "18157",
            )
            self.assertTrue(
                any(
                    seed_path.name in item
                    for item in spec["policy"]["experiment_argv"]
                )
            )
            self.assertIn(
                "/runtime/c7-main",
                spec["runtime"]["policy"]["pythonpath"],
            )
            self.assertFalse(
                Path(spec["policy"]["file_bindings"][0]["source"]).is_absolute()
            )
            spec_path = temp / "spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            lock_path = temp / "lock.json"
            lock_path.write_text(
                json.dumps(
                    {
                        "release_status": "RELEASED",
                        "released_families": ["C7"],
                    }
                ),
                encoding="utf-8",
            )
            matrix = build_launch_matrix(
                lane_spec_path=spec_path,
                runtime_lock_path=lock_path,
                hardware_g4_path=ROOT
                / "artifacts/online_correction_v4/qualification/"
                "object_pair_g4_nano_a10080_a40_hardware_"
                "g4c7a100q20260906b.json",
                lane_count=40,
            )
            self.assertEqual(len(matrix["qualified_lanes"]), 40)
            self.assertEqual(
                matrix["qualified_lanes"][-1]["lane_id"],
                "c7m39",
            )


if __name__ == "__main__":
    unittest.main()
