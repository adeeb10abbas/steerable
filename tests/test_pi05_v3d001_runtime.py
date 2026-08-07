#!/usr/bin/env python3
"""Focused tests for the released π0.5 V3-D001 behavioral runtime."""

from __future__ import annotations

from pathlib import Path
import unittest

from experiments.v3.pi05_stochastic_v3d001.contract import (
    QUEUE_SHA256, RELEASE_MANIFEST_SHA256, SEEDS, cells_for_lane, load_release,
    sha256_file,
)
from experiments.v3.pi05_stochastic_v3d001.queue import cell_plan


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/"artifacts/vla_wam_shared_v3/prospective_tier_b/releases/v3d001/release_manifest.json"


class V3D001RuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.release = load_release(ROOT, MANIFEST)

    def test_release_is_exact_432_cells_and_hash_bound(self) -> None:
        self.assertEqual(sha256_file(MANIFEST), RELEASE_MANIFEST_SHA256)
        self.assertEqual(sha256_file(self.release.queue_path), QUEUE_SHA256)
        self.assertEqual(len(self.release.cells), 432)
        self.assertEqual({cell.environment_seed for cell in self.release.cells}, set(SEEDS))
        self.assertEqual(len({cell.block_id for cell in self.release.cells}), 216)

    def test_queue_order_preserves_adjacent_matched_blocks(self) -> None:
        for offset in range(0, len(self.release.cells), 2):
            block = self.release.cells[offset:offset+2]
            self.assertEqual(block[0].block_id, block[1].block_id)
            self.assertEqual({cell.relation for cell in block}, {"left", "right"})
            self.assertEqual(
                [cell.row["execution_order_index_within_matched_stochastic_block"] for cell in block],
                [0, 1],
            )

    def test_sampling_seed_bases_are_shared_only_inside_matched_block(self) -> None:
        for offset in range(0, len(self.release.cells), 2):
            left, right = self.release.cells[offset:offset+2]
            self.assertEqual(left.sampling_seed_base, right.sampling_seed_base)
            self.assertEqual(
                left.sampling_seed_base,
                left.environment_seed * 1_000_000 + left.sampling_index * 1_000,
            )

    def test_whole_seed_lanes_are_disjoint_and_complete(self) -> None:
        lanes = [cells_for_lane(self.release.cells, index, 7) for index in range(7)]
        self.assertEqual(sum(map(len, lanes)), 432)
        self.assertEqual(len({cell.cell_id for lane in lanes for cell in lane}), 432)
        owners: dict[int, set[int]] = {}
        for lane_index, lane in enumerate(lanes):
            for cell in lane:
                owners.setdefault(cell.environment_seed, set()).add(lane_index)
        self.assertTrue(all(len(indices) == 1 for indices in owners.values()))

    def test_plan_uses_released_seed_and_thermal_process_group_guard(self) -> None:
        cell = self.release.cells[0]
        plan = cell_plan(
            repo_root=ROOT, release_manifest=MANIFEST,
            runtime_identity=Path("/pvc/runtime.json"),
            phase_a_release_gate=Path("/pvc/gate.json"), raw_root=Path("/pvc/raw"),
            cell=cell, remote_host="10.0.0.10", remote_port=8001,
            device="cuda:0", gpu_index=0, lane_pod_uid="ali-pod-uid",
            lane_gpu_uuid="GPU-ALI", attempt_index=2,
        )
        self.assertIn("native_process_group_thermal_guard.py", " ".join(plan["bridge_command"]))
        self.assertIn(str(cell.sampling_seed_base), plan["bridge_command"])
        self.assertEqual(plan["environment"]["OMNI_KIT_ACCEPT_EULA"], "YES")
        self.assertEqual(plan["matched_stochastic_block_id"], cell.block_id)
        self.assertTrue(plan["attempt_dir"].endswith("attempt02"))


if __name__ == "__main__":
    unittest.main()
