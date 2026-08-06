#!/usr/bin/env python3
"""Focused fail-closed tests for the π0.5 V3-B002 launch contract."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.v3.pi05_phase_b.contract import (
    SEEDS, cells_for_lane, expected_cells, load_release_bundle, sha256_file,
)
from experiments.v3.pi05_phase_b.runtime import validate_live_topology


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/"artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/pi05_mirror_v3b002_manifest.json"


def _topology() -> dict:
    row = {"owner": "ali", "pod": "sim", "pod_uid": "uid-sim", "pod_ip": "10.0.0.2",
           "gpu_uuid": "GPU-sim", "gpu_model": "RTX PRO 6000 Blackwell", "driver_version": "580.105.08"}
    return {
        "schema_version": "vla-wam-shared-v3b-pi05-live-topology-v1",
        "policy_server": {"owner": "ali", "pod": "policy", "pod_uid": "uid-policy",
                          "pod_ip": "10.0.0.1", "gpu_uuid": "GPU-policy",
                          "gpu_model": "B200", "driver_version": "580.105.08",
                          "gpu_index": 2, "port": 8001, "model_request_count_at_capture": 0},
        "simulator_lanes": [row],
    }


class ContractTest(unittest.TestCase):
    def test_committed_registration_is_the_only_108_cell_release(self) -> None:
        release = load_release_bundle(ROOT, MANIFEST, expected_manifest_sha256=sha256_file(MANIFEST))
        self.assertEqual(len(release.cells), 108)
        self.assertEqual(len(expected_cells(ROOT)), 108)
        for seed in SEEDS:
            block = [cell for cell in release.cells if cell.seed == seed]
            self.assertEqual([cell.row["execution_order_index_within_seed"] for cell in block], [1, 2, 3, 4])
            self.assertEqual({(cell.arm, cell.relation) for cell in block}, {
                ("control", "left"), ("control", "right"),
                ("position_mirrored", "left"), ("position_mirrored", "right"),
            })

    def test_whole_seed_lanes_are_disjoint_and_complete(self) -> None:
        release = load_release_bundle(ROOT, MANIFEST, expected_manifest_sha256=sha256_file(MANIFEST))
        lanes = [cells_for_lane(release.cells, lane_index=index, lane_count=7) for index in range(7)]
        self.assertEqual(sum(len(lane) for lane in lanes), 108)
        self.assertEqual(len({cell.cell_id for lane in lanes for cell in lane}), 108)
        owners = {}
        for index, lane in enumerate(lanes):
            for cell in lane:
                owners.setdefault(cell.seed, set()).add(index)
        self.assertTrue(all(len(value) == 1 for value in owners.values()))

    def test_topology_requires_explicit_ali_ownership_and_zero_request_server(self) -> None:
        self.assertEqual(validate_live_topology(_topology())["policy_server"]["gpu_index"], 2)
        bad = json.loads(json.dumps(_topology()))
        bad["simulator_lanes"][0]["owner"] = "someone_else"
        with self.assertRaisesRegex(ValueError, "ali-owned"):
            validate_live_topology(bad)
        bad = _topology()
        bad["policy_server"]["model_request_count_at_capture"] = 1
        with self.assertRaisesRegex(ValueError, "request zero"):
            validate_live_topology(bad)


if __name__ == "__main__":
    unittest.main()
