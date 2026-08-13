"""Focused static tests for A004 mixed-epoch raw routing."""

import unittest
from pathlib import Path

from tools.aggregate_v3c002r001_activation_v4_raw import (
    A003_RELEASE_SHA256,
    RETRY,
    SLOTS,
    _expected_epoch,
    _lane_bindings,
    _same_bound_bytes,
)


class A004RawAggregationTests(unittest.TestCase):
    def test_retry_map_and_slots_are_frozen(self):
        self.assertEqual(len(SLOTS), 8)
        self.assertEqual(RETRY["repair-lane-00"], 12060)
        self.assertEqual(RETRY["repair-lane-07"], 12112)

    def test_epoch_routing_is_structural(self):
        continuation = {"remaining_seed_blocks_by_lane": {slot: [13000 + i] for i, slot in enumerate(SLOTS)}}
        lane = {slot: {"lane_slot": slot} for slot in SLOTS}
        binding = {slot: {"path": __file__, "bytes": Path(__file__).stat().st_size, "sha256": "x"} for slot in SLOTS}
        # Patch only file-binding inputs with this existing file; returned epoch is the assertion target.
        args = dict(continuation=continuation, original_path=Path(__file__), original_lanes=lane, original_bindings=binding,
                    a003_path=Path(__file__), a003_lanes=lane, a003_bindings=binding,
                    continuation_path=Path(__file__), continuation_lanes=lane, continuation_bindings=binding)
        self.assertEqual(_expected_epoch(slot="repair-lane-00", seed=13000, **args)[0], "continuation")
        self.assertEqual(_expected_epoch(slot="repair-lane-00", seed=12060, **args)[0], "a003_replacement_retry")
        self.assertEqual(_expected_epoch(slot="repair-lane-02", seed=12128, **args)[0], "original_release")

    def test_binding_comparison_is_checkout_portable_but_not_digest_lax(self):
        left = {"path": "/checkout-a/evidence.json", "bytes": 19, "sha256": "a" * 64}
        right = {"path": "/checkout-b/evidence.json", "bytes": 19, "sha256": "a" * 64}
        self.assertTrue(_same_bound_bytes(left, right))
        self.assertFalse(_same_bound_bytes(left, {**right, "sha256": "b" * 64}))
        self.assertFalse(_same_bound_bytes(left, {**right, "bytes": 20}))

    def test_real_a003_absolute_checkout_bindings_relocate_by_full_repo_path(self):
        root = Path(__file__).resolve().parents[1]
        gate = root / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002r001/activation_v3/lane_replacement_a003/release_gate.released.json"
        _, values, bindings = _lane_bindings(
            gate,
            expected_sha256=A003_RELEASE_SHA256,
            expected_schema="vla-wam-shared-v3c002r001-activation-v3-lane-replacement-gate-v1",
            expected_status="passed_activation_v3_cluster_termination_lane_replacement",
        )
        self.assertEqual(set(values), set(SLOTS))
        self.assertEqual(bindings["repair-lane-00"]["sha256"], "7cb1121b3f8bcd6527c1a484376b0d51330f771de788a701324c47938ab6891f")


if __name__ == "__main__": unittest.main()
