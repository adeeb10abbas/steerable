"""Focused contract tests for the additive A004 mixed-epoch finalizer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest

from experiments.v3.phase_c_semantic_equivalence_v3c002 import compiler as parent
from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import ContractError, file_binding
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v4_finalizer.finalizer import (
    A003_RELEASE_SHA,
    IDENTITY_KEYS,
    ORIGINAL_RELEASE_SHA,
    RETRY,
    SLOTS,
    _released_lanes,
    build_exact_routing,
    add_r001_diagnostics,
    identity_normalized_copy,
    validate_infrastructure,
    validate_finalization_admission,
)
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v4_finalizer.registration import V10_CONTINUATION_SHA
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v4_finalizer.source_gate import (
    _verify_commit,
    verify_pushed_lineage,
)


def _lane(slot: str, binding: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    return binding, {"lane_slot": slot, "lane_id": f"lane-{slot}", "server_port": 8110}


class A004FinalizerTests(unittest.TestCase):
    def test_a003_absolute_artifact_bindings_relocate_only_to_checkout_artifacts(self):
        base = Path(__file__).resolve().parents[1] / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002r001/activation_v3"
        _, original = _released_lanes(base / "release_gate.released.json", expected_sha=ORIGINAL_RELEASE_SHA, label="original")
        _, replacement = _released_lanes(base / "lane_replacement_a003/release_gate.released.json", expected_sha=A003_RELEASE_SHA, label="a003")
        self.assertEqual(set(original), set(SLOTS))
        self.assertEqual(set(replacement), set(SLOTS))
        self.assertIn("/artifacts/", replacement["repair-lane-00"][0]["path"])

    def test_source_gate_rejects_non_commit_metadata(self):
        root = Path(__file__).resolve().parents[1]
        with self.assertRaises(ContractError):
            _verify_commit(root, "0" * 40, "not-a-real-commit")

    def test_finalizer_refuses_to_open_raw_or_receipt_without_final_analysis_admission(self):
        root = Path(__file__).resolve().parents[1]
        base = root / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002r001"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ContractError):
                validate_finalization_admission(
                    finalization_registration=Path(tmp) / "absent-registration.json",
                    finalization_source_gate=Path(tmp) / "absent-source-gate.json",
                    parent_registration=root / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002/active/registration.json",
                    repair_registration=file_binding(base / "activation_v3/registration.json"),
                    queue=base / "activation_v3/queue.jsonl",
                    original_release=base / "activation_v3/release_gate.released.json",
                    a003_release=base / "activation_v3/lane_replacement_a003/release_gate.released.json",
                    continuation_gate=base / "activation_v4/v10/continuation_gate.released.json",
                    v11_registration=base / "activation_v4/v11/registration.json",
                    v11_source_gate=base / "activation_v4/v11/source_push_gate.released.json",
                )

    def test_v10_continuation_identity_is_fixed(self):
        root = Path(__file__).resolve().parents[1]
        gate = root / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002r001/activation_v4/v10/continuation_gate.released.json"
        self.assertEqual(__import__("hashlib").sha256(gate.read_bytes()).hexdigest(), V10_CONTINUATION_SHA)

    def test_source_gate_rejects_wrong_remote_before_network_or_git_claims(self):
        with self.assertRaises(ContractError):
            verify_pushed_lineage(
                registration=Path(__file__), implementation_commit="0" * 40,
                registration_commit="1" * 40, remote_head="1" * 40,
                remote="https://example.invalid/not-the-study.git",
                branch="experiment/v3c002-semantic-equivalence", inventory={},
            )

    def test_exact_seed_epoch_routing_rejects_count_drift(self):
        seeds = list(range(12000, 12341))
        assignment = {seed: "repair-lane-03" for seed in seeds}
        assignment[RETRY["repair-lane-00"]] = "repair-lane-00"
        assignment[RETRY["repair-lane-01"]] = "repair-lane-01"
        continuation_seeds = set(seed for seed in seeds if seed not in set(RETRY.values()))
        continuation_seeds = set(sorted(continuation_seeds)[:209])
        remaining = {slot: set() for slot in SLOTS}
        remaining["repair-lane-03"] = continuation_seeds
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source_paths = []
            for name in ("original", "a003", "continuation"):
                path = directory / name
                path.write_text(name, encoding="utf-8")
                source_paths.append(path)
            bindings = [file_binding(path) for path in source_paths]
            original_lanes = {slot: _lane(slot, bindings[0]) for slot in SLOTS}
            a003_lanes = {slot: _lane(slot, bindings[1]) for slot in SLOTS}
            continuation_lanes = {slot: _lane(slot, bindings[2]) for slot in SLOTS}
            routes = build_exact_routing(
                assignment=assignment,
                original_gate={"_path": str(source_paths[0])}, original_lanes=original_lanes,
                a003_gate={"_path": str(source_paths[1])}, a003_lanes=a003_lanes,
                continuation_gate={"_path": str(source_paths[2])}, continuation_lanes=continuation_lanes,
                continuation_remaining=remaining,
            )
            self.assertEqual(routes[12060]["epoch"], "a003_replacement_retry")
            self.assertEqual(routes[12101]["epoch"], "a003_replacement_retry")
            self.assertEqual(routes[next(iter(continuation_seeds))]["epoch"], "continuation")
            original_seed = next(seed for seed in seeds if seed not in continuation_seeds and seed not in {12060, 12101})
            self.assertEqual(routes[original_seed]["epoch"], "original_release")
            remaining["repair-lane-03"].remove(next(iter(continuation_seeds)))
            with self.assertRaises(ContractError):
                build_exact_routing(
                    assignment=assignment,
                    original_gate={"_path": str(source_paths[0])}, original_lanes=original_lanes,
                    a003_gate={"_path": str(source_paths[1])}, a003_lanes=a003_lanes,
                    continuation_gate={"_path": str(source_paths[2])}, continuation_lanes=continuation_lanes,
                    continuation_remaining=remaining,
                )

    def test_analysis_normalization_is_deep_and_cannot_change_pair_rows(self):
        actual = []
        for seed in range(12000, 12341):
            for condition, goal in (
                ("canonical_left", "left"), ("inverse_reference_left", "left"),
                ("canonical_right", "right"), ("inverse_reference_right", "right"),
            ):
                identity = {key: f"first-{key}" if seed == 12000 else f"later-{key}" for key in IDENTITY_KEYS}
                actual.append({
                    "cell_id": f"cell-{seed}-{condition}",
                    "episode_seed": seed, "prompt_condition": condition, "physical_goal": goal,
                    "initial_state_sha256": f"state-{seed}",
                    "requested_side_depth": 0.1, "success": True,
                    "action_trace_sha256": f"trace-{seed}-{condition}", "lane_id": "lane-zero",
                    "runtime_identity": {"nested": identity["server_port"]}, **identity,
                })
        normalized = identity_normalized_copy(actual)
        self.assertEqual(parent._pair_rows(actual), parent._pair_rows(normalized))
        self.assertEqual(actual[-1]["server_port"], "later-server_port")
        self.assertEqual(normalized[-1]["server_port"], "first-server_port")
        normalized[-1]["runtime_identity"]["nested"] = "changed-only-in-copy"
        self.assertNotEqual(actual[-1]["runtime_identity"]["nested"], "changed-only-in-copy")

    def test_r001_diagnostics_consume_exact_parent_pair_schema(self):
        episodes = []
        assignment = {}
        for seed in range(12000, 12341):
            slot = SLOTS[(seed - 12000) % len(SLOTS)]
            assignment[seed] = slot
            for index, (condition, goal) in enumerate((
                ("canonical_left", "left"), ("inverse_reference_left", "left"),
                ("canonical_right", "right"), ("inverse_reference_right", "right"),
            )):
                episodes.append({
                    "cell_id": f"cell-{seed}-{condition}", "episode_seed": seed,
                    "prompt_condition": condition, "physical_goal": goal,
                    "initial_state_sha256": f"state-{seed}",
                    "requested_side_depth": float(index), "success": True,
                    "action_trace_sha256": f"trace-{seed}-{condition}",
                })
        pairs = parent._pair_rows(episodes)
        self.assertIn("depth_difference_inverse_minus_canonical_m", pairs[0])
        self.assertNotIn("depth_inverse_minus_canonical_m", pairs[0])
        results = {}
        add_r001_diagnostics(results, pairs, assignment)
        self.assertEqual(set(results["lane_diagnostics_descriptive_only"]), set(SLOTS))
        self.assertEqual(set(results["leave_one_lane_out_diagnostics_descriptive_only"]), set(SLOTS))

    def test_all_fourteen_infrastructure_rows_are_mandatory(self):
        @dataclass(frozen=True)
        class Cell:
            seed: int
            block_id: str

        cell = Cell(seed=12000, block_id="v3c002:seed12000")
        row = {
            "schema_version": "vla-wam-shared-v3c002-infrastructure-attempt-v1",
            "record_type": "infrastructure_attempt",
            "infrastructure_status": "infrastructure_invalid_excluded",
            "denominator_eligible": False,
            "authorization_mode": "behavioral",
            "cell_id": "cell", "seed_block_id": cell.block_id, "attempt_root": "/retained/repair-lane-00/attempt001",
        }
        with self.assertRaises(ContractError):
            validate_infrastructure([row] * 13, cells_by_id={"cell": cell}, assignment={12000: "repair-lane-00"})
        self.assertEqual(
            len(validate_infrastructure([row] * 14, cells_by_id={"cell": cell}, assignment={12000: "repair-lane-00"})),
            14,
        )


if __name__ == "__main__":
    unittest.main()
