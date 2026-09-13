from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


SOURCE_ROOT = Path(__file__).resolve().parents[3]
FORECAST = SOURCE_ROOT / "workshops/corl2026_world_models/experiments/forecast_layout"
sys.path.insert(0, str(FORECAST))

import confirmation_fixture_freeze as fixtures
import confirmation_runtime_common as common
import d1_confirmation_block_jobs as d1
import n3_confirmation_block_job as n3


STUDY_COMMIT = "1" * 40
SHA = "a" * 64


def write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(common.canonical_bytes(value) + b"\n")
    return common.sha256_file(path)


class ScheduleTests(unittest.TestCase):
    @staticmethod
    def _schedule() -> dict:
        path = SOURCE_ROOT / "workshops/corl2026_world_models/execution/20260912/parallel_schedule.json"
        return json.loads(path.read_text())

    def test_all_24_permutations_and_model_orders_are_exact(self) -> None:
        orders = []
        for index, layout in enumerate(common.CONFIRMATION_LAYOUT_IDS, start=1):
            n3_block = common.load_confirmation_schedule_block(SOURCE_ROOT, layout, "N3")
            d1_block = common.load_confirmation_schedule_block(SOURCE_ROOT, layout, "D1")
            self.assertEqual(n3_block.condition_order, d1_block.condition_order)
            self.assertEqual(n3_block.environment_seed, 2026091200 + index)
            self.assertEqual(n3_block.effective_model_seed, 2026091200 + index)
            self.assertEqual(d1_block.environment_seed, 2026091200 + index)
            self.assertEqual(d1_block.effective_model_seed, 1140)
            self.assertTrue(all(cell.startswith(f"wmf1__confirmation__{layout}__") for cell in n3_block.cell_ids))
            orders.append(n3_block.condition_order)
        self.assertEqual(len(set(orders)), 24)

    def test_schedule_rejects_nonconfirmation_layout(self) -> None:
        with self.assertRaisesRegex(common.ConfirmationRuntimeError, "confirmation_layout_unsupported"):
            common.load_confirmation_schedule_block(SOURCE_ROOT, "D01", "N3")

    def test_schedule_rejects_model_matched_permutation_reassignment(self) -> None:
        schedule = self._schedule()
        rows = {
            (row["layout_pair_id"], row["model_config"]): row
            for row in schedule["jobs"]
            if row["phase"] == "confirmation"
        }
        for model in ("N3", "D1"):
            first = rows[("C01", model)]
            second = rows[("C02", model)]
            first["condition_order"], second["condition_order"] = (
                second["condition_order"], first["condition_order"],
            )
        with self.assertRaisesRegex(
            common.ConfirmationRuntimeError,
            "confirmation_condition_order_assignment_changed",
        ):
            common._validate_global_confirmation_order(schedule)

    def test_schedule_rejects_permutation_index_reassignment(self) -> None:
        schedule = self._schedule()
        schedule["order_assignment"]["permutation_index_by_block"]["C01"] = 0
        with self.assertRaisesRegex(
            common.ConfirmationRuntimeError,
            "confirmation_permutation_index_changed",
        ):
            common._validate_global_confirmation_order(schedule)

    def test_d1_contract_is_official_conditional_and_fixed_noise(self) -> None:
        block = d1.load_confirmation_block(SOURCE_ROOT, "C24")
        self.assertEqual(block.contract["phase"], "confirmation")
        self.assertEqual(block.contract["action_path"], "official conditional DreamZero-DROID path")
        self.assertEqual(block.contract["effective_model_noise_seed"], 1140)
        self.assertIn("not independent noise draws", block.contract["noise_semantics"])
        self.assertFalse(block.contract["transport"]["stateful_request_replay"])


class ReleaseFreezeAdmissionTests(unittest.TestCase):
    def test_runtime_calls_release_api_and_extracts_exact_model_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = {
                "schema_version": "wmf-development-confirmation-release-freeze-v1",
                "status": "frozen_for_confirmation",
                "cohort_branch": "full_two_model",
                "qualified_model_ids": ["N3", "D1"],
                "alignment_contracts_by_model": {
                    "N3": {"path": "n3.json", "contract_sha256": "b" * 64},
                    "D1": {"path": "d1.json", "contract_sha256": "c" * 64},
                },
                "release_decision": {"eligible": True, "blockers": []},
            }
            path = root / "confirmation_release.json"
            digest = write_json(path, value)
            validator = mock.Mock(return_value=value)
            module = SimpleNamespace(validate_release_freeze=validator)
            with mock.patch.object(common, "_load_release_module", return_value=module):
                result = common.verify_confirmation_freeze(
                    path, digest, source_root=SOURCE_ROOT, model_config="D1"
                )
            validator.assert_called_once_with(path.resolve(), digest, expected_model="D1")
            self.assertEqual(result["alignment_contract"], value["alignment_contracts_by_model"]["D1"])
            self.assertEqual(result["confirmation_freeze"]["sha256"], digest)

    def test_release_file_symlink_is_rejected_before_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.json"
            digest = write_json(target, {})
            link = root / "link.json"
            link.symlink_to(target)
            with self.assertRaisesRegex(common.ConfirmationRuntimeError, "confirmation_evidence_symlink"):
                common.verify_exact_file(link, digest, "freeze")


class FixtureFreezeTests(unittest.TestCase):
    def _frozen_rows(self, root: Path) -> dict[str, dict]:
        artifact = root / "artifact.bin"
        artifact.write_bytes(b"immutable fixture evidence")
        identity = common.file_identity(artifact)
        rows = {}
        for index, layout in enumerate(common.CONFIRMATION_LAYOUT_IDS, start=1):
            rank = 1 if layout in {"C01", "C17", "C24"} else 0
            rows[layout] = {
                "layout_pair_id": layout,
                "environment_seed": 2026091200 + index,
                "candidate_id": f"{layout}__candidate_{rank:02d}",
                "candidate_payload_sha256": hashlib.sha256(f"candidate:{layout}:{rank}".encode()).hexdigest(),
                "accepted_gate_record_sha256": hashlib.sha256(f"gate:{layout}:{rank}".encode()).hexdigest(),
                "gate_receipt": identity,
                "pose_manifest": identity,
                "gate_ledger": identity,
                "gate_attempt_receipt": identity,
                "capture_receipt": identity,
                "raw_capture_receipt": identity,
                "n3_fixed_observation": identity,
                "d1_fixed_observation": identity,
                "model_request_count": 0,
                "behavioral_action_count": 0,
            }
        return rows

    def _inventory(self, rows: dict[str, dict]) -> dict:
        return {
            "schema_version": fixtures.INVENTORY_SCHEMA,
            "study_id": common.STUDY_ID,
            "namespace": common.NAMESPACE,
            "selection_uses_target_model_outcomes": False,
            "candidate_selection_rule": "lowest deterministic candidate rank passing model-blind physical checks",
            "layouts": [
                {
                    "layout_pair_id": layout,
                    "candidate_id": row["candidate_id"],
                    "gate_receipt_path": row["gate_receipt"]["path"],
                    "gate_receipt_sha256": row["gate_receipt"]["sha256"],
                    "pose_manifest_path": row["pose_manifest"]["path"],
                    "pose_manifest_sha256": row["pose_manifest"]["sha256"],
                    "capture_receipt_path": row["capture_receipt"]["path"],
                    "capture_receipt_sha256": row["capture_receipt"]["sha256"],
                }
                for layout, row in rows.items()
            ],
        }

    def test_builder_and_validator_require_complete_24_layout_model_blind_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = self._frozen_rows(root)
            inventory_path = root / "inventory.json"
            write_json(inventory_path, self._inventory(rows))
            with mock.patch.object(
                fixtures,
                "validate_one_fixture",
                side_effect=lambda **kwargs: copy.deepcopy(rows[kwargs["layout_pair_id"]]),
            ):
                freeze = fixtures.build_fixture_freeze(
                    source_root=SOURCE_ROOT,
                    inventory_path=inventory_path,
                    study_commit=STUDY_COMMIT,
                )
            self.assertEqual([row["layout_pair_id"] for row in freeze["layouts"]], list(common.CONFIRMATION_LAYOUT_IDS))
            self.assertEqual(freeze["model_request_count"], 0)
            self.assertFalse(freeze["selection_uses_target_model_outcomes"])
            freeze_path = root / "fixture_freeze.json"
            digest = write_json(freeze_path, freeze)
            with mock.patch.object(
                fixtures,
                "validate_one_fixture",
                return_value=copy.deepcopy(rows["C17"]),
            ):
                result = fixtures.validate_fixture_freeze(
                    freeze_path,
                    digest,
                    source_root=SOURCE_ROOT,
                    expected_layout_pair_id="C17",
                    expected_study_commit=STUDY_COMMIT,
                )
            self.assertEqual(result["layout_count"], 24)
            self.assertEqual(result["selected_layout"]["candidate_id"], "C17__candidate_01")
            with self.assertRaisesRegex(
                common.ConfirmationRuntimeError,
                "confirmation_fixture_study_commit_changed",
            ):
                fixtures.validate_fixture_freeze(
                    freeze_path,
                    digest,
                    source_root=SOURCE_ROOT,
                    expected_study_commit="2" * 40,
                    deep_validate_selected=False,
                )

    def test_builder_rejects_missing_layout_and_outcome_selected_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = self._frozen_rows(root)
            inventory = self._inventory(rows)
            inventory["layouts"].pop()
            path = root / "missing.json"
            write_json(path, inventory)
            with self.assertRaisesRegex(common.ConfirmationRuntimeError, "inventory_count_changed"):
                fixtures.build_fixture_freeze(source_root=SOURCE_ROOT, inventory_path=path, study_commit=STUDY_COMMIT)
            inventory = self._inventory(rows)
            inventory["selection_uses_target_model_outcomes"] = True
            path = root / "outcome.json"
            write_json(path, inventory)
            with self.assertRaisesRegex(common.ConfirmationRuntimeError, "selection_not_model_blind"):
                fixtures.build_fixture_freeze(source_root=SOURCE_ROOT, inventory_path=path, study_commit=STUDY_COMMIT)
            inventory = self._inventory(rows)
            inventory["layouts"][0]["candidate_id"] = "C01__candidate_00"
            path = root / "wrong-accepted-candidate.json"
            write_json(path, inventory)
            with self.assertRaisesRegex(
                common.ConfirmationRuntimeError,
                "accepted_candidate_changed",
            ):
                fixtures.build_fixture_freeze(
                    source_root=SOURCE_ROOT,
                    inventory_path=path,
                    study_commit=STUDY_COMMIT,
                )

    def test_signed_freeze_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = self._frozen_rows(root)
            inventory_path = root / "inventory.json"
            write_json(inventory_path, self._inventory(rows))
            with mock.patch.object(fixtures, "validate_one_fixture", side_effect=lambda **kwargs: copy.deepcopy(rows[kwargs["layout_pair_id"]])):
                freeze = fixtures.build_fixture_freeze(source_root=SOURCE_ROOT, inventory_path=inventory_path, study_commit=STUDY_COMMIT)
            freeze["layouts"][0]["candidate_id"] = "C01__candidate_00"
            path = root / "tampered.json"
            digest = write_json(path, freeze)
            with self.assertRaisesRegex(common.ConfirmationRuntimeError, "payload_hash_mismatch"):
                fixtures.validate_fixture_freeze(path, digest, source_root=SOURCE_ROOT, deep_validate_selected=False)


class LauncherContractTests(unittest.TestCase):
    def test_n3_every_mode_requires_both_freezes_and_exact_fixture(self) -> None:
        parser = n3.build_parser()
        sub = next(action for action in parser._actions if action.dest == "mode")
        required = {
            "candidate_id", "gate_receipt", "gate_receipt_sha256",
            "pose_manifest", "pose_manifest_sha256", "capture_receipt",
            "capture_receipt_sha256", "confirmation_freeze",
            "confirmation_freeze_sha256", "fixture_freeze",
            "fixture_freeze_sha256",
        }
        for mode in ("queue", "server", "cell"):
            self.assertTrue(required.issubset({action.dest for action in sub.choices[mode]._actions}))

    def test_n3_child_commands_reenter_confirmation_wrapper_with_freezes(self) -> None:
        block = n3.load_confirmation_block(SOURCE_ROOT, "C01")
        command = n3.build_server_command(
            source_root=SOURCE_ROOT,
            attempt_root=Path("/tmp/n3-confirmation-test"),
            port=18011,
            study_commit=STUDY_COMMIT,
            start_cell_index=2,
            block=block,
            candidate_id="C01__candidate_01",
            gate_receipt=Path("/tmp/gate.json"),
            gate_receipt_sha256=SHA,
            pose_manifest=Path("/tmp/pose.json"),
            pose_manifest_sha256=SHA,
            capture_receipt=Path("/tmp/capture.json"),
            capture_receipt_sha256=SHA,
            confirmation_freeze=Path("/tmp/release.json"),
            confirmation_freeze_sha256=SHA,
            fixture_freeze_path=Path("/tmp/fixtures.json"),
            fixture_freeze_sha256=SHA,
        )
        self.assertTrue(command[1].endswith(n3.RUNNER_FILENAME))
        self.assertEqual(common.descriptor_option(command, "--start-cell-index"), "2")
        self.assertEqual(common.descriptor_option(command, "--fixture-freeze-sha256"), SHA)

    def test_no_replay_contracts_are_explicit(self) -> None:
        self.assertEqual(n3.NO_REPLAY_TRANSPORT_CONTRACT["behavioral_request_retry_count"], 0)
        self.assertEqual(n3.NO_REPLAY_TRANSPORT_CONTRACT["lost_response_policy"], "fail_cell_without_replaying_request")
        self.assertFalse(d1.NO_REPLAY_TRANSPORT_CONTRACT["stateful_request_replay"])
        self.assertIn("without_replaying_reset_or_inference", d1.NO_REPLAY_TRANSPORT_CONTRACT["lost_response_policy"])

    def test_d1_metadata_binds_confirmation_contract_fixture_and_no_replay(self) -> None:
        block = d1.load_confirmation_block(SOURCE_ROOT, "C01")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cell.json"
            fixture = {"candidate_id": "C01__candidate_01", "execution_prerequisites_sha256": SHA}
            with d1.configured_pilot(block, d1.ALLOWED_SIMULATOR_ROLES[0]):
                with d1.installed_receipt_metadata(block, fixture=fixture):
                    d1.pilot.immutable_json(path, {"schema_version": d1.CELL_RECEIPT_SCHEMA})
            value = json.loads(path.read_text())
        self.assertEqual(value["phase"], "confirmation")
        self.assertEqual(value["layout_pair_id"], "C01")
        self.assertEqual(value["confirmation_contract_sha256"], block.contract_sha256)
        self.assertEqual(value["confirmation_fixture"], fixture)
        self.assertFalse(value["transport_contract"]["stateful_request_replay"])
        self.assertIn("450 actual actions", value["claim_boundary"])

    def test_n3_server_does_not_launch_when_release_validation_fails(self) -> None:
        block = n3.load_confirmation_block(SOURCE_ROOT, "C02")
        args = SimpleNamespace()
        with mock.patch.object(
            n3,
            "validate_prerequisites",
            side_effect=common.ConfirmationRuntimeError("confirmation_release_freeze_invalid"),
        ), mock.patch.object(n3.pilot, "run_server") as launch:
            with self.assertRaisesRegex(common.ConfirmationRuntimeError, "confirmation_release_freeze_invalid"):
                n3.run_server(args, block)
        launch.assert_not_called()

    def test_execution_prerequisite_hash_rejects_mutation(self) -> None:
        block = d1.load_confirmation_block(SOURCE_ROOT, "C02")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "execution.json"
            prerequisites = {"confirmation_release": {"freeze": SHA}}
            identity, digest = d1.write_execution_prerequisites(
                path, block=block, prerequisites=prerequisites
            )
            self.assertEqual(identity["sha256"], digest)
            d1.verify_execution_prerequisites(
                path, digest, block=block, expected_prerequisites=prerequisites
            )
            changed = json.loads(path.read_text())
            changed["prerequisites"]["confirmation_release"]["freeze"] = "b" * 64
            path.unlink()
            write_json(path, changed)
            with self.assertRaisesRegex(d1.pilot.D1BehavioralPilotError, "evidence_sha256_mismatch"):
                d1.verify_execution_prerequisites(path, digest, block=block)


if __name__ == "__main__":
    unittest.main()
