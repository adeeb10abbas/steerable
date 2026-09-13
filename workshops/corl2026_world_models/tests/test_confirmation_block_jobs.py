from __future__ import annotations

import copy
import dataclasses
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
            path = Path(temporary) / "cell_receipt.json"
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


class TerminalFailureContractTests(unittest.TestCase):
    @staticmethod
    def _n3_failure_path(root: Path, block, index: int = 0) -> Path:
        return (
            root
            / "attempt-1"
            / "cells"
            / f"{index:02d}-{n3.pilot.safe_cell_component(block.cell_ids[index])}"
            / "technical_failure.json"
        )

    @staticmethod
    def _n3_failure(block, *, status: str, actions: int, requests: int) -> dict:
        return {
            "schema_version": n3.CELL_RECEIPT_SCHEMA,
            "status": status,
            "recorded_stop_reason": "safety_abort",
            "study_id": n3.pilot.STUDY_ID,
            "phase": "confirmation",
            "block_id": block.block_id,
            "layout_pair_id": block.layout_pair_id,
            "model_config": "N3",
            "cell_id": block.cell_ids[0],
            "condition_index": 0,
            "actions_executed": actions,
            "request_count": requests,
            "episode_context_id": "n3-terminal-context",
            "server_context_id": "n3-terminal-context",
            "client_session_id": "n3-client-session",
            "server_context_terminal": None,
        }

    def test_n3_terminal_less_safety_is_technical_and_retryable(self) -> None:
        block = n3.load_confirmation_block(SOURCE_ROOT, "C01")
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            path = self._n3_failure_path(raw, block)
            safety = self._n3_failure(
                block, status="safety_abort", actions=1, requests=1
            )
            write_json(path, safety)
            with self.assertRaisesRegex(
                n3.pilot.N3BehavioralPilotError,
                "confirmation_safety_abort_terminal_missing",
            ):
                n3.validate_failed_confirmation_cell(
                    path, condition_index=0, block=block
                )

            technical = self._n3_failure(
                block, status="technical_failure", actions=0, requests=0
            )
            write_json(path, technical)
            observed = n3.validate_failed_confirmation_cell(
                path, condition_index=0, block=block
            )
            self.assertEqual(observed["recorded_stop_reason"], "safety_abort")
            counts = n3._receipt_counts(
                block=block,
                start_cell_index=0,
                launched=1,
                completed=[],
                attempt_root=path.parents[2],
            )
            self.assertEqual(counts["technically_invalid_behavioral_cells"], 1)
            self.assertEqual(counts["right_censored_behavioral_cells"], 0)
            prerequisites = {
                "confirmation_release": {"confirmation_freeze": {}},
                "confirmation_fixture_freeze": {},
            }
            receipts, _identities, provenance = n3.discover_completed_prefix(
                raw,
                block=block,
                prerequisites=prerequisites,
                study_commit=STUDY_COMMIT,
            )
            self.assertEqual(receipts, [])
            self.assertEqual(provenance["start_cell_index"], 0)

    def test_n3_authenticated_safety_terminal_censors_and_stops_resume(self) -> None:
        block = n3.load_confirmation_block(SOURCE_ROOT, "C01")
        with tempfile.TemporaryDirectory() as temporary, n3.configured_pilot(block):
            raw = Path(temporary)
            failure_path = self._n3_failure_path(raw, block)
            attempt = failure_path.parents[2]
            protocol = n3.pilot.ServerProtocol()
            layout_arm, command, _task = block.conditions[0]
            begin = protocol.begin(
                {
                    "study_id": n3.pilot.STUDY_ID,
                    "block_id": block.block_id,
                    "cell_id": block.cell_ids[0],
                    "condition_index": 0,
                    "layout_arm": layout_arm,
                    "command": command,
                    "prompt": n3.pilot.PROMPTS[command],
                    "effective_seed": block.effective_seed,
                    "expected_actions": n3.pilot.ACTION_CAP,
                    "expected_requests": n3.pilot.REQUEST_COUNT,
                    "client_session_id": "n3-client-session",
                },
                server_context_id="n3-terminal-context",
                temporal_reset_evidence={
                    "passed": True,
                    "unresolved_mutable_temporal_fields": [],
                },
            )
            protocol.complete_behavioral()
            end = protocol.end(
                {
                    "study_id": n3.pilot.STUDY_ID,
                    "block_id": block.block_id,
                    "cell_id": block.cell_ids[0],
                    "condition_index": 0,
                    "server_context_id": begin["server_context_id"],
                    "client_session_id": begin["client_session_id"],
                    "status": "safety_abort",
                    "stop_reason": "safety_abort",
                    "actions_executed": 17,
                    "request_count": 1,
                    "final_chunk_executed_actions": None,
                }
            )
            terminal = n3.pilot.persist_server_terminal_receipt(
                attempt_root=attempt,
                end_response=end,
                protocol_context_active=False,
                model_capture_active=False,
            )
            failure = self._n3_failure(
                block, status="safety_abort", actions=17, requests=1
            )
            failure["server_context_terminal"] = terminal
            write_json(failure_path, failure)
            observed = n3.validate_failed_confirmation_cell(
                failure_path, condition_index=0, block=block
            )
            self.assertEqual(observed["status"], "safety_abort")

            tampered = dict(failure)
            tampered["server_context_id"] = "other-context"
            write_json(failure_path, tampered)
            with self.assertRaisesRegex(
                n3.pilot.N3BehavioralPilotError,
                "confirmation_failure_context_identity_changed",
            ):
                n3.validate_failed_confirmation_cell(
                    failure_path, condition_index=0, block=block
                )
            write_json(failure_path, failure)

            terminal_path = Path(terminal["path"])
            original_terminal = json.loads(terminal_path.read_text())
            for label, client_session_id in (
                ("null", None),
                ("alias", begin["server_context_id"]),
                ("unsafe", "bad/session"),
            ):
                with self.subTest(coordinated_failure_session=label):
                    write_json(
                        terminal_path,
                        {
                            **original_terminal,
                            "client_session_id": client_session_id,
                        },
                    )
                    changed_failure = {
                        **failure,
                        "client_session_id": client_session_id,
                        "server_context_terminal": n3.pilot.file_identity(
                            terminal_path
                        ),
                    }
                    write_json(failure_path, changed_failure)
                    with self.assertRaisesRegex(
                        n3.pilot.N3BehavioralPilotError,
                        "server_context_terminal_identity_invalid",
                    ):
                        n3.validate_failed_confirmation_cell(
                            failure_path, condition_index=0, block=block
                        )
                    write_json(terminal_path, original_terminal)
                    failure["server_context_terminal"] = n3.pilot.file_identity(
                        terminal_path
                    )
                    write_json(failure_path, failure)

            prerequisites = {
                "confirmation_release": {"confirmation_freeze": {}},
                "confirmation_fixture_freeze": {},
            }
            with mock.patch.object(
                n3, "validate_cells_bind_prerequisites"
            ), self.assertRaisesRegex(
                n3.pilot.N3BehavioralPilotError,
                "confirmation_safety_censored_block_is_terminal",
            ):
                n3.discover_completed_prefix(
                    raw,
                    block=block,
                    prerequisites=prerequisites,
                    study_commit=STUDY_COMMIT,
                )

    def test_n3_passed_confirmation_rejects_coordinated_unsafe_sessions(self) -> None:
        block = n3.load_confirmation_block(SOURCE_ROOT, "C01")
        with tempfile.TemporaryDirectory() as temporary, n3.configured_pilot(block):
            raw = Path(temporary)
            cell_path = (
                raw
                / "attempt-1"
                / "cells"
                / f"00-{n3.pilot.safe_cell_component(block.cell_ids[0])}"
                / "cell_receipt.json"
            )
            write_json(cell_path, {"placeholder": True})
            attempt = cell_path.parents[2]
            protocol = n3.pilot.ServerProtocol()
            layout_arm, command, _task = block.conditions[0]
            begin = protocol.begin(
                {
                    "study_id": n3.pilot.STUDY_ID,
                    "block_id": block.block_id,
                    "cell_id": block.cell_ids[0],
                    "condition_index": 0,
                    "layout_arm": layout_arm,
                    "command": command,
                    "prompt": n3.pilot.PROMPTS[command],
                    "effective_seed": block.effective_seed,
                    "expected_actions": n3.pilot.ACTION_CAP,
                    "expected_requests": n3.pilot.REQUEST_COUNT,
                    "client_session_id": "n3-passed-client-session",
                },
                server_context_id="n3-passed-terminal-context",
                temporal_reset_evidence={
                    "passed": True,
                    "unresolved_mutable_temporal_fields": [],
                },
            )
            for _ in range(n3.pilot.REQUEST_COUNT):
                protocol.complete_behavioral()
            end = protocol.end(
                {
                    "study_id": n3.pilot.STUDY_ID,
                    "block_id": block.block_id,
                    "cell_id": block.cell_ids[0],
                    "condition_index": 0,
                    "server_context_id": begin["server_context_id"],
                    "client_session_id": begin["client_session_id"],
                    "status": "completed",
                    "stop_reason": "action_cap",
                    "actions_executed": n3.pilot.ACTION_CAP,
                    "request_count": n3.pilot.REQUEST_COUNT,
                    "final_chunk_executed_actions": n3.pilot.FINAL_EXECUTED_ACTIONS,
                }
            )
            terminal = n3.pilot.persist_server_terminal_receipt(
                attempt_root=attempt,
                end_response=end,
                protocol_context_active=False,
                model_capture_active=False,
            )
            receipt = {
                "phase": "confirmation",
                "source_pins": {"study_commit": STUDY_COMMIT},
                "transport_contract": copy.deepcopy(
                    n3.NO_REPLAY_TRANSPORT_CONTRACT
                ),
                "confirmation_prerequisites": {},
                "server_begin_receipt": dict(begin),
                "server_end_receipt": {
                    **end,
                    "server_context_terminal": terminal,
                },
                "server_context_terminal": terminal,
            }
            identity = n3.pilot.file_identity(cell_path)

            def validate(candidate: dict) -> None:
                with mock.patch.object(
                    n3.development,
                    "_validate_cell_receipt",
                    return_value=(candidate, identity),
                ):
                    n3.validate_passed_confirmation_cell(
                        cell_path,
                        condition_index=0,
                        study_commit=STUDY_COMMIT,
                        block=block,
                    )

            validate(receipt)
            changed_end = copy.deepcopy(receipt)
            changed_end["server_end_receipt"]["client_session_id"] = "other-session"
            with self.assertRaisesRegex(
                n3.pilot.N3BehavioralPilotError,
                "confirmation_end_context_identity_changed",
            ):
                validate(changed_end)

            terminal_path = Path(terminal["path"])
            original_terminal = json.loads(terminal_path.read_text())
            for label, client_session_id in (
                ("null", None),
                ("alias", begin["server_context_id"]),
                ("unsafe", "bad/session"),
            ):
                with self.subTest(coordinated_passed_session=label):
                    write_json(
                        terminal_path,
                        {
                            **original_terminal,
                            "client_session_id": client_session_id,
                        },
                    )
                    changed = copy.deepcopy(receipt)
                    changed["server_begin_receipt"][
                        "client_session_id"
                    ] = client_session_id
                    changed["server_end_receipt"][
                        "client_session_id"
                    ] = client_session_id
                    changed_terminal = n3.pilot.file_identity(terminal_path)
                    changed["server_end_receipt"][
                        "server_context_terminal"
                    ] = changed_terminal
                    changed["server_context_terminal"] = changed_terminal
                    with self.assertRaisesRegex(
                        n3.pilot.N3BehavioralPilotError,
                        "server_context_terminal_identity_invalid",
                    ):
                        validate(changed)
                    write_json(terminal_path, original_terminal)

    @staticmethod
    def _d1_failure_path(root: Path, block, index: int = 0) -> Path:
        return (
            root
            / "simulator_attempts"
            / "sim-1"
            / "cells"
            / f"{index:02d}-{d1.pilot.safe_component(block.cell_ids[index])}"
            / "technical_failure.json"
        )

    @staticmethod
    def _d1_failure(block, *, status: str, actions: int, requests: int) -> dict:
        return {
            "schema_version": d1.CELL_RECEIPT_SCHEMA,
            "status": status,
            "recorded_stop_reason": "safety_abort",
            "study_id": d1.pilot.STUDY_ID,
            "phase": "confirmation",
            "block_id": block.block_id,
            "layout_pair_id": block.layout_pair_id,
            "model_config": "D1",
            "cell_id": block.cell_ids[0],
            "condition_index": 0,
            "run_id": "run-1",
            "simulator_job_id": "sim-1",
            "server_job_id": "server-1",
            "study_commit": STUDY_COMMIT,
            "actions_executed": actions,
            "request_count": requests,
            "episode_id": "d1c01-terminal-context",
            "episode_context_id": "d1c01-terminal-context",
            "server_context_id": "d1c01-terminal-context",
            "client_session_id": "d1-client-session",
            "server_context_terminal": None,
            "server_episode_manifest": None,
            "server_terminal_reset_scan": None,
        }

    def test_d1_terminal_less_safety_is_technical_and_retryable(self) -> None:
        template = d1.load_confirmation_block(SOURCE_ROOT, "C01")
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            block = dataclasses.replace(template, raw_root=raw)
            path = self._d1_failure_path(raw, block)
            with d1.configured_pilot(block, d1.ALLOWED_SIMULATOR_ROLES[0]):
                safety = self._d1_failure(
                    block, status="safety_abort", actions=1, requests=1
                )
                write_json(path, safety)
                with self.assertRaisesRegex(
                    d1.pilot.D1BehavioralPilotError,
                    "confirmation_safety_abort_terminal_missing",
                ):
                    d1.validate_failed_confirmation_cell(
                        path, condition_index=0, block=block
                    )

                technical = self._d1_failure(
                    block, status="technical_failure", actions=0, requests=0
                )
                write_json(path, technical)
                observed = d1.validate_failed_confirmation_cell(
                    path, condition_index=0, block=block
                )
                self.assertEqual(observed["recorded_stop_reason"], "safety_abort")
                with d1.development._patched_pilot(
                    {
                        "validate_failure_cell_receipt": (
                            lambda failure_path, *, condition_index:
                            d1.validate_failed_confirmation_cell(
                                failure_path,
                                condition_index=condition_index,
                                block=block,
                            )
                        )
                    }
                ):
                    counts = d1.pilot._receipt_counts(
                        start_cell_index=0,
                        launched=1,
                        completed=[],
                        attempt_root=path.parents[2],
                    )
                self.assertEqual(counts["technically_invalid_behavioral_cells"], 1)
                self.assertEqual(counts["right_censored_behavioral_cells"], 0)

                prerequisites = {
                    "confirmation_release": {"confirmation_freeze": {}},
                    "confirmation_fixture_freeze": {},
                }
                with mock.patch.object(d1, "validate_cells_bind_prerequisites"):
                    receipts, _identities, provenance = d1.discover_completed_prefix(
                        raw,
                        block=block,
                        prerequisites=prerequisites,
                        study_commit=STUDY_COMMIT,
                        simulator_worker_role=d1.ALLOWED_SIMULATOR_ROLES[0],
                    )
                self.assertEqual(receipts, [])
                self.assertEqual(provenance["start_cell_index"], 0)

    def _write_d1_ready_bundle(self, raw: Path, block) -> tuple[dict, dict, dict]:
        future = raw / "server_attempts" / "server-1" / "future"
        future.mkdir(parents=True)
        identity_receipt = {
            "status": "passed",
            "source": {
                "commit": d1.pilot.D1_SOURCE_COMMIT,
                "git_tree": d1.pilot.D1_SOURCE_TREE,
                "aggregate_sha256": d1.pilot.D1_SOURCE_AGGREGATE_SHA256,
            },
            "checkpoint": {
                "revision": d1.pilot.CHECKPOINT_REVISION,
                "aggregate_sha256": d1.pilot.CHECKPOINT_AGGREGATE_SHA256,
            },
            "tokenizer": {
                "revision": d1.pilot.TOKENIZER_REVISION,
                "aggregate_sha256": d1.pilot.TOKENIZER_AGGREGATE_SHA256,
            },
        }
        identity_path = future / "identity_receipt.json"
        write_json(identity_path, identity_receipt)
        overlay = FORECAST / "d1_instrumented_server.py"
        contract = {
            "schema_version": d1.pilot.D1_SERVER_CONTRACT_SCHEMA,
            "status": "passed",
            "configuration_id": "D1",
            "official_repository_commit": d1.pilot.D1_SOURCE_COMMIT,
            "official_repository_tree": d1.pilot.D1_SOURCE_TREE,
            "official_action_path": "GrootSimPolicy.lazy_joint_forward_causal",
            "custom_s2_used": False,
            "patched_s1_used": False,
            "world_size": 2,
            "port": d1.pilot.SERVICE_PORT,
            "returned_action_shape": [
                d1.pilot.RETURNED_ACTION_HORIZON,
                d1.pilot.ACTION_DIM,
            ],
            "executed_action_prefix": d1.pilot.EXECUTED_PREFIX_HORIZON,
            "effective_official_model_noise_seed": d1.pilot.EFFECTIVE_MODEL_NOISE_SEED,
            "video_guidance_scale": 5.0,
            "configured_inference_steps": 16,
            "evaluated_dit_step_count": 8,
            "dynamic_cache_schedule": False,
            "tensorrt_engine_active": False,
            "enable_dit_cache": True,
            "future_root": str(future.resolve()),
            "noise_semantics": "fixed; no request is an independent noise draw",
            "instrumentation_overlay": {
                "path": str(overlay.resolve()),
                "sha256": d1.pilot.sha256_file(overlay),
                "returned_action_modified": False,
            },
            "topology": [
                {
                    "rank": rank,
                    "cuda_device_index": rank,
                    "cuda_device_name": "NVIDIA B200",
                }
                for rank in (0, 1)
            ],
            "head_contracts": [
                {"rank": rank, "status": "passed"} for rank in (0, 1)
            ],
            "bounded_loader_receipts": [
                {
                    "rank": rank,
                    "receipt": {"passed": True, "forward_path_modified": False},
                }
                for rank in (0, 1)
            ],
            "identity_receipt": str(identity_path.resolve()),
            "identity_receipt_sha256": d1.pilot.sha256_file(identity_path),
        }
        contract_path = future / "server_contract.json"
        write_json(contract_path, contract)
        execution_path = (
            raw / "simulator_attempts" / "sim-1" / "execution_prerequisites.json"
        )
        execution, execution_sha = d1.write_execution_prerequisites(
            execution_path, block=block, prerequisites={}
        )
        ready = {
            "schema_version": d1.pilot.SERVER_READY_SCHEMA,
            "status": "ready",
            "run_id": "run-1",
            "server_job_id": "server-1",
            "paired_simulator_job_id": "sim-1",
            "study_commit": STUDY_COMMIT,
            "study_id": d1.pilot.STUDY_ID,
            "phase": "confirmation",
            "block_id": block.block_id,
            "layout_pair_id": block.layout_pair_id,
            "model_config": "D1",
            "service_host": d1.pilot.SERVICE_HOST,
            "service_port": d1.pilot.SERVICE_PORT,
            "future_root": str(future.resolve()),
            "server_contract": d1.pilot.file_identity(contract_path),
            "server_contract_sha256": d1.pilot.sha256_file(contract_path),
            "runtime_identity": d1.pilot.file_identity(identity_path),
            "pilot_contract": dict(block.contract),
            "pilot_contract_sha256": block.contract_sha256,
            "confirmation_contract_sha256": block.contract_sha256,
            "confirmation_prerequisites_sha256": execution_sha,
            "expected_cell_ids": list(block.cell_ids),
            "returned_action_shape": [
                d1.pilot.RETURNED_ACTION_HORIZON,
                d1.pilot.ACTION_DIM,
            ],
            "executed_prefix_horizon": d1.pilot.EXECUTED_PREFIX_HORIZON,
            "effective_model_noise_seed": d1.pilot.EFFECTIVE_MODEL_NOISE_SEED,
            "global_state_noninterleaving": True,
        }
        paths = d1.pilot.coordination_paths(raw, "run-1")
        write_json(paths["server_ready"], ready)
        ready_identity = d1.pilot.file_identity(paths["server_ready"])
        claim = {
            "schema_version": d1.pilot.SIMULATOR_CLAIM_SCHEMA,
            "status": "claimed",
            "run_id": "run-1",
            "simulator_job_id": "sim-1",
            "server_job_id": "server-1",
            "server_ready_sha256": ready_identity["sha256"],
            "study_commit": STUDY_COMMIT,
            "block_id": block.block_id,
            "worker_role": d1.ALLOWED_SIMULATOR_ROLES[0],
            "pilot_contract_sha256": block.contract_sha256,
            "lease_token": "lease-1",
            "start_cell_index": 0,
        }
        write_json(paths["simulator_claim"], claim)
        return ready, execution, claim

    def test_d1_failure_deep_ready_claim_context_and_terminal_stop(self) -> None:
        template = d1.load_confirmation_block(SOURCE_ROOT, "C01")
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            block = dataclasses.replace(template, raw_root=raw)
            role = d1.ALLOWED_SIMULATOR_ROLES[0]
            with d1.configured_pilot(block, role):
                ready, execution, claim = self._write_d1_ready_bundle(raw, block)
                paths = d1.pilot.coordination_paths(raw, "run-1")
                failure_path = self._d1_failure_path(raw, block)
                attempt = failure_path.parents[2]
                write_json(
                    attempt / "resume.json",
                    {
                        "schema_version": d1.RESUME_SCHEMA,
                        "block_id": block.block_id,
                        "layout_pair_id": block.layout_pair_id,
                        "start_cell_index": 0,
                    },
                )
                future = Path(ready["future_root"])
                episode = "d1c01-terminal-context"
                terminal_path = (
                    future / "episodes" / episode / "terminal_context_receipt.json"
                )
                manifest_path = future / "episodes" / episode / "episode_manifest.json"
                write_json(terminal_path, {"server-authored": True})
                write_json(manifest_path, {"server-authored": True})
                terminal_identity = d1.pilot.file_identity(terminal_path)
                manifest_identity = d1.pilot.file_identity(manifest_path)
                reset_scan = {"passed": True, "world_size": 2}
                failure = self._d1_failure(
                    block, status="safety_abort", actions=449, requests=57
                )
                failure.update(
                    {
                        "server_ready": d1.pilot.file_identity(paths["server_ready"]),
                        "simulator_claim": d1.pilot.file_identity(
                            paths["simulator_claim"]
                        ),
                        "confirmation_fixture": {
                            "execution_prerequisites": execution,
                            "execution_prerequisites_sha256": execution["sha256"],
                        },
                        "server_begin_receipt": {
                            "passed": True,
                            "episode_context_id": episode,
                            "client_session_id": "d1-client-session",
                        },
                        "server_context_terminal": terminal_identity,
                        "server_episode_manifest": manifest_identity,
                        "server_terminal_reset_scan": reset_scan,
                    }
                )
                write_json(failure_path, failure)

                terminal_result = (
                    {"terminal_state": "context_closed"},
                    terminal_identity,
                    {},
                    manifest_identity,
                    [],
                    reset_scan,
                )
                with mock.patch.object(
                    d1.pilot,
                    "validate_server_terminal_receipt",
                    return_value=terminal_result,
                ) as terminal_validator:
                    observed = d1.validate_failed_confirmation_cell(
                        failure_path, condition_index=0, block=block
                    )
                self.assertEqual(observed["status"], "safety_abort")
                terminal_validator.assert_called_once()
                finalize_control = terminal_validator.call_args.kwargs[
                    "expected_finalize_control"
                ]
                self.assertEqual(finalize_control["actions_executed"], 449)
                self.assertEqual(finalize_control["request_count"], 57)

                original_ready = dict(ready)
                coordinated_ready = {**ready, "server_job_id": "evil-server"}
                write_json(paths["server_ready"], coordinated_ready)
                coordinated_ready_identity = d1.pilot.file_identity(
                    paths["server_ready"]
                )
                coordinated_claim = {
                    **claim,
                    "server_job_id": "evil-server",
                    "server_ready_sha256": coordinated_ready_identity["sha256"],
                }
                write_json(paths["simulator_claim"], coordinated_claim)
                coordinated_failure = {
                    **failure,
                    "server_ready": coordinated_ready_identity,
                    "simulator_claim": d1.pilot.file_identity(
                        paths["simulator_claim"]
                    ),
                }
                write_json(failure_path, coordinated_failure)
                with self.assertRaisesRegex(
                    d1.pilot.D1BehavioralPilotError,
                    "confirmation_failure_server_ready_changed",
                ):
                    d1.validate_failed_confirmation_cell(
                        failure_path, condition_index=0, block=block
                    )

                write_json(paths["server_ready"], original_ready)
                ready_identity = d1.pilot.file_identity(paths["server_ready"])
                bad_claim = {
                    **claim,
                    "server_ready_sha256": ready_identity["sha256"],
                    "worker_role": "wrong-role",
                }
                write_json(paths["simulator_claim"], bad_claim)
                claim_failure = {
                    **failure,
                    "server_ready": ready_identity,
                    "simulator_claim": d1.pilot.file_identity(
                        paths["simulator_claim"]
                    ),
                }
                write_json(failure_path, claim_failure)
                with self.assertRaisesRegex(
                    d1.pilot.D1BehavioralPilotError, "simulator_claim_mismatch"
                ):
                    d1.validate_failed_confirmation_cell(
                        failure_path, condition_index=0, block=block
                    )

                write_json(paths["simulator_claim"], claim)
                restored = {
                    **failure,
                    "server_ready": ready_identity,
                    "simulator_claim": d1.pilot.file_identity(
                        paths["simulator_claim"]
                    ),
                }
                changed_context = {**restored, "server_context_id": "other-context"}
                write_json(failure_path, changed_context)
                with self.assertRaisesRegex(
                    d1.pilot.D1BehavioralPilotError,
                    "confirmation_failure_context_ids_changed",
                ):
                    d1.validate_failed_confirmation_cell(
                        failure_path, condition_index=0, block=block
                    )

                write_json(failure_path, restored)
                prerequisites = {
                    "confirmation_release": {"confirmation_freeze": {}},
                    "confirmation_fixture_freeze": {},
                }
                with (
                    mock.patch.object(
                        d1.pilot,
                        "validate_server_terminal_receipt",
                        return_value=terminal_result,
                    ),
                    mock.patch.object(d1, "verify_execution_prerequisites"),
                    mock.patch.object(d1, "validate_cells_bind_prerequisites"),
                    self.assertRaisesRegex(
                        d1.pilot.D1BehavioralPilotError,
                        "confirmation_safety_censored_block_is_terminal",
                    ),
                ):
                    d1.discover_completed_prefix(
                        raw,
                        block=block,
                        prerequisites=prerequisites,
                        study_commit=STUDY_COMMIT,
                        simulator_worker_role=role,
                    )


if __name__ == "__main__":
    unittest.main()
