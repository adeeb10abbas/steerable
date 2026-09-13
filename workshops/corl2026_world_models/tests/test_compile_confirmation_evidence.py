from __future__ import annotations

import ast
from contextlib import nullcontext
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


WORKSHOP = Path(__file__).resolve().parents[1]
REPOSITORY = WORKSHOP.parents[1]
MODULE_PATH = WORKSHOP / "analysis/compile_confirmation_evidence.py"
SPEC = importlib.util.spec_from_file_location("compile_confirmation_evidence_tests", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
compiler = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compiler
SPEC.loader.exec_module(compiler)


def write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def descriptor(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": compiler.sha256_file(path),
        "bytes": path.stat().st_size,
    }


def stored(value: object) -> object:
    if isinstance(value, dict):
        return {
            "__type__": "mapping",
            "items": {key: stored(item) for key, item in value.items()},
        }
    if isinstance(value, list):
        return {"__type__": "list", "items": [stored(item) for item in value]}
    return value


def recorder_payload(role: str, value: object) -> dict:
    result = {
        "role": role,
        "structure": stored(value),
        "array_count": 0,
        "artifact": None,
    }
    result["payload_sha256"] = compiler.sha256_bytes(
        compiler.development.canonical_bytes(result, ensure_ascii=True)
    )
    return result


def alignment(model: str = "N3") -> dict:
    value = {
        "contract_id": f"alignment-{model.lower()}",
        "model_id": model,
        "mapping_receipt_id": f"mapping-{model.lower()}",
        "mapping_receipt_sha256": "1" * 64,
        "primary_horizon_s": 0.4,
        "generated_frame_index": 2,
        "target_executed_action_offset": 4,
        "control_step_s": 0.1,
        "captured_frame_interval_s": 0.1,
        "timestamp_tolerance_s": 0.05,
        "camera_id": "over_shoulder_left_camera",
        "camera_crop_id": f"crop-{model.lower()}",
        "camera_crop_sha256": "2" * 64,
        "image_width_px": 320,
        "image_height_px": 180,
        "early_horizon": {
            "horizon_s": 0.2,
            "generated_frame_index": 1,
            "target_executed_action_offset": 2,
        },
    }
    value["contract_sha256"] = compiler.sha256_bytes(compiler.canonical_bytes(value))
    return value


def counts(*, launched: int = 0, completed: int = 0, invalid: int = 0, censored: int = 0,
           actions: int = 0, requests: int = 0) -> dict:
    return {
        "planned_behavioral_cells": 4,
        "launched_behavioral_cells": launched,
        "resumed_valid_behavioral_cells": completed,
        "newly_launched_behavioral_cells": launched - completed,
        "completed_valid_behavioral_cells": completed,
        "technically_invalid_behavioral_cells": invalid,
        "right_censored_behavioral_cells": censored,
        "unrun_behavioral_cells": 4 - launched,
        "actual_behavioral_actions": actions,
        "actual_behavioral_model_requests": requests,
        "new_generation_qualification_requests": 0,
        "reused_prerequisite_generation_qualification_requests": 6,
        "recorder_only_episodes_counted_as_behavioral": 0,
    }


class ConfirmationEvidenceCompilerTests(unittest.TestCase):
    def test_source_contains_no_duplicate_literal_dictionary_keys(self) -> None:
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        duplicates = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = [
                key.value for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            ]
            duplicates.extend((node.lineno, key) for key in set(keys) if keys.count(key) > 1)
        self.assertEqual(duplicates, [])

    def test_contract_matches_native_terminal_runtime_acceptance_rule(self) -> None:
        contract = json.loads(compiler.CONTRACT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            contract["compiler"]["terminal_runtime_source_commit"],
            compiler.TERMINAL_RUNTIME_SOURCE_COMMIT,
        )
        censored = contract["source_authentication"]["censored"]
        self.assertIn(compiler.N3_CONTEXT_TERMINAL_SCHEMA, censored)
        self.assertIn(compiler.D1_CONTEXT_TERMINAL_SCHEMA, censored)
        self.assertIn(compiler.TERMINAL_RUNTIME_SOURCE_COMMIT, censored)
        self.assertIn("forbids a retry", censored)
        technical = contract["source_authentication"]["technical_invalid"]
        self.assertIn("0 through 450", technical)
        self.assertIn("excluded from annotation requests", technical)

    def test_confirmation_cell_paths_match_both_runtime_normalizers(self) -> None:
        for model, loader, normalizer in (
            (
                "N3", compiler.n3_confirmation.load_confirmation_block,
                compiler.n3_confirmation.pilot.safe_cell_component,
            ),
            (
                "D1", compiler.d1_confirmation.load_confirmation_block,
                compiler.d1_confirmation.pilot.safe_component,
            ),
        ):
            schedule = loader(REPOSITORY, "C01")
            for index, cell_id in enumerate(schedule.cell_ids):
                observed = compiler._cell_root(Path("/raw/attempt"), index, cell_id)
                self.assertEqual(
                    observed.name,
                    f"{index:02d}-{normalizer(cell_id)}",
                    model,
                )

    def test_count_gate_preserves_all_four_status_categories(self) -> None:
        invalid_block = compiler._validate_count_object(
            counts(launched=3, completed=2, invalid=1, actions=111, requests=9),
            model="N3",
            layout="C01",
        )
        censored_block = compiler._validate_count_object(
            counts(launched=1, censored=1, actions=7, requests=1),
            model="N3",
            layout="C02",
        )
        self.assertEqual(invalid_block["completed_valid_behavioral_cells"], 2)
        self.assertEqual(invalid_block["technically_invalid_behavioral_cells"], 1)
        self.assertEqual(invalid_block["unrun_behavioral_cells"], 1)
        self.assertEqual(censored_block["right_censored_behavioral_cells"], 1)
        self.assertEqual(censored_block["unrun_behavioral_cells"], 3)
        broken = counts(launched=4, completed=2, invalid=1, censored=0)
        with self.assertRaisesRegex(compiler.ConfirmationCompilerError, "do not close"):
            compiler._validate_count_object(broken, model="N3", layout="C01")

    def test_signed_input_tamper_fails_closed(self) -> None:
        value = compiler.sign_document({"schema_version": compiler.INPUT_SCHEMA, "study_id": compiler.STUDY_ID})
        compiler.verify_signed(value, "input")
        value["study_id"] = "another-study"
        with self.assertRaisesRegex(compiler.ConfirmationCompilerError, "payload hash mismatch"):
            compiler.verify_signed(value, "input")

    def test_direct_compiler_rejects_a_different_study_head(self) -> None:
        with self.assertRaisesRegex(
            compiler.ConfirmationCompilerError, "HEAD differs from study commit"
        ):
            compiler._verify_staged_source(REPOSITORY, "0" * 40)

    def test_direct_compiler_rejects_untracked_source_residue(self) -> None:
        calls = []

        def fake_run(argv, **_kwargs):
            calls.append(argv)
            return SimpleNamespace(
                stdout=("a" * 40 + "\n" if argv[1:3] == ["rev-parse", "HEAD"] else "?? residue\n")
            )

        with mock.patch.object(compiler.subprocess, "run", side_effect=fake_run):
            with self.assertRaisesRegex(
                compiler.ConfirmationCompilerError, "staged source is dirty"
            ):
                compiler._verify_staged_source(REPOSITORY, "a" * 40)
        self.assertIn("--untracked-files=all", calls[1])

    def test_compiler_dependency_replay_rejects_active_analyzer_or_spec_drift(self) -> None:
        dependencies = {
            "compiler_source": Path(compiler.__file__).resolve(),
            "contract": compiler.CONTRACT_PATH.resolve(),
            "ablation_spec_dependency": compiler.SPEC_PATH.resolve(),
            "development_compiler_dependency": compiler.DEVELOPMENT_COMPILER_PATH.resolve(),
            "annotation_validator_dependency": compiler.ANNOTATION_PATH.resolve(),
            "final_analyzer_dependency": compiler.ANALYZER_PATH.resolve(),
            "release_validator_dependency": compiler.FREEZE_PATH.resolve(),
            "fixture_freeze_dependency": compiler.FIXTURE_FREEZE_PATH.resolve(),
            "n3_confirmation_validator_dependency": compiler.N3_CONFIRMATION_PATH.resolve(),
            "d1_confirmation_validator_dependency": compiler.D1_CONFIRMATION_PATH.resolve(),
            "n3_pilot_dependency": compiler.N3_PILOT_PATH.resolve(),
            "d1_pilot_dependency": compiler.D1_PILOT_PATH.resolve(),
            "d1_server_dependency": compiler.D1_SERVER_PATH.resolve(),
            "resource_qualification_contract_dependency": (
                compiler.RESOURCE_QUALIFICATION_CONTRACT_PATH.resolve()
            ),
        }
        receipt = {
            "source_root": str(REPOSITORY.resolve()),
            "study_commit": "a" * 40,
            "terminal_runtime_source_commit": compiler.TERMINAL_RUNTIME_SOURCE_COMMIT,
            **{key: descriptor(path) for key, path in dependencies.items()},
        }
        with mock.patch.object(compiler, "_verify_staged_source"):
            compiler._validate_compiler_dependencies(receipt)
        original_descriptor = compiler.development.file_descriptor
        for drifted in (compiler.SPEC_PATH.resolve(), compiler.ANALYZER_PATH.resolve()):
            with self.subTest(drifted=drifted.name), mock.patch.object(
                compiler, "_verify_staged_source"
            ), mock.patch.object(
                compiler.development,
                "file_descriptor",
                side_effect=lambda path, drifted=drifted: (
                    {**original_descriptor(path), "sha256": "0" * 64}
                    if Path(path).resolve() == drifted
                    else original_descriptor(path)
                ),
            ), self.assertRaisesRegex(
                compiler.ConfirmationCompilerError,
                "differs from the active staged dependency",
            ):
                compiler._validate_compiler_dependencies(receipt)

    def test_cohort_close_sealer_copy_must_equal_exact_staged_wrapper_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_file = (
                root / "source/workshops/corl2026_world_models/experiments/"
                "forecast_layout/confirmation_evidence_compiler_jobs.py"
            )
            source_file.parent.mkdir(parents=True)
            source_file.write_bytes(b"authoritative sealer bytes\n")
            bundle = root / "seal"
            bundle.mkdir()
            copied = bundle / "sealer_source.py"
            copied.write_bytes(source_file.read_bytes())
            close_path = bundle / "cohort_close_receipt.json"
            relative = {
                "path": "sealer_source.py",
                "sha256": compiler.sha256_file(copied),
                "bytes": copied.stat().st_size,
            }
            observed, path = compiler._validate_staged_sealer_copy(
                relative, close_path=close_path, source_root=root / "source"
            )
            self.assertEqual(path, copied.resolve())
            self.assertEqual(observed["sha256"], compiler.sha256_file(source_file))

            copied.write_bytes(b"different but freshly described\n")
            changed = {
                "path": "sealer_source.py",
                "sha256": compiler.sha256_file(copied),
                "bytes": copied.stat().st_size,
            }
            with self.assertRaisesRegex(
                compiler.ConfirmationCompilerError, "staged authoritative sealer source"
            ):
                compiler._validate_staged_sealer_copy(
                    changed, close_path=close_path, source_root=root / "source"
                )

    def test_empty_terminal_cohort_compiles_exact_not_run_roster_without_science(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_root = root / "raw"
            raw_root.mkdir()
            freeze_path = write_json(root / "release.json", {"fixture": True})
            release_descriptor = descriptor(freeze_path)
            alignment_value = alignment()
            release = {
                "cohort_branch": "reduced_n3",
                "qualified_model_ids": ["N3"],
                "alignment_contracts_by_model": {"N3": {"fake": "entry"}},
            }
            blocks = []
            schedules = {}
            for layout in compiler.LAYOUTS:
                schedule = compiler.n3_confirmation.load_confirmation_block(REPOSITORY, layout)
                schedules[("N3", layout)] = schedule
                attempt = raw_root / "attempts" / layout
                attempt.mkdir(parents=True)
                receipt = {
                    "schema_version": compiler.n3_confirmation.QUEUE_RECEIPT_SCHEMA,
                    "status": "technical_failure",
                    "exit_code": 1,
                    "study_id": compiler.STUDY_ID,
                    "phase": "confirmation",
                    "layout_pair_id": layout,
                    "model_config": "N3",
                    "block_id": schedule.block_id,
                    "source_commit": "a" * 40,
                    "condition_order": list(schedule.condition_order),
                    "cell_ids": list(schedule.cell_ids),
                    "counts": counts(),
                    "prerequisites": {
                        "confirmation_release": {
                            "confirmation_freeze": release_descriptor,
                            "cohort_branch": "reduced_n3",
                            "qualified_model_ids": ["N3"],
                            "alignment_contract": {"fake": "entry"},
                        }
                    },
                    "cell_receipts": [],
                    "raw_attempt_root": str(attempt.resolve()),
                    "raw_attempt_recoverable_on_gm_pvc": True,
                    "failure": {"reason": "not launched"},
                }
                receipt_path = write_json(root / "receipts" / f"{layout}.json", receipt)
                blocks.append({
                    "model_id": "N3",
                    "layout_pair_id": layout,
                    "block_id": schedule.block_id,
                    "schedule_row_sha256": compiler.sha256_bytes(
                        compiler.canonical_bytes(schedule.schedule_row)
                    ),
                    "evidence_form": compiler.FULL_AGGREGATE_FORM,
                    "receipt": descriptor(receipt_path),
                    "failure_cells": [],
                    "queue_jobs": [],
                    "selected_queue_job_ids": [],
                    "d1_pair": None,
                })
            close_path = write_json(root / "close.json", {"close": True})
            manifest = compiler.sign_document({
                "schema_version": compiler.INPUT_SCHEMA,
                "study_id": compiler.STUDY_ID,
                "cohort_branch": "reduced_n3",
                "inventory_finalized_at": "2026-09-13T10:00:00Z",
                "source_root": str(REPOSITORY.resolve()),
                "raw_root": str(raw_root.resolve()),
                "camera_id": "over_shoulder_left_camera",
                "study_commit": "a" * 40,
                "development_release_freeze": release_descriptor,
                "cohort_close_receipt": descriptor(close_path),
                "block_receipts": blocks,
            })
            manifest_path = write_json(root / "manifest.json", manifest)
            output = raw_root / "compiled"
            fake_alignments = {
                "N3": {
                    "entry": {"fake": "entry"},
                    "alignment": alignment_value,
                    "alignment_descriptor": {"path": "/unused", "sha256": "3" * 64, "bytes": 1},
                    "mapping": {},
                    "mapping_descriptor": {"path": "/unused", "sha256": "1" * 64, "bytes": 1},
                }
            }
            def fake_aggregate(**kwargs):
                value = json.loads(Path(kwargs["path"]).read_text())
                return {
                    "descriptor": dict(kwargs["descriptor"]),
                    "path": Path(kwargs["path"]),
                    "receipt": value,
                    "counts": value["counts"],
                    "raw_attempt": Path(value["raw_attempt_root"]),
                    "cell_descriptors": [],
                    "cell_paths": [],
                    "selected_fixture": None,
                    "evidence_form": compiler.FULL_AGGREGATE_FORM,
                }

            close = {
                "sealed_at_utc": "2026-09-13T10:00:00Z",
                "raw_root": str(raw_root.resolve()),
                "_producer_outer_result_state": "pending_current_wrapper_exit",
                "prepared_schedule": descriptor(
                    REPOSITORY / "workshops/corl2026_world_models/execution/20260912/parallel_schedule.json"
                ),
            }
            active_producer = {"fixture": "active-sealer"}
            closures = {
                key: {
                    "failures_by_index": {},
                    "queue_jobs": {},
                    "selected_queue_job_ids": [],
                    "d1_pair": None,
                }
                for key in schedules
            }
            with mock.patch.object(compiler, "_verify_staged_source"), \
                 mock.patch.object(compiler.freeze, "validate_release_freeze", return_value=release), \
                 mock.patch.object(compiler, "_alignment_objects", return_value=fake_alignments), \
                 mock.patch.object(compiler, "_load_cohort_close", return_value=(
                     descriptor(close_path), close_path.resolve(), close
                 )), mock.patch.object(compiler, "_validate_aggregate", side_effect=fake_aggregate), \
                 mock.patch.object(compiler, "_validate_cohort_close_blocks", return_value=closures):
                result = compiler.compile_manifest(
                    manifest_path,
                    compiler.sha256_file(manifest_path),
                    output,
                    active_seal_producer=active_producer,
                )
            self.assertEqual(result["counts"]["planned_cells"], 96)
            self.assertEqual(result["counts"]["not_run"], 96)
            self.assertEqual(result["counts"]["valid_complete"], 0)
            self.assertIs(result["safe_for_request_selection"], False)
            inventory = json.loads((output / "request_inventory.json").read_text())
            self.assertEqual(len(inventory["episode_roster"]), 96)
            self.assertEqual(inventory["requests"], [])
            self.assertEqual(
                {row["recording_status"] for row in inventory["episode_roster"]},
                {"not_run"},
            )
            zero = json.loads((output / "zero_science_receipt.json").read_text())
            compiler.verify_signed(zero, "zero science")
            self.assertEqual(zero["confirmation_jobs_released_by_compiler"], 0)
            self.assertEqual(zero["labels_created_by_compiler"], 0)

            compiler_receipt_path = output / "compiler_receipt.json"
            seal_receipt_path = write_json(
                root / "confirmation_cohort_seal_job_receipt.json",
                compiler.sign_document({
                    "compiler_receipt": descriptor(compiler_receipt_path),
                }),
            )
            replayed = {
                "_producer_outer_result_state": "validated_succeeded_reaped",
                "producer_queue_job": {
                    "seal_job_receipt": descriptor(seal_receipt_path),
                },
            }
            with mock.patch.object(
                compiler,
                "_load_cohort_close",
                return_value=(descriptor(close_path), close_path.resolve(), replayed),
            ) as final_replay, mock.patch.object(compiler, "_verify_staged_source"):
                compiler._validate_compiled_bundle(output)
            final_replay.assert_called_once()

            substituted = json.loads(compiler_receipt_path.read_text())
            substituted.pop("payload_sha256")
            substituted["cohort_close_outer_result_state"] = (
                "validated_succeeded_reaped"
            )
            write_json(compiler_receipt_path, compiler.sign_document(substituted))
            with mock.patch.object(
                compiler,
                "_load_cohort_close",
                return_value=(descriptor(close_path), close_path.resolve(), replayed),
            ) as forged_replay, mock.patch.object(compiler, "_verify_staged_source"):
                with self.assertRaisesRegex(
                    compiler.ConfirmationCompilerError,
                    "exact output authenticated by the sealer job",
                ):
                    compiler._validate_compiled_bundle(output)
            forged_replay.assert_called_once()

    def _failure_fixture(
        self, root: Path, *, aggregate_invalid: int = 1, aggregate_censored: int = 0,
        failure_status: str = "technical_failure", with_sidecars: bool = False,
    ) -> tuple[dict, SimpleNamespace, list[dict], Path]:
        raw = root / "raw"
        attempt = raw / "attempt"
        cell_id = "wmf1__confirmation__C01__N3__original__left"
        cell_root = compiler._cell_root(attempt, 0, cell_id)
        cell_root.mkdir(parents=True)
        failure = {
            "schema_version": compiler.n3_confirmation.CELL_RECEIPT_SCHEMA,
            "study_id": compiler.STUDY_ID,
            "phase": "confirmation",
            "block_id": "block-C01",
            "layout_pair_id": "C01",
            "model_config": "N3",
            "cell_id": cell_id,
            "condition_index": 0,
            "status": failure_status,
            "recorded_stop_reason": (
                "safety_abort" if failure_status == "safety_abort" else "technical_failure"
            ),
            "actions_executed": 0,
            "request_count": 0,
            "server_context_terminal": None,
            "confirmation_prerequisites": {
                "confirmation_release": {},
                "confirmation_fixture_freeze": {},
                "selected_fixture": {},
            },
        }
        failure_path = write_json(cell_root / "technical_failure.json", failure)
        completion = journal = video = context = None
        if with_sidecars:
            completion = descriptor(write_json(cell_root / "recording/completion.json", {"x": 1}))
            journal_path = cell_root / "recording/events.partial.jsonl"
            journal_path.write_text("{}\n", encoding="utf-8")
            journal = descriptor(journal_path)
            video_path = cell_root / "native_simulator/view.mp4"
            video_path.parent.mkdir(parents=True)
            video_path.write_bytes(b"video")
            video = descriptor(video_path)
            if failure_status == "safety_abort":
                context = descriptor(write_json(cell_root / "context_terminal.json", {"done": True}))
                failure["server_context_terminal"] = context
                write_json(failure_path, failure)
        row = {
            "cell_id": cell_id,
            "condition_index": 0,
            "status": failure_status,
            "actions_executed": 0,
            "request_count": 0,
            "artifact_state": "present_hash_bound",
            "failure_receipt": descriptor(failure_path),
            "absence_reason": None,
            "adapter_completion": completion,
            "adapter_journal": journal,
            "source_video": video,
            "context_terminal": context,
        }
        schedule = SimpleNamespace(
            cell_ids=(cell_id, "c2", "c3", "c4"),
            block_id="block-C01",
            layout_pair_id="C01",
        )
        aggregate = {
            "raw_attempt": attempt,
            "receipt": {
                "prerequisites": {
                    "confirmation_release": {},
                    "confirmation_fixture_freeze": {},
                    "selected_fixture": {},
                }
            },
            "counts": counts(
                launched=1,
                invalid=aggregate_invalid,
                censored=aggregate_censored,
            ),
        }
        return aggregate, schedule, [row], raw

    def test_missing_failure_receipt_is_only_explicit_zero_evidence_technical_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            attempt = raw / "attempt"
            cell_id = "wmf1__confirmation__C01__N3__original__left"
            compiler._cell_root(attempt, 0, cell_id).mkdir(parents=True)
            schedule = SimpleNamespace(cell_ids=(cell_id, "c2", "c3", "c4"))
            aggregate = {"raw_attempt": attempt, "counts": counts(launched=1, invalid=1)}
            row = {
                "cell_id": cell_id,
                "condition_index": 0,
                "status": "technical_failure",
                "actions_executed": 0,
                "request_count": 0,
                "artifact_state": "absent_verified_at_cohort_close",
                "failure_receipt": None,
                "absence_reason": "child_terminated_before_failure_receipt",
                "adapter_completion": None,
                "adapter_journal": None,
                "source_video": None,
                "context_terminal": None,
            }
            observed = compiler._validate_failure_cells(
                [row], close_path=root / "close.json", model="N3", layout="C01",
                schedule=schedule, aggregate=aggregate, raw_root=raw,
            )
            self.assertIsNone(observed[0]["failure_receipt"])
            self.assertEqual(observed[0]["status"], "technical_failure")

    def test_failure_category_must_close_aggregate_even_when_totals_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            aggregate, schedule, rows, raw = self._failure_fixture(
                Path(temporary), aggregate_invalid=1, failure_status="safety_abort",
                with_sidecars=True,
            )
            with mock.patch.object(
                compiler.n3_confirmation,
                "validate_failed_confirmation_cell",
                return_value=json.loads(Path(rows[0]["failure_receipt"]["path"]).read_text()),
            ), mock.patch.object(
                compiler.n3_confirmation, "validate_cells_bind_prerequisites"
            ), self.assertRaisesRegex(
                compiler.ConfirmationCompilerError, "do not close the aggregate"
            ):
                    compiler._validate_failure_cells(
                        rows, close_path=Path(temporary) / "close.json", model="N3",
                        layout="C01", schedule=schedule, aggregate=aggregate, raw_root=raw,
                    )

    def test_close_bound_failure_completion_journal_and_video_mutations_fail(self) -> None:
        for artifact in ("failure", "completion", "journal", "video"):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                aggregate, schedule, rows, raw = self._failure_fixture(
                    root, with_sidecars=True
                )
                paths = {
                    "failure": Path(rows[0]["failure_receipt"]["path"]),
                    "completion": Path(rows[0]["adapter_completion"]["path"]),
                    "journal": Path(rows[0]["adapter_journal"]["path"]),
                    "video": Path(rows[0]["source_video"]["path"]),
                }
                with paths[artifact].open("ab") as stream:
                    stream.write(b"mutation")
                with self.assertRaisesRegex(
                    compiler.ConfirmationCompilerError, "(byte count|file hash) changed"
                ):
                    compiler._validate_failure_cells(
                        rows, close_path=root / "close.json", model="N3",
                        layout="C01", schedule=schedule, aggregate=aggregate, raw_root=raw,
                    )

    def test_safety_abort_without_context_terminal_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            aggregate, schedule, rows, raw = self._failure_fixture(
                Path(temporary), aggregate_invalid=0, aggregate_censored=1,
                failure_status="safety_abort", with_sidecars=True,
            )
            rows[0]["context_terminal"] = None
            failure_path = Path(rows[0]["failure_receipt"]["path"])
            failure = json.loads(failure_path.read_text())
            failure["server_context_terminal"] = None
            write_json(failure_path, failure)
            rows[0]["failure_receipt"] = descriptor(failure_path)
            with mock.patch.object(
                compiler.n3_confirmation,
                "validate_failed_confirmation_cell",
                return_value=failure,
            ), mock.patch.object(
                compiler.n3_confirmation, "validate_cells_bind_prerequisites"
            ), self.assertRaisesRegex(
                compiler.ConfirmationCompilerError, "terminal model-context"
            ):
                    compiler._validate_failure_cells(
                        rows, close_path=Path(temporary) / "close.json", model="N3",
                        layout="C01", schedule=schedule, aggregate=aggregate, raw_root=raw,
                    )

    def test_n3_censored_context_cannot_use_unvalidated_terminal_blob(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = write_json(Path(temporary) / "terminal.json", {"passed": True})
            failure = {
                "status": "safety_abort",
                "recorded_stop_reason": "safety_abort",
                "server_context_terminal": descriptor(path),
            }
            failure_path = write_json(Path(temporary) / "technical_failure.json", failure)
            with self.assertRaisesRegex(
                compiler.ConfirmationCompilerError,
                "wmf-n3-terminal-context-receipt-v1",
            ):
                compiler._validate_censored_context_terminal(
                    model="N3", cell_id="cell", layout="C01", condition_index=0,
                    condition="original_left", schedule=SimpleNamespace(
                        block_id="block", condition_order=("original-left",)
                    ),
                    failure_path=failure_path, failure_value=failure,
                    study_commit="a" * 40, d1_runtime=None,
                    completion={"request_count": 1}, context_reset={"value": {}},
                    terminal_descriptor=descriptor(path), terminal_path=path.resolve(),
                    request_descriptors=[], request_values=[], raw_root=Path(temporary),
                )

    def test_d1_pre_finalize_manifest_cannot_be_terminal_context_proof(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = write_json(Path(temporary) / "episode_manifest.json", {
                "schema_version": "wmf-d1-episode-manifest-v1",
                "two_rank_reset": {"passed": True},
            })
            failure = {
                "status": "safety_abort",
                "recorded_stop_reason": "safety_abort",
                "server_context_terminal": descriptor(path),
            }
            failure_path = write_json(Path(temporary) / "technical_failure.json", failure)
            with self.assertRaisesRegex(
                compiler.ConfirmationCompilerError,
                "wmf-d1-terminal-context-receipt-v1",
            ):
                compiler._validate_censored_context_terminal(
                    model="D1", cell_id="cell", layout="C01", condition_index=0,
                    condition="original_left", schedule=SimpleNamespace(
                        block_id="block", condition_order=("original-left",)
                    ),
                    failure_path=failure_path, failure_value=failure,
                    study_commit="a" * 40,
                    d1_runtime={
                        "study_commit": "a" * 40,
                        "run_id": "run",
                        "server_job_id": "server",
                        "simulator_job_id": "simulator",
                        "simulator_worker_role": "wmf-forecast-0912-worker-05",
                    },
                    completion={"request_count": 1}, context_reset={"value": {}},
                    terminal_descriptor=descriptor(path), terminal_path=path.resolve(),
                    request_descriptors=[], request_values=[], raw_root=Path(temporary),
                )

    def test_native_n3_terminal_receipt_promotes_only_safety_censored_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            terminal = {
                "schema_version": compiler.N3_CONTEXT_TERMINAL_SCHEMA,
                "status": "passed",
                "terminal_state": "context_closed",
                "model_config": "N3",
                "study_id": compiler.STUDY_ID,
                "phase": "confirmation",
                "block_id": "block",
                "layout_pair_id": "C01",
                "cell_id": "cell",
                "condition_index": 0,
                "stop_reason": "safety_abort",
                "actions_executed": 7,
                "request_count": 1,
                "server_request_count": 1,
                "context_active_after_end": False,
                "model_capture_active_after_end": False,
                "server_context_id": "server-context",
                "client_session_id": "client-session",
            }
            terminal_path = write_json(root / "terminal.json", terminal)
            terminal_descriptor = descriptor(terminal_path)
            failure = {
                "status": "safety_abort",
                "recorded_stop_reason": "safety_abort",
                "server_context_terminal": terminal_descriptor,
            }
            failure_path = write_json(root / "technical_failure.json", failure)
            validator = mock.Mock(return_value=failure)
            with mock.patch.object(
                compiler.n3_confirmation,
                "validate_failed_confirmation_cell",
                validator,
            ):
                observed = compiler._validate_censored_context_terminal(
                    model="N3", cell_id="cell", layout="C01", condition_index=0,
                    condition="original_left", schedule=SimpleNamespace(
                        block_id="block", condition_order=("original-left",)
                    ),
                    failure_path=failure_path, failure_value=failure,
                    study_commit="a" * 40, d1_runtime=None,
                    completion={"actions_executed": 7, "request_count": 1},
                    context_reset={"binding": {"payload_sha256": "b" * 64}},
                    terminal_descriptor=terminal_descriptor,
                    terminal_path=terminal_path.resolve(),
                    request_descriptors=[{"sha256": "c" * 64}],
                    request_values=[{"request_index": 0}], raw_root=root,
                )
            validator.assert_called_once()
            self.assertEqual(observed["schema_version"], compiler.N3_CONTEXT_TERMINAL_SCHEMA)
            self.assertEqual(observed["terminal_runtime_source_commit"], "03732d3c6fa37c26a3ab2e8608a80d076fdcc33a")
            self.assertEqual(observed["stop_reason"], "safety_abort")

    def test_native_d1_terminal_receipt_is_bound_to_selected_pair_and_final_reset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            terminal = {
                "schema_version": compiler.D1_CONTEXT_TERMINAL_SCHEMA,
                "status": "passed",
                "terminal_state": "context_closed",
                "model_config": "D1",
                "study_id": compiler.STUDY_ID,
                "phase": "confirmation",
                "block_id": "block",
                "layout_pair_id": "C01",
                "cell_id": "cell",
                "condition_index": 0,
                "stop_reason": "safety_abort",
                "actions_executed": 7,
                "request_count": 1,
                "server_request_count": 1,
                "episode_bookkeeping_cleared": True,
                "context_active_after_finalize": False,
                "episode_manifest": {"path": "/future/episode_manifest.json", "sha256": "d" * 64},
                "final_two_rank_reset": {"reset_id": "post-episode-reset", "rank_receipts": [{"rank": 0}, {"rank": 1}]},
                "server_context_id": "episode-context",
                "client_session_id": "client-session",
            }
            terminal_path = write_json(root / "terminal.json", terminal)
            terminal_descriptor = descriptor(terminal_path)
            failure = {
                "status": "safety_abort",
                "recorded_stop_reason": "safety_abort",
                "server_context_terminal": terminal_descriptor,
                "study_commit": "a" * 40,
                "run_id": "run",
                "server_job_id": "server",
                "simulator_job_id": "simulator",
            }
            failure_path = write_json(root / "technical_failure.json", failure)
            validator = mock.Mock(return_value=failure)
            runtime = {
                "study_commit": "a" * 40,
                "run_id": "run",
                "server_job_id": "server",
                "simulator_job_id": "simulator",
                "simulator_worker_role": "wmf-forecast-0912-worker-05",
            }
            with mock.patch.object(
                compiler.d1_confirmation, "_configured_for_validation",
                return_value=nullcontext(),
            ), mock.patch.object(
                compiler.d1_confirmation,
                "validate_failed_confirmation_cell",
                validator,
            ):
                observed = compiler._validate_censored_context_terminal(
                    model="D1", cell_id="cell", layout="C01", condition_index=0,
                    condition="original_left", schedule=SimpleNamespace(
                        block_id="block", condition_order=("original-left",)
                    ),
                    failure_path=failure_path, failure_value=failure,
                    study_commit="a" * 40, d1_runtime=runtime,
                    completion={"actions_executed": 7, "request_count": 1},
                    context_reset={"binding": {"payload_sha256": "b" * 64}},
                    terminal_descriptor=terminal_descriptor,
                    terminal_path=terminal_path.resolve(),
                    request_descriptors=[{"sha256": "c" * 64}],
                    request_values=[{"request_index": 0}], raw_root=root,
                )
            validator.assert_called_once()
            self.assertEqual(observed["schema_version"], compiler.D1_CONTEXT_TERMINAL_SCHEMA)
            self.assertEqual(
                observed["final_two_rank_reset_sha256"],
                compiler.sha256_bytes(compiler.canonical_bytes(terminal["final_two_rank_reset"])),
            )

            substituted = dict(failure, run_id="another-run")
            write_json(failure_path, substituted)
            with mock.patch.object(
                compiler.d1_confirmation, "_configured_for_validation",
                return_value=nullcontext(),
            ), mock.patch.object(
                compiler.d1_confirmation,
                "validate_failed_confirmation_cell",
                return_value=substituted,
            ), self.assertRaisesRegex(
                compiler.ConfirmationCompilerError, "detached from its selected queue pair"
            ):
                compiler._validate_censored_context_terminal(
                    model="D1", cell_id="cell", layout="C01", condition_index=0,
                    condition="original_left", schedule=SimpleNamespace(
                        block_id="block", condition_order=("original-left",)
                    ),
                    failure_path=failure_path, failure_value=substituted,
                    study_commit="a" * 40, d1_runtime=runtime,
                    completion={"actions_executed": 7, "request_count": 1},
                    context_reset={"binding": {"payload_sha256": "b" * 64}},
                    terminal_descriptor=terminal_descriptor,
                    terminal_path=terminal_path.resolve(),
                    request_descriptors=[{"sha256": "c" * 64}],
                    request_values=[{"request_index": 0}], raw_root=root,
                )

    def test_technical_completion_allows_action_cap_but_not_beyond_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            journal = root / "events.partial.jsonl"
            journal.write_text("placeholder\n", encoding="utf-8")
            identity = {
                "attempt_id": "attempt",
                "cell_id": "cell",
                "stage": "confirmation",
                "layout_pair_id": "C01",
                "layout_arm": "original",
                "command": "left",
                "prompt": "prompt",
                "model_config": "N3",
                "effective_seed": 17,
            }
            completion = {
                "schema_version": "wmf-forecast-recording-attempt-v1",
                "study_id": compiler.STUDY_ID,
                "identity": identity,
                "actions_executed": 450,
                "observation_count": 451,
                "request_count": 15,
                "success_configured_as_termination": False,
                "behavioral_result_valid": False,
                "technical_invalid": True,
                "right_censored": False,
                "stop_reason": "technical_failure",
                "request_execution": [],
                "validation_errors": ["final receipt failed"],
                "event_count": 2,
                "journal_tail_sha256": "f" * 64,
                "journal_path": str(journal.resolve()),
            }
            rows = [
                {"kind": "attempt_started", "payload": {}},
                {"kind": "attempt_finalized", "payload": dict(completion)},
            ]
            completion_path = write_json(root / "completion.json", completion)
            with mock.patch.object(
                compiler.development, "_verify_journal", return_value=(rows, "f" * 64)
            ):
                observed, _, _ = compiler._validate_completion(
                    completion_path=completion_path, journal_path=journal.resolve(),
                    expected_cell_id="cell", model="N3", layout="C01",
                    condition="original_left", status="technical_invalid",
                    expected_actions=450, expected_prompt="prompt",
                    expected_effective_seed=17,
                )
            self.assertEqual(observed["actions_executed"], 450)
            completion["actions_executed"] = 451
            completion["observation_count"] = 452
            rows[-1]["payload"] = dict(completion)
            write_json(completion_path, completion)
            with mock.patch.object(
                compiler.development, "_verify_journal", return_value=(rows, "f" * 64)
            ), self.assertRaisesRegex(
                compiler.ConfirmationCompilerError, "technical status differs"
            ):
                compiler._validate_completion(
                    completion_path=completion_path, journal_path=journal.resolve(),
                    expected_cell_id="cell", model="N3", layout="C01",
                    condition="original_left", status="technical_invalid",
                    expected_actions=451, expected_prompt="prompt",
                    expected_effective_seed=17,
                )

    def test_compiled_bundle_consumer_replay_accepts_technical_450_without_science(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            planned = compiler._planned_cells("reduced_n3")
            technical_id = sorted(planned)[0]
            model, layout, condition = planned[technical_id]
            cell_root = bundle / "cells/n3" / compiler._safe_component(technical_id)
            actions = [
                {
                    "action_index": index,
                    "request_index": index // 32,
                    "executed_action_sha256": compiler.sha256_bytes(
                        f"{technical_id}:{index}".encode()
                    ),
                    "control_timestamp": f"2026-09-13T00:00:00.{index:06d}Z",
                    "physics_step_id": f"physics-step:{index + 1}",
                    "camera_frame_id": f"camera:native-int:{index + 1}",
                }
                for index in range(450)
            ]
            action_path = write_json(
                cell_root / "action_manifest.json",
                compiler.sign_document({
                    "schema_version": compiler.ACTION_MANIFEST_SCHEMA,
                    "study_id": compiler.STUDY_ID,
                    "cell_id": technical_id,
                    "recording_id": "technical-recording",
                    "model_id": model,
                    "executed_action_count": 450,
                    "actions": actions,
                }),
            )
            recording_path = write_json(
                cell_root / "recording_receipt.json",
                compiler.sign_document({
                    "schema_version": compiler.RECORDING_RECEIPT_SCHEMA,
                    "study_id": compiler.STUDY_ID,
                    "receipt_id": "technical-recording-receipt",
                    "stage": "confirmation",
                    "cell_id": technical_id,
                    "recording_id": "technical-recording",
                    "model_id": model,
                    "layout_pair_id": layout,
                    "condition_id": condition,
                    "recording_status": "technical_invalid",
                    "executed_action_count": 450,
                    "censor_reason": "recording_integrity_failure",
                    "source_video_id": None,
                    "source_video_sha256": None,
                    "action_manifest_path": "action_manifest.json",
                    "action_manifest_sha256": compiler.sha256_file(action_path),
                }),
            )
            roster = []
            for cell_id, (cell_model, cell_layout, cell_condition) in sorted(planned.items()):
                technical = cell_id == technical_id
                roster.append({
                    "cell_id": cell_id,
                    "recording_id": "technical-recording" if technical else None,
                    "model_id": cell_model,
                    "layout_pair_id": cell_layout,
                    "condition_id": cell_condition,
                    "recording_status": "technical_invalid" if technical else "not_run",
                    "executed_action_count": 450 if technical else None,
                    "censor_reason": "recording_integrity_failure" if technical else None,
                    "recording_receipt_path": (
                        str(recording_path.relative_to(bundle)) if technical else None
                    ),
                    "recording_receipt_sha256": (
                        compiler.sha256_file(recording_path) if technical else None
                    ),
                    "action_manifest_path": (
                        str(action_path.relative_to(bundle)) if technical else None
                    ),
                    "action_manifest_sha256": (
                        compiler.sha256_file(action_path) if technical else None
                    ),
                    "source_video_id": None,
                    "source_video_sha256": None,
                })
            inventory_path = write_json(bundle / "request_inventory.json", {
                "schema_version": compiler.REQUEST_INVENTORY_SCHEMA,
                "study_id": compiler.STUDY_ID,
                "stage": "confirmation",
                "cohort_branch": "reduced_n3",
                "inventory_complete": True,
                "inventory_finalized_at": "2026-09-20T01:48:52Z",
                "annotation_state": "not_started",
                "episode_roster": roster,
                "alignment_contracts": [],
                "requests": [],
            })
            inventory_descriptor = descriptor(inventory_path)
            wrapper_path = write_json(
                bundle / "request_inventory_receipt.json",
                compiler.sign_document({
                    "schema_version": compiler.REQUEST_INVENTORY_RECEIPT_SCHEMA,
                    "request_inventory": inventory_descriptor,
                }),
            )
            provenance_path = write_json(
                bundle / "request_provenance.json",
                compiler.sign_document({
                    "schema_version": compiler.PROVENANCE_SCHEMA,
                    "request_inventory_sha256": inventory_descriptor["sha256"],
                    "requests": [],
                }),
            )
            private_path = write_json(
                bundle / "private_video_inventory.json",
                compiler.sign_document({
                    "schema_version": compiler.PRIVATE_VIDEO_SCHEMA,
                    "visibility": "private_source_evidence_not_blind_annotation_media",
                    "videos": [],
                }),
            )
            zero_path = write_json(
                bundle / "zero_science_receipt.json",
                compiler.sign_document({
                    "schema_version": compiler.ZERO_SCIENCE_SCHEMA,
                    "study_id": compiler.STUDY_ID,
                    "stage": "confirmation_evidence_compilation",
                    "model_runtime_loads": 0,
                    "model_servers_started": 0,
                    "model_requests_issued_by_compiler": 0,
                    "simulator_processes_started": 0,
                    "physical_resets": 0,
                    "robot_episodes": 0,
                    "behavioral_actions_executed_by_compiler": 0,
                    "behavioral_cells_launched_by_compiler": 0,
                    "labels_created_by_compiler": 0,
                    "confirmation_jobs_released_by_compiler": 0,
                }),
            )
            close_path = write_json(root / "cohort-close.json", {"fixture": True})
            close_descriptor = descriptor(close_path)
            input_manifest = compiler.sign_document({
                "schema_version": compiler.INPUT_SCHEMA,
                "study_id": compiler.STUDY_ID,
                "cohort_branch": "reduced_n3",
                "inventory_finalized_at": "2026-09-20T01:48:52Z",
                "source_root": str(REPOSITORY.resolve()),
                "raw_root": str(root.resolve()),
                "camera_id": "over_shoulder_left_camera",
                "study_commit": "a" * 40,
                "development_release_freeze": close_descriptor,
                "cohort_close_receipt": close_descriptor,
                "block_receipts": [],
            })
            manifest_path = write_json(root / "compiler-input.json", input_manifest)
            compiler_receipt = compiler.sign_document({
                "schema_version": compiler.COMPILER_SCHEMA,
                "study_id": compiler.STUDY_ID,
                "cohort_branch": "reduced_n3",
                "study_commit": "a" * 40,
                "source_root": str(REPOSITORY.resolve()),
                "cohort_close_outer_result_state": "pending_current_wrapper_exit",
                "input_manifest": descriptor(manifest_path),
                "cohort_close_receipt": close_descriptor,
                "outputs": {
                    "request_inventory": inventory_descriptor,
                    "request_inventory_receipt": descriptor(wrapper_path),
                    "request_provenance": descriptor(provenance_path),
                    "private_video_inventory": descriptor(private_path),
                    "zero_science_receipt": descriptor(zero_path),
                },
                "scientific_results_computed": False,
                "labels_created": False,
                "confirmation_released": False,
            })
            write_json(bundle / "compiler_receipt.json", compiler_receipt)
            replayed_close = {"_producer_outer_result_state": "pending_current_wrapper_exit"}
            with mock.patch.object(
                compiler, "_validate_compiler_dependencies"
            ), mock.patch.object(
                compiler, "_load_cohort_close",
                return_value=(close_descriptor, close_path.resolve(), replayed_close),
            ), mock.patch.object(
                compiler.annotation, "_validate_alignment_contracts"
            ):
                compiler._validate_compiled_bundle(
                    bundle, active_seal_producer={"job_id": "self"}
                )

    def test_technical_prefix_retains_deep_request_lifecycle_outside_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            completion_path = root / "recording/completion.json"
            completion_path.parent.mkdir(parents=True)
            official_path = write_json(root / "server/request.json", {"official": True})
            official_descriptor = descriptor(official_path)
            model_input = recorder_payload(
                "model_request",
                {
                    "extracted_preprocessing_output": {"pixels": "retained"},
                    "wire_request": {"cell_id": "cell"},
                },
            )
            response = recorder_payload(
                "model_response",
                {
                    "raw_response": {
                        "wmf_server_request_receipt": official_descriptor,
                    },
                    "future_evidence": {"decoded": "retained"},
                },
            )
            chunks = recorder_payload("action_chunks", {"chunk": "retained"})
            limits = compiler.development.MODEL_LIMITS["N3"]
            execution = {
                "request_index": 0,
                "action_step_start": 0,
                "returned_actions": limits["returned_actions"],
                "eligible_executable_prefix_actions": limits["executed_prefix"],
                "executed_actions": 1,
                "unused_executable_prefix_actions": limits["executed_prefix"] - 1,
                "returned_actions_outside_executable_prefix": (
                    limits["returned_actions"] - limits["executed_prefix"]
                ),
                "current_observation_id": "obs_000000",
                "preceding_observation_id": None,
                "future_kinds": limits["future_kinds"],
                "action_chunks_artifact": chunks,
            }
            completion = {
                "actions_executed": 1,
                "observation_count": 2,
                "request_count": 1,
                "request_execution": [execution],
            }
            rows = [
                {
                    "sequence": 1,
                    "kind": "model_request_packed",
                    "payload": {
                        "request_index": 0,
                        "action_step_start": 0,
                        "current_observation_id": "obs_000000",
                        "preceding_observation_id": None,
                        "returned_action_horizon": limits["returned_actions"],
                        "executed_prefix_horizon": limits["executed_prefix"],
                        "required_future_evidence": limits["required_future_evidence"],
                        "model_request_artifact": model_input,
                        "pack_monotonic_ns": 10,
                    },
                },
                {
                    "sequence": 2,
                    "kind": "model_request_sent",
                    "payload": {"request_index": 0, "send_monotonic_ns": 11},
                },
                {
                    "sequence": 3,
                    "kind": "model_response_received",
                    "payload": {
                        "request_index": 0,
                        "receive_monotonic_ns": 12,
                        "response_artifact": response,
                        "future_kinds": limits["future_kinds"],
                    },
                },
                {
                    "sequence": 4,
                    "kind": "model_request_completed",
                    "payload": {
                        "request_index": 0,
                        "action_chunks_artifact": chunks,
                        "returned_action_shape": [limits["returned_actions"], 8],
                        "executable_action_shape": [limits["executed_prefix"], 8],
                        "missing_future_evidence": [],
                    },
                },
            ]
            retained_action = {
                "action_index": 0,
                "request_index": 0,
                "executed_action_sha256": "e" * 64,
                "control_timestamp": "2026-09-13T00:00:00.000000Z",
                "physics_step_id": "physics-step:1",
                "camera_frame_id": "camera:native-int:1",
            }
            deep_replay = mock.Mock(return_value={
                "request_descriptors": [official_descriptor],
                "request_values": [{"official": True}],
                "response_descriptors": [response],
                "packed_request_bindings": [{"payload_sha256": model_input["payload_sha256"]}],
                "observations": [{"observation_id": "obs_000000"}],
                "actions": [retained_action],
            })
            with mock.patch.object(
                compiler.development, "_verify_request_receipt",
                return_value={"official": True},
            ), mock.patch.object(
                compiler, "_technical_response_equivalence"
            ) as response_replay, mock.patch.object(
                compiler, "_recorded_action_and_request_evidence", deep_replay
            ):
                observed = compiler._technical_retained_evidence(
                    completion=completion, completion_path=completion_path,
                    rows=rows, cell={"cell_id": "cell", "effective_seed": 17},
                    model="N3", camera_id="camera", raw_root=root,
                )
            deep_replay.assert_called_once()
            response_replay.assert_called_once()
            self.assertEqual(observed["actions"], [retained_action])
            self.assertEqual(observed["technical_request_evidence"][0]["stage"], "completed")
            self.assertEqual(
                observed["technical_request_evidence"][0]["official_request_receipt"],
                official_descriptor,
            )

            mutated_rows = json.loads(json.dumps(rows))
            mutated_rows[0]["payload"]["model_request_artifact"]["payload_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                compiler.development.CompilerError, "payload descriptor hash changed"
            ):
                compiler._technical_retained_evidence(
                    completion=completion, completion_path=completion_path,
                    rows=mutated_rows, cell={"cell_id": "cell", "effective_seed": 17},
                    model="N3", camera_id="camera", raw_root=root,
                )

    @staticmethod
    def _attempt(
        *, job_ids: list[str], before: list[dict], new: list[dict], start: int,
        end: int, status: str = "technical_failure", invalid: int = 0,
        censored: int = 0, launched: int | None = None,
    ) -> dict:
        after = before + new
        launched_count = (
            len(after) + (1 if invalid or censored else 0)
            if launched is None else launched
        )
        count_value = counts(
            launched=launched_count,
            completed=len(after),
            invalid=invalid,
            censored=censored,
        )
        count_value["resumed_valid_behavioral_cells"] = len(before)
        count_value["newly_launched_behavioral_cells"] = launched_count - len(before)
        return {
            "job_ids": job_ids,
            "started_at": datetime.fromtimestamp(start, tz=timezone.utc),
            "ended_at": datetime.fromtimestamp(end, tz=timezone.utc),
            "before": before,
            "new": new,
            "after": after,
            "aggregate": {"counts": count_value, "receipt": {"status": status}},
            "failures_by_index": {},
            "d1_pair": None,
        }

    def test_retry_tail_is_derived_from_unique_maximal_prefix(self) -> None:
        cells = [
            {"path": f"/cell-{index}", "sha256": str(index + 1) * 64, "bytes": 1}
            for index in range(4)
        ]
        attempts = [
            self._attempt(
                job_ids=["attempt-1"], before=[], new=cells[:2],
                start=10, end=20, invalid=1,
            ),
            self._attempt(
                job_ids=["attempt-2"], before=cells[:2], new=cells[2:],
                start=30, end=40, status="passed", launched=4,
            ),
        ]
        tail = compiler._derive_attempt_tail(attempts, model="N3", layout="C01")
        self.assertEqual(tail["job_ids"], ["attempt-2"])
        self.assertEqual(tail["after"], cells)

    def test_retry_after_safety_abort_is_rejected(self) -> None:
        cell = {"path": "/cell-0", "sha256": "1" * 64, "bytes": 1}
        attempts = [
            self._attempt(
                job_ids=["censored"], before=[], new=[cell], start=10, end=20,
                censored=1,
            ),
            self._attempt(
                job_ids=["retry"], before=[cell], new=[], start=30, end=40,
                invalid=1,
            ),
        ]
        with self.assertRaisesRegex(
            compiler.ConfirmationCompilerError, "retry after passed or safety-censored"
        ):
            compiler._derive_attempt_tail(attempts, model="N3", layout="C01")

    def test_retry_overlap_and_prefix_regression_are_rejected(self) -> None:
        first = {"path": "/cell-0", "sha256": "1" * 64, "bytes": 1}
        replacement = {"path": "/cell-X", "sha256": "2" * 64, "bytes": 1}
        base = self._attempt(
            job_ids=["first"], before=[], new=[first], start=10, end=20, invalid=1
        )
        overlapping = self._attempt(
            job_ids=["second"], before=[first], new=[], start=19, end=30, invalid=1
        )
        with self.assertRaisesRegex(compiler.ConfirmationCompilerError, "overlap"):
            compiler._derive_attempt_tail([base, overlapping], model="N3", layout="C01")
        replaced = self._attempt(
            job_ids=["second"], before=[replacement], new=[], start=30, end=40,
            invalid=1,
        )
        with self.assertRaisesRegex(compiler.ConfirmationCompilerError, "exact prior valid prefix"):
            compiler._derive_attempt_tail([base, replaced], model="N3", layout="C01")

    @staticmethod
    def _normalized_queue_job(
        job_id: str, *, start: int, end: int, selected: bool,
        mode: str = "queue", run_id: str | None = None,
    ) -> dict:
        receipt_descriptor = {
            "path": f"/{job_id}-runtime.json",
            "sha256": ("a" if job_id.endswith("1") else "b") * 64,
            "bytes": 1,
        }
        queue_descriptor = {
            "path": f"/{job_id}-queue.json",
            "sha256": ("c" if job_id.endswith("1") else "d") * 64,
            "bytes": 1,
        }
        claim_time = start - 1
        return {
            "job_id": job_id,
            "model_id": "N3" if mode == "queue" else "D1",
            "layout_pair_id": "C01",
            "block_id": "block",
            "mode": mode,
            "run_id": run_id,
            "selected_for_block_evidence": selected,
            "descriptor": queue_descriptor,
            "descriptor_value": {
                "argv": [],
                "role": "wmf-forecast-0912-worker-05",
            },
            "claim_value": {
                "claimed_unix": float(claim_time),
                "claimed_at": datetime.fromtimestamp(
                    claim_time, tz=timezone.utc
                ).isoformat(),
            },
            "result_value": {
                "started_at": datetime.fromtimestamp(start, tz=timezone.utc).isoformat(),
                "ended_at": datetime.fromtimestamp(end, tz=timezone.utc).isoformat(),
            },
            "runtime_terminal_evidence": {
                "runtime_receipt": receipt_descriptor,
                "runtime_value": {"counts": {}},
                "failure_cells": [],
                "protocol_terminal": None,
            },
        }

    def test_caller_cannot_select_an_earlier_retry(self) -> None:
        first = self._normalized_queue_job(
            "attempt1", start=10, end=20, selected=True
        )
        second = self._normalized_queue_job(
            "attempt2", start=30, end=40, selected=False
        )
        cells = [
            {"path": f"/cell-{index}", "sha256": str(index + 1) * 64, "bytes": 1}
            for index in range(4)
        ]

        def fake_aggregate(**kwargs):
            is_first = str(kwargs["path"]).endswith("attempt1-runtime.json")
            count_value = counts(
                launched=1 if is_first else 4,
                completed=0 if is_first else 4,
                invalid=1 if is_first else 0,
            )
            return {
                "descriptor": kwargs["descriptor"],
                "receipt": {
                    "status": "technical_failure" if is_first else "passed",
                    "queue_descriptor": first["descriptor"] if is_first else second["descriptor"],
                },
                "counts": count_value,
                "evidence_form": compiler.FULL_AGGREGATE_FORM,
                "cell_descriptors": [] if is_first else cells,
            }

        def fake_prefix(*, aggregate, **_kwargs):
            return ([], [], []) if not aggregate["cell_descriptors"] else (
                [], list(aggregate["cell_descriptors"]), list(aggregate["cell_descriptors"])
            )

        with mock.patch.object(compiler, "_validate_queue_job", side_effect=lambda value, **_: value), \
             mock.patch.object(compiler, "_validate_aggregate", side_effect=fake_aggregate), \
             mock.patch.object(compiler, "_validate_failure_cells", return_value={}), \
             mock.patch.object(compiler, "_attempt_prefix", side_effect=fake_prefix):
            with self.assertRaisesRegex(
                compiler.ConfirmationCompilerError, "caller-selected lifecycle"
            ):
                compiler._validate_block_attempt_lifecycles(
                    queue_rows=[first, second], close_path=Path("/close.json"),
                    model="N3", layout="C01", schedule=SimpleNamespace(block_id="block"),
                    study_commit="a" * 40, source_root=REPOSITORY,
                    raw_root=Path("/raw"), release_descriptor={}, release_freeze={},
                )

    def test_d1_retry_interval_uses_both_server_and_simulator_results(self) -> None:
        def pair(run: str, *, server_start: int, server_end: int,
                 simulator_start: int, simulator_end: int, selected: bool) -> list[dict]:
            server_id = f"{run}-server"
            simulator_id = f"{run}-sim"
            server = self._normalized_queue_job(
                server_id, start=server_start, end=server_end, selected=selected,
                mode="server-job", run_id=run,
            )
            simulator = self._normalized_queue_job(
                simulator_id, start=simulator_start, end=simulator_end, selected=selected,
                mode="simulator-job", run_id=run,
            )
            common = ["/usr/bin/python3", "/runner.py", "unused"]
            server["descriptor_value"]["argv"] = common + [
                "--simulator-job-id", simulator_id
            ]
            simulator["descriptor_value"]["argv"] = common + [
                "--server-job-id", server_id
            ]
            simulator["runtime_terminal_evidence"]["protocol_terminal"] = {
                "path": f"/{run}-terminal.json", "sha256": "e" * 64, "bytes": 1
            }
            return [server, simulator]

        rows = pair(
            "run1", server_start=10, server_end=50,
            simulator_start=20, simulator_end=40, selected=False,
        ) + pair(
            "run2", server_start=45, server_end=80,
            simulator_start=50, simulator_end=70, selected=True,
        )

        def fake_aggregate(**kwargs):
            return {
                "descriptor": kwargs["descriptor"],
                "receipt": {"status": "technical_failure"},
                "counts": counts(),
                "evidence_form": compiler.FULL_AGGREGATE_FORM,
                "cell_descriptors": [],
            }

        with mock.patch.object(compiler, "_validate_queue_job", side_effect=lambda value, **_: value), \
             mock.patch.object(compiler, "_validate_aggregate", side_effect=fake_aggregate), \
             mock.patch.object(compiler, "_validate_failure_cells", return_value={}), \
             mock.patch.object(compiler, "_attempt_prefix", return_value=([], [], [])), \
             mock.patch.object(compiler, "_validate_d1_pair", return_value={}):
            with self.assertRaisesRegex(
                compiler.ConfirmationCompilerError, "overlap or have ambiguous ordering"
            ):
                compiler._validate_block_attempt_lifecycles(
                    queue_rows=rows, close_path=Path("/close.json"),
                    model="D1", layout="C01", schedule=SimpleNamespace(block_id="block"),
                    study_commit="a" * 40, source_root=REPOSITORY,
                    raw_root=Path("/raw"), release_descriptor={}, release_freeze={},
                )

    def test_minimal_d1_aggregate_cannot_downgrade_full_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt = root / "raw/attempt"
            attempt.mkdir(parents=True)
            receipt = {
                "schema_version": compiler.d1_confirmation.SIMULATOR_RECEIPT_SCHEMA,
                "status": "technical_failure",
                "exit_code": 1,
                "run_id": "run",
                "server_job_id": "server",
                "simulator_job_id": "simulator",
                "study_commit": "a" * 40,
                "block_id": "block",
                "all_simulator_children_reaped": True,
                "raw_attempt_root": str(attempt),
                "counts": counts(),
                "cell_receipts": [],
                "prerequisites": {},
            }
            path = write_json(root / "receipt.json", receipt)
            with self.assertRaisesRegex(compiler.ConfirmationCompilerError, "cannot be downgraded"):
                compiler._validate_aggregate(
                    descriptor=descriptor(path), path=path.resolve(), model="D1", layout="C01",
                    schedule=SimpleNamespace(block_id="block"), study_commit="a" * 40,
                    source_root=REPOSITORY, raw_root=root / "raw",
                    release_descriptor={}, release_freeze={},
                    evidence_form=compiler.D1_ZERO_LAUNCH_FORM,
                )

    def test_minimal_d1_zero_launch_form_has_exact_zero_of_four_roster_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt = root / "raw/attempt"
            attempt.mkdir(parents=True)
            receipt = {
                "schema_version": compiler.d1_confirmation.SIMULATOR_RECEIPT_SCHEMA,
                "status": "technical_failure",
                "exit_code": 1,
                "run_id": "run",
                "server_job_id": "server",
                "simulator_job_id": "simulator",
                "study_commit": "a" * 40,
                "block_id": "block",
                "all_simulator_children_reaped": True,
                "raw_attempt_root": str(attempt),
            }
            path = write_json(root / "receipt.json", receipt)
            observed = compiler._validate_aggregate(
                descriptor=descriptor(path), path=path.resolve(), model="D1", layout="C01",
                schedule=SimpleNamespace(block_id="block"), study_commit="a" * 40,
                source_root=REPOSITORY, raw_root=root / "raw",
                release_descriptor={}, release_freeze={},
                evidence_form=compiler.D1_ZERO_LAUNCH_FORM,
            )
            self.assertEqual(observed["counts"]["launched_behavioral_cells"], 0)
            self.assertEqual(observed["counts"]["unrun_behavioral_cells"], 4)

    def test_d1_pair_rejects_server_simulator_status_contradiction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aggregate_path = write_json(root / "simulator.json", {"receipt": True})
            terminal_path = write_json(root / "terminal.json", {
                "schema_version": compiler.d1_confirmation.pilot.SIMULATOR_TERMINAL_SCHEMA,
                "run_id": "run", "server_job_id": "server", "simulator_job_id": "sim",
                "block_id": "block", "status": "technical_failure",
                "all_simulator_children_reaped": True, "safe_for_server_shutdown": True,
                "simulator_receipt": descriptor(aggregate_path),
                "simulator_claim_sha256": None,
            })
            server_path = write_json(root / "server.json", {
                "schema_version": compiler.d1_confirmation.SERVER_RECEIPT_SCHEMA,
                "run_id": "run", "server_job_id": "server",
                "paired_simulator_job_id": "sim", "study_commit": "a" * 40,
                "block_id": "block", "status": "passed", "exit_code": 0,
                "all_server_children_reaped": True, "server_process_exit": None,
                "server_ready": None, "queue_descriptor": {},
                "simulator_terminal": descriptor(terminal_path), "simulator_claim": None,
            })
            aggregate = {
                "receipt": {
                    "status": "technical_failure", "exit_code": 1, "run_id": "run",
                    "server_job_id": "server", "simulator_job_id": "sim",
                    "all_simulator_children_reaped": True, "simulator_claim": None,
                    "queue_descriptor": {},
                },
                "counts": compiler._zero_launch_counts(),
                "evidence_form": compiler.D1_ZERO_LAUNCH_FORM,
            }
            jobs = {
                "server": {"mode": "server-job", "run_id": "run", "descriptor": {},
                           "result_value": {"status": "succeeded", "returncode": 0}},
                "sim": {"mode": "simulator-job", "run_id": "run", "descriptor": {},
                        "result_value": {"status": "failed", "returncode": 1}},
            }
            with self.assertRaisesRegex(
                compiler.ConfirmationCompilerError, "server terminal receipt is incomplete"
            ):
                compiler._validate_d1_pair(
                    {
                        "run_id": "run", "server_job_id": "server", "simulator_job_id": "sim",
                        "server_receipt": descriptor(server_path),
                        "simulator_terminal": descriptor(terminal_path),
                    },
                    close_path=root / "close.json", layout="C01", block_id="block",
                    study_commit="a" * 40, aggregate=aggregate,
                    aggregate_descriptor=descriptor(aggregate_path), selected_jobs=jobs,
                )

    def _minimal_bundle(self, root: Path) -> Path:
        bundle = root / "bundle"
        bundle.mkdir()
        compiled_release = write_json(bundle / "compiled-release.json", {})
        inventory = {
            "schema_version": compiler.REQUEST_INVENTORY_SCHEMA,
            "study_id": compiler.STUDY_ID,
            "stage": "confirmation",
            "cohort_branch": "reduced_n3",
            "inventory_complete": True,
            "inventory_finalized_at": "2026-09-13T10:00:00Z",
            "annotation_state": "not_started",
            "episode_roster": [],
            "alignment_contracts": [],
            "requests": [],
        }
        inventory_path = write_json(bundle / "request_inventory.json", inventory)
        receipt = compiler.sign_document({
            "schema_version": compiler.COMPILER_SCHEMA,
            "study_id": compiler.STUDY_ID,
            "status": "compiled_complete_roster",
            "cohort_branch": "reduced_n3",
            "scientific_results_computed": False,
            "labels_created": False,
            "confirmation_released": False,
            "confirmation_release_freeze": descriptor(compiled_release),
            "ablation_spec_dependency": descriptor(compiler.SPEC_PATH),
            "outputs": {
                "request_inventory": descriptor(inventory_path),
                "endpoint_trace_receipts": [],
                "request_history_receipts": [],
            },
        })
        write_json(bundle / "compiler_receipt.json", receipt)
        return bundle

    def test_assembler_emits_exact_analyzer_schema_and_replays_final_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = self._minimal_bundle(root)
            selection = write_json(root / "selection.json", {
                "stage": "confirmation",
                "cohort_branch": "reduced_n3",
                "provenance": {
                    "request_inventory_sha256": compiler.sha256_file(bundle / "request_inventory.json")
                },
            })
            other_paths = {
                "ablation_spec": compiler.SPEC_PATH,
                "development_release_freeze": bundle / "compiled-release.json",
                "request_selection": selection,
                "annotation_freeze": write_json(root / "annotation.json", {}),
                "restricted_map": write_json(root / "restricted.json", {}),
                "final_consensus": write_json(root / "consensus.json", {}),
            }
            references = {
                key: {"path": str(path.resolve()), "sha256": compiler.sha256_file(path)}
                for key, path in other_paths.items()
            }
            output = root / "analysis_manifest.json"
            with mock.patch.object(compiler.freeze, "validate_release_freeze", return_value={
                "cohort_branch": "reduced_n3"
            }), mock.patch.object(compiler.annotation, "select_requests", return_value=json.loads(selection.read_text())), \
                 mock.patch.object(compiler.annotation, "_selected_requests", return_value={}), \
                 mock.patch.object(compiler, "_validate_compiled_bundle"), \
                 mock.patch.object(compiler.analyzer, "load_evidence", return_value={}) as final_gate:
                result = compiler.assemble_analysis_manifest(
                    compiler_bundle=bundle, output=output, **references
                )
            self.assertEqual(set(result), compiler.ANALYSIS_MANIFEST_KEYS)
            self.assertEqual(set(result["sources"]), compiler.ANALYSIS_SOURCE_KEYS)
            self.assertTrue(all(set(row) == compiler.REFERENCE_KEYS for row in result["sources"].values()))
            self.assertEqual(result["endpoint_trace_receipts"], [])
            self.assertEqual(result["request_history_receipts"], [])
            final_gate.assert_called_once()
            self.assertTrue(output.is_file())

    def test_assembler_rejects_a_different_valid_same_branch_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = self._minimal_bundle(root)
            selection = write_json(root / "selection.json", {
                "stage": "confirmation",
                "cohort_branch": "reduced_n3",
                "provenance": {
                    "request_inventory_sha256": compiler.sha256_file(
                        bundle / "request_inventory.json"
                    )
                },
            })
            sources = {
                "ablation_spec": compiler.SPEC_PATH,
                "development_release_freeze": write_json(
                    root / "different-valid-release.json", {"different": True}
                ),
                "request_selection": selection,
                "annotation_freeze": write_json(root / "annotation.json", {}),
                "restricted_map": write_json(root / "restricted.json", {}),
                "final_consensus": write_json(root / "consensus.json", {}),
            }
            references = {
                key: {"path": str(path.resolve()), "sha256": compiler.sha256_file(path)}
                for key, path in sources.items()
            }
            with mock.patch.object(
                compiler.freeze, "validate_release_freeze",
                return_value={"cohort_branch": "reduced_n3"},
            ), mock.patch.object(compiler, "_validate_compiled_bundle"):
                with self.assertRaisesRegex(
                    compiler.ConfirmationCompilerError, "exact freeze used to compile"
                ):
                    compiler.assemble_analysis_manifest(
                        compiler_bundle=bundle,
                        output=root / "manifest.json",
                        **references,
                    )

    def test_assembler_failure_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = self._minimal_bundle(root)
            selection = write_json(root / "selection.json", {
                "stage": "confirmation",
                "cohort_branch": "reduced_n3",
                "provenance": {
                    "request_inventory_sha256": compiler.sha256_file(bundle / "request_inventory.json")
                },
            })
            paths = {
                "ablation_spec": compiler.SPEC_PATH,
                "development_release_freeze": bundle / "compiled-release.json",
                "request_selection": selection,
                "annotation_freeze": write_json(root / "annotation.json", {}),
                "restricted_map": write_json(root / "restricted.json", {}),
                "final_consensus": write_json(root / "consensus.json", {}),
            }
            refs = {
                key: {"path": str(path.resolve()), "sha256": compiler.sha256_file(path)}
                for key, path in paths.items()
            }
            output = root / "analysis_manifest.json"
            with mock.patch.object(compiler.freeze, "validate_release_freeze", return_value={
                "cohort_branch": "reduced_n3"
            }), mock.patch.object(compiler.annotation, "select_requests", return_value=json.loads(selection.read_text())), \
                 mock.patch.object(compiler.annotation, "_selected_requests", return_value={}), \
                 mock.patch.object(compiler, "_validate_compiled_bundle"), \
                 mock.patch.object(compiler.analyzer, "load_evidence", side_effect=ValueError("bad labels")):
                with self.assertRaisesRegex(compiler.ConfirmationCompilerError, "final analyzer replay"):
                    compiler.assemble_analysis_manifest(
                        compiler_bundle=bundle, output=output, **refs
                    )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
