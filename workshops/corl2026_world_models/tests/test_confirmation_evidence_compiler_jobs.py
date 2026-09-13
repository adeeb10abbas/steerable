from __future__ import annotations

import ast
from contextlib import nullcontext
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
MODULE_PATH = (
    WORKSHOP / "experiments/forecast_layout/confirmation_evidence_compiler_jobs.py"
)
SPEC = importlib.util.spec_from_file_location(
    "confirmation_evidence_compiler_jobs_tests", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
jobs = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = jobs
SPEC.loader.exec_module(jobs)


def write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def descriptor(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": jobs.sha256_file(path),
        "bytes": path.stat().st_size,
    }


class ConfirmationEvidenceCompilerJobTests(unittest.TestCase):
    def test_source_has_no_duplicate_literal_keys(self) -> None:
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

    def test_contract_closes_science_and_queue_release(self) -> None:
        contract = json.loads(jobs.CONTRACT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            contract["schema_version"],
            "wmf-confirmation-evidence-compiler-contract-v1",
        )
        self.assertEqual(contract["prohibited_effects"], {
            "behavioral_actions_executed_by_compiler": 0,
            "behavioral_cells_launched_by_compiler": 0,
            "confirmation_jobs_released_by_compiler": 0,
            "labels_created_by_compiler": 0,
            "model_requests_issued_by_compiler": 0,
            "model_runtime_loads": 0,
            "model_servers_started": 0,
            "physical_resets": 0,
            "robot_episodes": 0,
            "simulator_processes_started": 0,
        })
        parser = jobs.build_parser()
        subparsers = [
            action for action in parser._actions
            if action.__class__.__name__ == "_SubParsersAction"
        ]
        self.assertEqual(set(subparsers[0].choices), {"seal-cohort", "build-manifest", "run"})
        self.assertNotIn("release", subparsers[0].choices)
        self.assertNotIn("queue", subparsers[0].choices)
        for command in ("seal-cohort", "build-manifest", "run"):
            command_options = {
                option
                for action in subparsers[0].choices[command]._actions
                for option in action.option_strings
            }
            self.assertNotIn("--worker-role", command_options)
            self.assertNotIn("--pod-uid", command_options)
            self.assertNotIn("--output-dir", command_options)
            self.assertNotIn("--job-receipt", command_options)

    def test_wrapper_source_check_rejects_untracked_residue(self) -> None:
        calls = []

        def fake_run(argv, **_kwargs):
            calls.append(argv)
            return SimpleNamespace(
                stdout=("a" * 40 + "\n" if argv[1:3] == ["rev-parse", "HEAD"] else "?? residue\n")
            )

        with mock.patch.object(jobs.subprocess, "run", side_effect=fake_run):
            with self.assertRaisesRegex(jobs.ConfirmationCompilerJobError, "source is dirty"):
                jobs._verify_clean_source(REPOSITORY, "a" * 40)
        self.assertIn("--untracked-files=all", calls[1])

    def test_sealer_deep_validates_native_terminal_before_accepting_safety(self) -> None:
        for model in ("N3", "D1"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as temporary:
                attempt = Path(temporary) / "attempt"
                cell_id = f"{model.lower()}__confirmation__cell"
                schedule = SimpleNamespace(cell_ids=[cell_id])
                cell_root = jobs.compiler._cell_root(attempt, 0, cell_id)
                terminal_path = write_json(
                    cell_root / "server_context_terminal.json",
                    {"schema_version": f"wmf-{model.lower()}-terminal-context-receipt-v1"},
                )
                terminal = descriptor(terminal_path)
                failure = {
                    "status": "safety_abort",
                    "actions_executed": 20,
                    "request_count": 1,
                    "server_context_terminal": terminal,
                }
                failure_path = write_json(cell_root / "technical_failure.json", failure)
                aggregate = {
                    "raw_attempt": attempt,
                    "counts": {
                        "completed_valid_behavioral_cells": 0,
                        "launched_behavioral_cells": 1,
                        "technically_invalid_behavioral_cells": 0,
                        "right_censored_behavioral_cells": 1,
                        "actual_behavioral_actions": 20,
                        "actual_behavioral_model_requests": 1,
                    },
                }
                validator = (
                    jobs.compiler.n3_confirmation.validate_failed_confirmation_cell
                    if model == "N3"
                    else jobs.compiler.d1_confirmation.validate_failed_confirmation_cell
                )
                patches = [mock.patch.object(
                    jobs.compiler.n3_confirmation if model == "N3"
                    else jobs.compiler.d1_confirmation,
                    "validate_failed_confirmation_cell",
                    return_value=failure,
                )]
                if model == "D1":
                    patches.append(mock.patch.object(
                        jobs.compiler.d1_confirmation,
                        "_configured_for_validation",
                        return_value=nullcontext(),
                    ))
                with patches[0] as validate:
                    if len(patches) == 2:
                        with patches[1] as configured:
                            rows = jobs._seal_failure_rows(
                                aggregate=aggregate, schedule=schedule,
                                model=model, layout="C01",
                                simulator_worker_role="d1-simulator-role",
                            )
                        configured.assert_called_once_with(
                            schedule, "d1-simulator-role"
                        )
                    else:
                        rows = jobs._seal_failure_rows(
                            aggregate=aggregate, schedule=schedule,
                            model=model, layout="C01",
                            simulator_worker_role=None,
                        )
                self.assertIsNotNone(validator)
                validate.assert_called_once_with(
                    failure_path, condition_index=0, block=schedule
                )
                self.assertEqual(rows[0]["context_terminal"], terminal)

                failure["server_context_terminal"] = None
                write_json(failure_path, failure)
                with self.assertRaisesRegex(
                    jobs.ConfirmationCompilerJobError,
                    "lacks native server-context terminal evidence",
                ):
                    jobs._seal_failure_rows(
                        aggregate=aggregate, schedule=schedule,
                        model=model, layout="C01",
                        simulator_worker_role="d1-simulator-role",
                    )

    def _authenticated_queue_context(
        self, root: Path, *, dirty: bool = False, hostname: str | None = None,
        pod_uid: str | None = "pod-123", mutate_argv: bool = False,
    ) -> object:
        state = root / "control"
        commit = "a" * 40
        job_id = "confirmation-compiler-001"
        source = state / "sources" / commit
        job_dir = state / "jobs" / job_id
        source.mkdir(parents=True)
        job_dir.mkdir(parents=True)
        manifest = write_json(state / "inputs" / "manifest.json", {"input": True})
        manifest_sha = jobs.sha256_file(manifest)
        argv = jobs._queue_argv(
            command="run",
            source_root=source,
            study_commit=commit,
            job_dir=job_dir,
            job_id=job_id,
            manifest_path=manifest,
            manifest_sha256=manifest_sha,
        )
        if mutate_argv:
            argv[-1] = "0" * 64
        released = {
            "schema_version": "wmf-cluster-job-v1",
            "namespace": jobs.compiler.NAMESPACE,
            "job_id": job_id,
            "released": True,
            "source_commit": commit,
            "role": "wmf-forecast-0912-worker-05",
            "argv": argv,
            "max_wall_seconds": jobs.QUEUE_MAX_WALL_SECONDS,
            "publish_log_tail_bytes": jobs.QUEUE_PUBLISH_LOG_TAIL_BYTES,
        }
        descriptor_path = write_json(job_dir / "descriptor.json", released)
        write_json(job_dir / "claim/owner.json", {
            "worker_id": "wmf-forecast-0912-worker-05",
            "claimed_at": "2026-09-13T10:00:00+00:00",
            "claimed_unix": 1789293600.0,
            "worker_pid": 42,
            "control_commit": "b" * 40,
            "control_generation": 1,
            "descriptor_sha256": jobs.sha256_file(descriptor_path),
            "release_boundary": "claim_committed_under_shared_release_lock",
        })

        def fake_git(_source, *argv):
            if argv == ("rev-parse", "HEAD"):
                return commit + "\n"
            self.assertEqual(argv, ("status", "--porcelain=v1", "--untracked-files=all"))
            return "?? residue\n" if dirty else ""

        fake_wrapper = source / jobs.THIS_RELATIVE
        environment = {} if pod_uid is None else {"POD_UID": pod_uid}
        with mock.patch.object(jobs.queue, "CONTROL_ROOT", state), \
             mock.patch.object(jobs.queue, "_run_git", side_effect=fake_git), \
             mock.patch.object(
                 jobs.queue.socket, "gethostname",
                 return_value=hostname or "wmf-forecast-0912-worker-05-pod",
             ), mock.patch.dict(jobs.os.environ, environment, clear=True), \
             mock.patch.object(jobs, "__file__", str(fake_wrapper)):
            return jobs._queue_context(
                command="run",
                source_root=source,
                study_commit=commit,
                job_dir=job_dir,
                job_id=job_id,
                manifest_path=manifest,
                manifest_sha256=manifest_sha,
            )

    def test_queue_context_derives_role_claim_pod_hostname_and_exact_argv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = self._authenticated_queue_context(Path(temporary))
        self.assertEqual(context.role, "wmf-forecast-0912-worker-05")
        self.assertEqual(context.worker_id, "wmf-forecast-0912-worker-05")
        self.assertEqual(context.pod_uid, "pod-123")

        cases = {
            "dirty": {"dirty": True},
            "hostname": {"hostname": "unowned-pod"},
            "pod": {"pod_uid": None},
            "argv": {"mutate_argv": True},
        }
        for label, kwargs in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(
                    jobs.ConfirmationCompilerJobError,
                    "(queue context failed|descriptor differs)",
                ):
                    self._authenticated_queue_context(Path(temporary), **kwargs)

    def _seal_producer_fixture(
        self, root: Path, *, mutation: str | None = None
    ) -> tuple[dict, Path, Path, Path, Path, dict, str]:
        state = root / "control"
        commit = "a" * 40
        job_id = "confirmation-cohort-seal-001"
        worker = "wmf-forecast-0912-worker-05"
        source = state / "sources" / commit
        job_dir = state / "jobs" / job_id
        source.mkdir(parents=True)
        job_dir.mkdir(parents=True)
        seal_plan = root / "seal-plan.json"
        raw_root = root / "raw-evidence"
        raw_root.mkdir()
        write_json(seal_plan, {"plan": True})
        argv = jobs._queue_argv(
            command="seal-cohort",
            source_root=source,
            study_commit=commit,
            job_dir=job_dir,
            job_id=job_id,
            seal_plan_path=seal_plan,
            seal_plan_sha256=jobs.sha256_file(seal_plan),
            raw_root=raw_root,
            camera_id="over_shoulder_left_camera",
        )
        released = {
            "schema_version": "wmf-cluster-job-v1",
            "namespace": jobs.compiler.NAMESPACE,
            "job_id": job_id,
            "released": True,
            "source_commit": commit,
            "role": worker,
            "argv": argv,
            "max_wall_seconds": jobs.QUEUE_MAX_WALL_SECONDS,
            "publish_log_tail_bytes": jobs.QUEUE_PUBLISH_LOG_TAIL_BYTES,
        }
        descriptor_path = write_json(job_dir / "descriptor.json", released)
        descriptor_identity = descriptor(descriptor_path)
        claim_path = write_json(job_dir / "claim/owner.json", {
            "worker_id": worker,
            "claimed_at": "2026-09-13T10:00:00Z",
            "claimed_unix": 1789293600.0,
            "worker_pid": 42,
            "control_commit": "b" * 40,
            "control_generation": 1,
            "descriptor_sha256": descriptor_identity["sha256"],
            "release_boundary": "claim_committed_under_shared_release_lock",
        })
        seal_bundle = job_dir / jobs.SEAL_BUNDLE_RELATIVE
        close_path = write_json(
            seal_bundle / "cohort_close_receipt.json",
            jobs.sign_document({
                "schema_version": jobs.compiler.COHORT_CLOSE_SCHEMA,
                "study_id": jobs.STUDY_ID,
            }),
        )
        close_identity = descriptor(close_path)
        index_path = write_json(
            seal_bundle / "terminal_block_index.json",
            jobs.sign_document({
                "schema_version": jobs.BLOCK_INDEX_SCHEMA,
                "cohort_close_receipt": close_identity,
            }),
        )
        manifest_path = write_json(
            job_dir / jobs.MANIFEST_RELATIVE,
            jobs.sign_document({
                "schema_version": jobs.INPUT_SCHEMA,
                "cohort_close_receipt": close_identity,
            }),
        )
        compiler_counts = {
            "planned_cells": 96,
            "valid_complete": 0,
            "valid_censored": 0,
            "technical_invalid": 0,
            "not_run": 96,
        }
        compiler_receipt_path = write_json(
            job_dir / jobs.RUN_BUNDLE_RELATIVE / "compiler_receipt.json",
            jobs.sign_document({
                "schema_version": jobs.compiler.COMPILER_SCHEMA,
                "cohort_close_receipt": close_identity,
                "input_manifest": descriptor(manifest_path),
                "counts": compiler_counts,
                "labels_created": False,
                "scientific_results_computed": False,
                "confirmation_released": False,
            }),
        )
        runtime_identity = {
            "hostname": worker + "-pod",
            "pod_uid": "pod-123",
            "pid": 777,
        }
        claim_identity = descriptor(claim_path)
        result_lineage = {
            "path": str(job_dir / "result.json"),
            "availability": "written_by_queue_worker_after_wrapper_exit",
            "job_id": job_id,
            "source_commit": commit,
            "descriptor_sha256": descriptor_identity["sha256"],
            "worker_id": worker,
        }
        science_counts = jobs._zero_science_counts()
        seal_receipt_path = write_json(
            job_dir / "publish" / jobs.SEAL_RECEIPT_NAME,
            jobs.sign_document({
                "schema_version": jobs.SEAL_JOB_RECEIPT_SCHEMA,
                "study_id": jobs.STUDY_ID,
                "status": "cohort_terminal_no_live_claims_or_descendants",
                "job_id": job_id,
                "job_dir": str(job_dir),
                "study_commit": commit,
                "queue_role": worker,
                "worker_id": worker,
                "runtime_identity": runtime_identity,
                "queue_descriptor": descriptor_identity,
                "queue_claim": claim_identity,
                "queue_result_lineage": result_lineage,
                "seal_plan": descriptor(seal_plan),
                "cohort_close_receipt": close_identity,
                "terminal_block_index": descriptor(index_path),
                "compiler_input_manifest": descriptor(manifest_path),
                "compiler_receipt": descriptor(compiler_receipt_path),
                "compiler_counts": compiler_counts,
                "science_counts": science_counts,
                "queue_mutations": 0,
                "confirmation_released": False,
                "labels_created": False,
                "scientific_results_computed": False,
                "claim_boundary": "zero-science fixture",
            }),
        )
        summary = {
            "status": "cohort_terminal_no_live_claims_or_descendants",
            "science_counts": science_counts,
            "confirmation_released": False,
            "output": str(seal_receipt_path),
            "sha256": jobs.sha256_file(seal_receipt_path),
        }
        (job_dir / "stdout.log").write_text(
            json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
        )
        (job_dir / "stderr.log").write_bytes(b"")
        executed = [
            item.replace("{source_root}", str(source))
            .replace("{job_dir}", str(job_dir))
            .replace("{state_dir}", str(state))
            for item in argv
        ]
        result = {
            "schema_version": "wmf-cluster-result-v1",
            "namespace": jobs.compiler.NAMESPACE,
            "job_id": job_id,
            "worker_id": worker,
            "source_commit": commit,
            "descriptor_sha256": descriptor_identity["sha256"],
            "started_at": "2026-09-13T10:00:00Z",
            "argv": executed,
            "job_dir": str(job_dir),
            "status": "succeeded",
            "returncode": 0,
            "error_type": None,
            "ended_at": "2026-09-13T10:02:00Z",
            "wall_seconds": 120.0,
            "child_pid": 777,
            "child_reaped": True,
            "stdout": {
                "bytes": (job_dir / "stdout.log").stat().st_size,
                "sha256": jobs.sha256_file(job_dir / "stdout.log"),
            },
            "stderr": {
                "bytes": 0,
                "sha256": jobs.sha256_file(job_dir / "stderr.log"),
            },
        }
        producer = {
            "job_id": job_id,
            "job_dir": str(job_dir),
            "role": worker,
            "worker_id": worker,
            "runtime_identity": runtime_identity,
            "descriptor": descriptor_identity,
            "claim_owner": claim_identity,
            "queue_result_lineage": result_lineage,
        }
        if mutation == "argv":
            result["argv"][1] = "/tmp/substituted-sealer.py"
        elif mutation == "child_pid":
            result["child_pid"] = 778
        elif mutation == "hostname":
            producer["runtime_identity"]["hostname"] = "other-pod"
        elif mutation == "lineage":
            producer["queue_result_lineage"]["descriptor_sha256"] = "0" * 64
        write_json(job_dir / "result.json", result)
        return (
            producer, state, source, raw_root, close_path,
            descriptor(seal_plan), commit,
        )

    def test_seal_producer_binds_outer_result_descriptor_claim_and_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            (
                producer, state, source, raw_root, close_path, seal_plan, commit,
            ) = self._seal_producer_fixture(Path(temporary))
            validated = jobs.compiler._validate_seal_producer(
                producer,
                close_path=close_path,
                queue_state_dir=state,
                source_root=source,
                raw_root=raw_root,
                compiler_camera_id="over_shoulder_left_camera",
                study_commit=commit,
                sealed_at_utc="2026-09-13T10:01:00Z",
                seal_plan_descriptor=seal_plan,
            )
            self.assertEqual(validated["descriptor"], producer["descriptor"])

        with tempfile.TemporaryDirectory() as temporary:
            (
                producer, state, source, raw_root, close_path, seal_plan, commit,
            ) = self._seal_producer_fixture(Path(temporary))
            (state / "jobs" / producer["job_id"] / "result.json").unlink()
            with mock.patch.object(jobs.compiler.os, "getpid", return_value=777), \
                 mock.patch.object(
                     jobs.compiler.socket, "gethostname",
                     return_value=producer["runtime_identity"]["hostname"],
                 ), mock.patch.dict(
                     jobs.compiler.os.environ,
                     {"POD_UID": producer["runtime_identity"]["pod_uid"]},
                     clear=True,
                 ):
                pending = jobs.compiler._validate_seal_producer(
                    producer,
                    close_path=close_path,
                    queue_state_dir=state,
                    source_root=source,
                    raw_root=raw_root,
                    compiler_camera_id="over_shoulder_left_camera",
                    study_commit=commit,
                    sealed_at_utc="2026-09-13T10:01:00Z",
                    seal_plan_descriptor=seal_plan,
                    active_producer=producer,
                )
            self.assertEqual(pending["_outer_result_state"], "pending_current_wrapper_exit")
            with self.assertRaisesRegex(
                jobs.compiler.ConfirmationCompilerError, "exact live context"
            ):
                jobs.compiler._validate_seal_producer(
                    producer,
                    close_path=close_path,
                    queue_state_dir=state,
                    source_root=source,
                    raw_root=raw_root,
                    compiler_camera_id="over_shoulder_left_camera",
                    study_commit=commit,
                    sealed_at_utc="2026-09-13T10:01:00Z",
                    seal_plan_descriptor=seal_plan,
                )

        for mutation in ("argv", "child_pid", "hostname", "lineage"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                (
                    producer, state, source, raw_root, close_path, seal_plan, commit,
                ) = self._seal_producer_fixture(Path(temporary), mutation=mutation)
                with self.assertRaisesRegex(
                    jobs.compiler.ConfirmationCompilerError,
                    "(execution changed|contradictory|runtime identity|lineage changed)",
                ):
                    jobs.compiler._validate_seal_producer(
                        producer,
                        close_path=close_path,
                        queue_state_dir=state,
                        source_root=source,
                        raw_root=raw_root,
                        compiler_camera_id="over_shoulder_left_camera",
                        study_commit=commit,
                        sealed_at_utc="2026-09-13T10:01:00Z",
                        seal_plan_descriptor=seal_plan,
                    )

    def test_post_exit_close_or_index_substitution_is_not_authenticated(self) -> None:
        for artifact in ("close", "index"):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as temporary:
                (
                    producer, state, source, raw_root, close_path, seal_plan, commit,
                ) = self._seal_producer_fixture(Path(temporary))
                job_dir = Path(producer["job_dir"])
                index_path = close_path.parent / "terminal_block_index.json"
                manifest_path = job_dir / jobs.MANIFEST_RELATIVE
                compiler_receipt_path = (
                    job_dir / jobs.RUN_BUNDLE_RELATIVE / "compiler_receipt.json"
                )
                seal_receipt_path = job_dir / "publish" / jobs.SEAL_RECEIPT_NAME
                if artifact == "close":
                    write_json(
                        close_path,
                        jobs.sign_document({
                            "schema_version": jobs.compiler.COHORT_CLOSE_SCHEMA,
                            "study_id": jobs.STUDY_ID,
                            "substituted": True,
                        }),
                    )
                    close_identity = descriptor(close_path)
                    write_json(
                        manifest_path,
                        jobs.sign_document({
                            "schema_version": jobs.INPUT_SCHEMA,
                            "cohort_close_receipt": close_identity,
                        }),
                    )
                    compiler_receipt = json.loads(compiler_receipt_path.read_text())
                    compiler_receipt.pop("payload_sha256")
                    compiler_receipt["cohort_close_receipt"] = close_identity
                    compiler_receipt["input_manifest"] = descriptor(manifest_path)
                    write_json(
                        compiler_receipt_path, jobs.sign_document(compiler_receipt)
                    )
                else:
                    close_identity = descriptor(close_path)
                write_json(
                    index_path,
                    jobs.sign_document({
                        "schema_version": jobs.BLOCK_INDEX_SCHEMA,
                        "cohort_close_receipt": close_identity,
                        "substituted": True,
                    }),
                )
                seal_receipt = json.loads(seal_receipt_path.read_text())
                seal_receipt.pop("payload_sha256")
                seal_receipt["cohort_close_receipt"] = close_identity
                seal_receipt["terminal_block_index"] = descriptor(index_path)
                seal_receipt["compiler_input_manifest"] = descriptor(manifest_path)
                seal_receipt["compiler_receipt"] = descriptor(compiler_receipt_path)
                write_json(seal_receipt_path, jobs.sign_document(seal_receipt))
                with self.assertRaisesRegex(
                    jobs.compiler.ConfirmationCompilerError,
                    "stdout does not bind its signed job receipt",
                ):
                    jobs.compiler._validate_seal_producer(
                        producer,
                        close_path=close_path,
                        queue_state_dir=state,
                        source_root=source,
                        raw_root=raw_root,
                        compiler_camera_id="over_shoulder_left_camera",
                        study_commit=commit,
                        sealed_at_utc="2026-09-13T10:01:00Z",
                        seal_plan_descriptor=seal_plan,
                    )

    def _block_index(self, root: Path, *, duplicate: bool = False) -> tuple[Path, str, dict]:
        rows = []
        release_path = write_json(root / "release.json", {"release": True})
        schedule_path = write_json(root / "schedule.json", {"schedule": True})
        close_path = write_json(root / "close.json", {"close": True})
        for index in range(1, 25):
            layout = "C01" if duplicate and index == 24 else f"C{index:02d}"
            receipt_path = write_json(root / "receipts" / f"receipt-{index:02d}.json", {
                "layout_pair_id": layout,
                "terminal": True,
            })
            rows.append({
                "model_id": "N3",
                "layout_pair_id": layout,
                "block_id": f"block-{layout}",
                "schedule_row_sha256": "1" * 64,
                "evidence_form": jobs.compiler.FULL_AGGREGATE_FORM,
                "receipt": descriptor(receipt_path),
                "failure_cells": [],
                "queue_jobs": [],
                "selected_queue_job_ids": [],
                "d1_pair": None,
            })
        value = jobs.sign_document({
            "schema_version": jobs.BLOCK_INDEX_SCHEMA,
            "study_id": jobs.STUDY_ID,
            "cohort_branch": "reduced_n3",
            "study_commit": "a" * 40,
            "development_release_freeze": descriptor(release_path),
            "prepared_schedule": descriptor(schedule_path),
            "cohort_close_receipt": descriptor(close_path),
            "block_receipts": rows,
        })
        path = write_json(root / "block-index.json", value)
        close = {
            "prepared_schedule": descriptor(schedule_path),
            "raw_root": str(root.resolve()),
            "blocks": rows,
        }
        return path, jobs.sha256_file(path), close

    def test_block_index_requires_exact_unique_24_block_roster(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, digest, close = self._block_index(Path(temporary), duplicate=True)
            with mock.patch.object(
                jobs.compiler, "_load_cohort_close",
                return_value=(descriptor(Path(temporary) / "close.json"),
                              (Path(temporary) / "close.json").resolve(), close),
            ):
                with self.assertRaisesRegex(jobs.ConfirmationCompilerJobError, "invalid or duplicated"):
                    jobs._validate_block_index(
                        path, digest, source_root=REPOSITORY, raw_root=Path(temporary)
                    )

    def test_manifest_builder_hashes_every_terminal_receipt_and_stays_nonreleasing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            block_index, block_sha, close = self._block_index(root)
            release_path = write_json(root / "release.json", {"release": True})
            output = root / "compiler-input.json"
            release = {
                "cohort_branch": "reduced_n3",
                "qualified_model_ids": ["N3"],
            }
            schedule_identity = descriptor(
                REPOSITORY / "workshops/corl2026_world_models/execution/20260912/parallel_schedule.json"
            )
            close_path = root / "close-build.json"
            close_value = {"sealed_at_utc": "2026-09-13T10:00:00Z"}
            write_json(close_path, close_value)
            index_value = {
                "cohort_branch": "reduced_n3",
                "study_commit": "a" * 40,
                "development_release_freeze": descriptor(release_path),
                "prepared_schedule": schedule_identity,
                "cohort_close_receipt": descriptor(close_path),
                "block_receipts": close["blocks"],
            }
            with mock.patch.object(
                jobs.compiler.freeze, "validate_release_freeze", return_value=release
            ), mock.patch.object(
                jobs, "_validate_block_index",
                return_value=(descriptor(block_index), index_value),
            ), mock.patch.object(
                jobs, "_verify_clean_source",
            ):
                result = jobs.build_compiler_manifest(
                    block_index_path=block_index,
                    block_index_sha256=block_sha,
                    source_root=REPOSITORY,
                    raw_root=raw,
                    study_commit="a" * 40,
                    inventory_finalized_at="2026-09-13T10:00:00Z",
                    camera_id="over_shoulder_left_camera",
                    release_freeze_path=release_path,
                    release_freeze_sha256=jobs.sha256_file(release_path),
                    output=output,
                )
            jobs.verify_signed(result, "compiler input")
            self.assertEqual(result["schema_version"], jobs.INPUT_SCHEMA)
            self.assertEqual(len(result["block_receipts"]), 24)
            self.assertEqual(result["cohort_close_receipt"], descriptor(close_path))
            self.assertTrue(output.is_file())

    def test_run_receipt_repeats_zero_science_and_no_queue_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = jobs.sign_document({
                "schema_version": jobs.INPUT_SCHEMA,
                "study_id": jobs.STUDY_ID,
                "source_root": str(REPOSITORY.resolve()),
                "study_commit": "a" * 40,
            })
            manifest_path = write_json(root / "manifest.json", manifest)
            job_dir = root / "jobs" / "compiler-job"
            job_dir.mkdir(parents=True)
            output = job_dir / jobs.RUN_BUNDLE_RELATIVE

            def fake_compile(_path, _sha, output_dir):
                output_dir.mkdir()
                zero = jobs.compiler.sign_document({
                    "schema_version": jobs.compiler.ZERO_SCIENCE_SCHEMA,
                    "study_id": jobs.STUDY_ID,
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
                })
                write_json(output_dir / "zero_science_receipt.json", zero)
                write_json(output_dir / "compiler_receipt.json", {"receipt": True})
                return {"counts": {"planned_cells": 96}}

            context = SimpleNamespace(
                source_root=REPOSITORY.resolve(),
                study_commit="a" * 40,
                job_dir=job_dir.resolve(),
                job_id="compiler-job",
                role="wmf-forecast-0912-worker-05",
                worker_id="wmf-forecast-0912-worker-05",
                hostname="wmf-forecast-0912-worker-05-pod",
                pod_uid="pod-123",
                descriptor_identity={"path": str(job_dir / "descriptor.json"), "sha256": "1" * 64, "bytes": 1},
                claim_identity={"path": str(job_dir / "claim/owner.json"), "sha256": "2" * 64, "bytes": 1},
            )
            with mock.patch.object(jobs, "_queue_context", return_value=context), \
                 mock.patch.object(jobs.compiler, "compile_manifest", side_effect=fake_compile):
                result = jobs.run_compiler_job(
                    source_root=REPOSITORY,
                    study_commit="a" * 40,
                    job_dir=job_dir,
                    job_id="compiler-job",
                    manifest_path=manifest_path,
                    manifest_sha256=jobs.sha256_file(manifest_path),
                )
            jobs.verify_signed(result, "job receipt")
            self.assertEqual(result["science_counts"], jobs._zero_science_counts())
            self.assertEqual(result["queue_mutations"], 0)
            self.assertIs(result["confirmation_released"], False)
            self.assertIs(result["labels_created"], False)
            self.assertIs(result["scientific_results_computed"], False)
            self.assertEqual(result["queue_descriptor"], context.descriptor_identity)
            self.assertEqual(result["queue_claim"], context.claim_identity)
            self.assertEqual(result["runtime_identity"]["pod_uid"], "pod-123")
            self.assertTrue((job_dir / "publish" / jobs.RUN_RECEIPT_NAME).is_file())

    def _n3_queue_attempt(
        self, state: Path, *, job_id: str, selected: bool,
        runtime_status: str = "technical_failure", remaining_compute: bool = False,
        write_runtime: bool = True, layout: str = "C01",
        server_reaped: bool = True,
    ) -> None:
        job_dir = state / "jobs" / job_id
        argv = [
            "/usr/bin/python3",
            "{source_root}/workshops/corl2026_world_models/experiments/forecast_layout/"
            "n3_confirmation_block_job.py",
            "queue",
            "--layout-pair-id", layout,
            "--study-commit", "a" * 40,
            "--job-id", job_id,
        ]
        queue = {
            "schema_version": "wmf-cluster-job-v1",
            "namespace": jobs.compiler.NAMESPACE,
            "job_id": job_id,
            "released": True,
            "source_commit": "a" * 40,
            "role": "n3",
            "argv": argv,
            "max_wall_seconds": 100,
            "publish_log_tail_bytes": 0,
        }
        queue_path = write_json(job_dir / "descriptor.json", queue)
        queue_identity = descriptor(queue_path)
        write_json(job_dir / "claim/owner.json", {
            "worker_id": "worker-1",
            "claimed_at": "2026-09-13T10:00:00+00:00",
            "claimed_unix": 1789293600.0,
            "worker_pid": 100,
            "control_commit": "b" * 40,
            "control_generation": 1,
            "descriptor_sha256": queue_identity["sha256"],
            "release_boundary": "claim_committed_under_shared_release_lock",
        })
        terminal_status = "succeeded" if runtime_status == "passed" else "failed"
        returncode = 0 if runtime_status == "passed" else 1
        executed_argv = [
            value.replace("{source_root}", str(state / "sources" / ("a" * 40)))
            .replace("{job_dir}", str(job_dir))
            .replace("{state_dir}", str(state))
            for value in argv
        ]
        (job_dir / "stdout.log").write_bytes(b"")
        (job_dir / "stderr.log").write_bytes(b"")
        write_json(job_dir / "result.json", {
            "schema_version": "wmf-cluster-result-v1",
            "namespace": jobs.compiler.NAMESPACE,
            "job_id": job_id,
            "worker_id": "worker-1",
            "source_commit": "a" * 40,
            "descriptor_sha256": queue_identity["sha256"],
            "started_at": "2026-09-13T10:00:00+00:00",
            "argv": executed_argv,
            "job_dir": str(job_dir),
            "status": terminal_status,
            "returncode": returncode,
            "error_type": None,
            "ended_at": "2026-09-13T10:01:00+00:00",
            "wall_seconds": 60.0,
            "child_pid": None,
            "child_reaped": True,
            "stdout": {"bytes": 0, "sha256": jobs.compiler.sha256_bytes(b"")},
            "stderr": {"bytes": 0, "sha256": jobs.compiler.sha256_bytes(b"")},
        })
        if write_runtime:
            attempt = state / "raw" / "raw-attempts" / job_id
            attempt.mkdir(parents=True)
            write_json(attempt / "cleanup.json", {
                "schema_version": "wmf-n3-behavioral-confirmation-cleanup-v1",
                "all_children_reaped": True,
                "remaining_compute_processes": ([{"pid": 9}] if remaining_compute else []),
                "completed_at_utc": "2026-09-13T10:01:00Z",
            })
            server_exit = {
                "schema_version": "wmf-n3-server-supervisor-exit-v1",
                "child_reaped": server_reaped,
            }
            write_json(attempt / "server/supervisor_exit.json", server_exit)
            write_json(job_dir / "publish" / jobs.compiler.n3_confirmation.AGGREGATE_FILENAME, {
                "schema_version": jobs.compiler.n3_confirmation.QUEUE_RECEIPT_SCHEMA,
                "status": runtime_status,
                "exit_code": returncode,
                "source_commit": "a" * 40,
                "layout_pair_id": layout,
                "block_id": f"block-{layout}",
                "queue_descriptor": queue_identity,
                "server_exit": server_exit,
                "raw_attempt_root": str(attempt),
                "counts": {
                    "planned_behavioral_cells": 4,
                    "launched_behavioral_cells": 0,
                    "resumed_valid_behavioral_cells": 0,
                    "newly_launched_behavioral_cells": 0,
                    "completed_valid_behavioral_cells": 0,
                    "technically_invalid_behavioral_cells": 0,
                    "right_censored_behavioral_cells": 0,
                    "unrun_behavioral_cells": 4,
                    "actual_behavioral_actions": 0,
                    "actual_behavioral_model_requests": 0,
                    "new_generation_qualification_requests": 0,
                    "reused_prerequisite_generation_qualification_requests": 6,
                    "recorder_only_episodes_counted_as_behavioral": 0,
                },
            })

    def _scan_one_layout(self, state: Path, selected_id: str) -> dict:
        return jobs._scan_confirmation_queue_jobs(
            state_dir=state,
            study_commit="a" * 40,
            branch="reduced_n3",
            schedules={("N3", "C01"): SimpleNamespace(block_id="block-C01")},
            selected={("N3", "C01"): [selected_id]},
            raw_root=state / "raw",
        )

    def test_unselected_retry_without_deep_runtime_receipt_blocks_seal_scan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            self._n3_queue_attempt(state, job_id="selected", selected=True)
            self._n3_queue_attempt(
                state, job_id="retry-old", selected=False, write_runtime=False
            )
            with self.assertRaisesRegex(jobs.ConfirmationCompilerJobError, "runtime receipt is missing"):
                self._scan_one_layout(state, "selected")

    def test_n3_cleanup_with_remaining_compute_blocks_seal_scan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            self._n3_queue_attempt(
                state, job_id="selected", selected=True, remaining_compute=True
            )
            with self.assertRaisesRegex(
                jobs.compiler.ConfirmationCompilerError,
                "nested scientific children were not closed",
            ):
                self._scan_one_layout(state, "selected")

    def test_queue_success_contradicting_technical_runtime_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            self._n3_queue_attempt(state, job_id="selected", selected=True)
            result_path = state / "jobs/selected/result.json"
            result = json.loads(result_path.read_text())
            result["status"] = "succeeded"
            result["returncode"] = 0
            write_json(result_path, result)
            with self.assertRaisesRegex(
                jobs.compiler.ConfirmationCompilerError,
                "contradicts its runtime terminal receipt",
            ):
                self._scan_one_layout(state, "selected")

    def test_executed_argv_job_dir_and_log_mutations_fail_closed(self) -> None:
        for field in ("argv", "job_dir", "stdout"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                state = Path(temporary)
                self._n3_queue_attempt(state, job_id="selected", selected=True)
                result_path = state / "jobs/selected/result.json"
                result = json.loads(result_path.read_text())
                if field == "argv":
                    result["argv"][1] = "/tmp/substituted-wrong-runner.py"
                    expected = "executed argv/job directory"
                elif field == "job_dir":
                    result["job_dir"] = "/tmp/wrong-job-dir"
                    expected = "executed argv/job directory"
                else:
                    (state / "jobs/selected/stdout.log").write_bytes(b"mutated")
                    expected = "stdout log changed"
                write_json(result_path, result)
                with self.assertRaisesRegex(
                    jobs.compiler.ConfirmationCompilerError, expected
                ):
                    self._scan_one_layout(state, "selected")

    def test_scientific_queue_row_cannot_cross_the_sealed_state_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state-a"
            other = Path(temporary) / "state-b"
            other.mkdir()
            self._n3_queue_attempt(state, job_id="selected", selected=True)
            grouped = self._scan_one_layout(state, "selected")
            row = grouped[("N3", "C01")][0]
            with self.assertRaisesRegex(
                jobs.compiler.ConfirmationCompilerError, "outside the sealed queue state"
            ):
                jobs.compiler._validate_queue_job(
                    row,
                    close_path=state / "close.json",
                    model="N3",
                    layout="C01",
                    block_id="block-C01",
                    study_commit="a" * 40,
                    expected_state_dir=other,
                )

    def test_n3_unreaped_nested_server_child_blocks_seal_scan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            self._n3_queue_attempt(
                state, job_id="selected", selected=True, server_reaped=False
            )
            with self.assertRaisesRegex(
                jobs.compiler.ConfirmationCompilerError,
                "server child lacks a reaped exit receipt",
            ):
                self._scan_one_layout(state, "selected")

    def test_missing_selected_attempt_and_extra_layout_both_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            self._n3_queue_attempt(state, job_id="only-attempt", selected=False)
            with self.assertRaisesRegex(
                jobs.ConfirmationCompilerJobError, "selected confirmation queue jobs are missing"
            ):
                self._scan_one_layout(state, "omitted-attempt")
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            self._n3_queue_attempt(state, job_id="selected", selected=True)
            self._n3_queue_attempt(
                state, job_id="extra-layout", selected=False, layout="C02"
            )
            with self.assertRaisesRegex(
                jobs.ConfirmationCompilerJobError, "out-of-cohort confirmation descriptor"
            ):
                self._scan_one_layout(state, "selected")

    def test_global_d1_lock_must_be_acquirable_for_seal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / "d1.lock"
            with jobs._exclusive_lock(lock, "global D1 server lock"):
                with self.assertRaisesRegex(
                    jobs.ConfirmationCompilerJobError, "global D1 server lock is held"
                ):
                    with jobs._exclusive_lock(lock, "global D1 server lock"):
                        pass

    def test_cohort_seal_requires_shutdown_or_expired_admission_deadline(self) -> None:
        control = {
            "control_commit": "a" * 40,
            "control_generation": 7,
            "active_job_ids": ["confirmation-job"],
            "admission_deadline_unix": 200.0,
            "shutdown": False,
        }
        with self.assertRaisesRegex(
            jobs.ConfirmationCompilerJobError, "before shutdown or its absolute admission deadline"
        ):
            jobs._confirmation_admission_barrier(
                control, sealed_at="1970-01-01T00:01:40Z"
            )
        control["admission_deadline_unix"] = 50.0
        expired = jobs._confirmation_admission_barrier(
            control, sealed_at="1970-01-01T00:01:40Z"
        )
        self.assertEqual(expired["kind"], "expired_absolute_admission_deadline")
        self.assertEqual(expired["active_job_ids"], ["confirmation-job"])
        control["shutdown"] = True
        control["admission_deadline_unix"] = None
        stopped = jobs._confirmation_admission_barrier(
            control, sealed_at="1970-01-01T00:01:40Z"
        )
        self.assertEqual(stopped["kind"], "global_shutdown")

    def _sealer_wait_context(
        self, root: Path, *, active_ids: list[str], deadline: float
    ) -> SimpleNamespace:
        job_id = "confirmation-cohort-seal-001"
        job_dir = root / "jobs" / job_id
        job_dir.mkdir(parents=True)
        descriptor_path = write_json(job_dir / "descriptor.json", {"job_id": job_id})
        claim_path = write_json(job_dir / "claim/owner.json", {
            "worker_id": "wmf-forecast-0912-worker-05",
            "claimed_at": "2026-09-13T10:00:00Z",
            "claimed_unix": deadline - 1.0,
            "worker_pid": 42,
            "control_commit": "a" * 40,
            "control_generation": 7,
            "descriptor_sha256": "1" * 64,
            "release_boundary": "claim_committed_under_shared_release_lock",
        })
        write_json(root / "control.json", {
            "namespace": jobs.compiler.NAMESPACE,
            "control_commit": "a" * 40,
            "control_generation": 7,
            "admission_deadline_unix": deadline,
            "shutdown": False,
            "active_job_ids": active_ids,
        })
        return SimpleNamespace(
            job_id=job_id,
            job_dir=job_dir.resolve(),
            descriptor_identity=descriptor(descriptor_path),
            claim_identity=descriptor(claim_path),
        )

    def test_claimed_sealer_wait_gate_requires_sole_job_and_bounded_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            deadline = jobs.time.time() - 0.1
            context = self._sealer_wait_context(
                root, active_ids=["confirmation-cohort-seal-001"], deadline=deadline
            )
            identity = jobs._wait_for_seal_admission_cutoff(context)
            self.assertEqual(identity, descriptor(root / "control.json"))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            deadline = jobs.time.time() - 0.1
            context = self._sealer_wait_context(
                root,
                active_ids=["confirmation-cohort-seal-001", "another-job"],
                deadline=deadline,
            )
            with self.assertRaisesRegex(
                jobs.ConfirmationCompilerJobError, "sole active job"
            ):
                jobs._wait_for_seal_admission_cutoff(context)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            deadline = jobs.time.time() + jobs.MAX_SEAL_ADMISSION_WAIT_SECONDS + 1
            context = self._sealer_wait_context(
                root, active_ids=["confirmation-cohort-seal-001"], deadline=deadline
            )
            with self.assertRaisesRegex(
                jobs.ConfirmationCompilerJobError, "bounded near-term deadline"
            ):
                jobs._wait_for_seal_admission_cutoff(context)

    def test_orphan_d1_claim_is_found_outside_selected_run_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            claim = raw / "behavioral/confirmation/D1/C17/coordination/orphan/simulator_claim.json"
            write_json(claim, {
                "schema_version": jobs.compiler.d1_confirmation.pilot.SIMULATOR_CLAIM_SCHEMA,
                "status": "claimed",
                "run_id": "orphan",
            })
            unexpired, unterminated = jobs._audit_d1_protocol(raw_root=raw)
            self.assertEqual(unexpired, [])
            self.assertEqual(unterminated, [str(claim.resolve())])

    def _seal_plan(self, root: Path, *, schedule_path: Path) -> tuple[Path, str, Path]:
        release_path = write_json(root / "release.json", {"release": True})
        value = jobs.sign_document({
            "schema_version": jobs.SEAL_PLAN_SCHEMA,
            "study_id": jobs.STUDY_ID,
            "cohort_branch": "reduced_n3",
            "study_commit": "a" * 40,
            "development_release_freeze": descriptor(release_path),
            "prepared_schedule": descriptor(schedule_path),
            "selected_blocks": [
                {
                    "model_id": "N3",
                    "layout_pair_id": f"C{index:02d}",
                    "selected_queue_job_ids": [f"n3-c{index:02d}"],
                }
                for index in range(1, 25)
            ],
        })
        path = write_json(root / "seal-plan.json", value)
        return path, jobs.sha256_file(path), release_path

    def test_seal_plan_rejects_cross_schedule_and_post_sign_release_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrong_schedule = write_json(root / "not-the-staged-schedule.json", {"x": 1})
            path, digest, _release = self._seal_plan(root, schedule_path=wrong_schedule)
            with self.assertRaisesRegex(
                jobs.ConfirmationCompilerJobError, "not the staged authoritative schedule"
            ):
                jobs._load_seal_plan(path, digest, source_root=REPOSITORY)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule = (
                REPOSITORY
                / "workshops/corl2026_world_models/execution/20260912/parallel_schedule.json"
            )
            path, digest, release = self._seal_plan(root, schedule_path=schedule)
            release.write_text('{"mutated":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(Exception, "(byte count|file hash) changed"):
                jobs._load_seal_plan(path, digest, source_root=REPOSITORY)


if __name__ == "__main__":
    unittest.main()
