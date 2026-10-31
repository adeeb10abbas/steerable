from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


WORKSHOP_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    WORKSHOP_ROOT
    / "experiments/forecast_layout/camera_crop_witness_terminal_jobs.py"
)
SPEC = importlib.util.spec_from_file_location("camera_crop_witness_terminal_jobs", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
jobs = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = jobs
SPEC.loader.exec_module(jobs)

ALIGNMENT_PATH = (
    WORKSHOP_ROOT
    / "experiments/forecast_layout/development_physical_alignment_jobs.py"
)
ALIGNMENT_SPEC = importlib.util.spec_from_file_location(
    "terminal_camera_alignment_consumer", ALIGNMENT_PATH
)
assert ALIGNMENT_SPEC is not None and ALIGNMENT_SPEC.loader is not None
alignment = importlib.util.module_from_spec(ALIGNMENT_SPEC)
sys.modules[ALIGNMENT_SPEC.name] = alignment
ALIGNMENT_SPEC.loader.exec_module(alignment)


class CameraCropWitnessTerminalJobTests(unittest.TestCase):
    class _FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid
            self.returncode = None
            self.terminated = False
            self.wait_count = 0

        def poll(self):
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, timeout=None):
            self.wait_count += 1
            if self.returncode is None:
                raise subprocess.TimeoutExpired("fake", timeout)
            return self.returncode

    def _implementation(self) -> dict[str, dict]:
        return {
            name: {
                "path": str(path),
                "bytes": index + 100,
                "sha256": f"{index + 1:064x}",
            }
            for index, (name, path) in enumerate(
                jobs._source_paths(jobs.REPOSITORY_ROOT).items()
            )
        }

    @staticmethod
    def _descriptor(path: str, ordinal: int) -> dict:
        return {"path": path, "sha256": f"{ordinal:064x}", "bytes": ordinal + 10}

    def _prior_receipt(self) -> dict:
        diagnostics = {}
        raw_root = (
            jobs.CONTROL_ROOT
            / "jobs"
            / jobs.PRIOR_JOB_ID
            / "raw/camera_crop_witness"
        )
        for ordinal, model in enumerate(("N3", "D1"), start=1):
            diagnostics[model] = {
                "model_id": model,
                "exit_code": 0,
                "full_log": self._descriptor(
                    str(raw_root / f"{model.lower()}_replay.log"), ordinal
                ),
                "bounded_tail": {
                    "maximum_bytes": jobs.CHILD_LOG_TAIL_BYTES,
                    "source_offset_bytes": 0,
                    "raw_bytes": ordinal + 10,
                    "raw_sha256": f"{ordinal + 20:064x}",
                    "partial_leading_line_dropped_bytes": 0,
                    "safe_structured_summary": {
                        "preflight_events": [],
                        "traceback_frames": [],
                        "terminal_errors": [],
                        "line_count": 0,
                        "recognized_line_count": 0,
                        "arbitrary_text_published": False,
                    },
                    "truncated": False,
                },
                "argv_published": False,
                "environment_values_published": False,
                "launched": True,
                "launch_receipt": self._descriptor(
                    str(
                        raw_root
                        / "child_launch_receipts"
                        / f"{model.lower()}_launch.json"
                    ),
                    ordinal + 2,
                ),
                "runtime_preflight": {
                    "python": {
                        "lexical_path": {
                            "N3": (
                                "/data/users/ali/vla_wam/envs/"
                                "robolab-v2-isaac50/bin/python"
                            ),
                            "D1": (
                                "/data/users/ali/vla_wam/envs/"
                                "dreamzero-ab790c1-py311/bin/python"
                            ),
                        }[model],
                        "resolved_file": self._descriptor(
                            {
                                "N3": (
                                    "/data/users/ali/vla_wam/envs/"
                                    "robolab-v2-isaac50/bin/python"
                                ),
                                "D1": (
                                    "/data/users/ali/vla_wam/envs/"
                                    "dreamzero-ab790c1-py311/bin/python"
                                ),
                            }[model],
                            ordinal + 30,
                        ),
                    },
                    "path_environment": {
                        "PYTHONPATH": [],
                        "PYTHONHOME": [],
                        "VIRTUAL_ENV": [],
                    },
                    "cpu_offline_overrides": {
                        "CUDA_VISIBLE_DEVICES": "",
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTHONUNBUFFERED": "1",
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                    },
                    "inherited_queue_environment": True,
                    "argv_published": False,
                    "arbitrary_environment_values_published": False,
                },
            }
        return jobs.queue.signed_document({
            "schema_version": jobs.FAILURE_SCHEMA,
            "namespace": jobs.NAMESPACE,
            "study_id": jobs.STUDY_ID,
            "status": "technical_invalid",
            "decision": "no_go",
            "job_id": jobs.PRIOR_JOB_ID,
            "study_commit": "a" * 40,
            "queue_role": jobs.WORKER_ROLE,
            "failure": {
                "error_type": "CameraCropChildError",
                "message_bytes": 120,
                "message_sha256": (
                    "7730b172cf378e1abfc2d769f743c2be98457c10e6505562fe60e9e6af1ed1d9"
                ),
                "traceback_frames": [],
                "traceback_frame_count": 0,
                "exception_message_published": False,
                "traceback_source_text_published": False,
                "child_exits": {"N3": 0, "D1": 0},
                "child_diagnostics": diagnostics,
                "child_orchestration_failure": None,
            },
            "diagnostic_only": True,
            "prior_attempt": jobs.PRIOR_PRIOR_JOB_ID,
            "science_counts": jobs.zero_science_counts(),
            "simulator_state_render_used": False,
            "whole_frame_identity": False,
            "safe_to_release_confirmation": False,
            "confirmation_released": False,
            "claim_boundary": (
                "Diagnostic-only technical-invalid CPU replay preserving attempt 001. No model, "
                "simulator, request, action, behavior, label, or confirmation job was started; "
                "no crop contract is published. Bounded signed log evidence is diagnostic only."
            ),
            "completed_at_utc": "2026-10-31T00:00:00Z",
        })

    @staticmethod
    def _resign(value: dict) -> dict:
        unsigned = dict(value)
        unsigned.pop("payload_sha256", None)
        return jobs.queue.signed_document(unsigned)

    def _raw_contracts(self, root: Path) -> dict[str, dict]:
        result = {}
        for model in ("N3", "D1"):
            model_root = root / model
            model_root.mkdir()
            original = model_root / "original_camera_crop.npy"
            generated = model_root / "generated_decoded_crop.npy"
            original.write_bytes((model + "-original").encode())
            generated.write_bytes((model + "-generated").encode())
            value = jobs.queue.signed_document({
                "schema_version": jobs.replay.SCHEMA,
                "model_id": model,
                "camera_crop_id": jobs.EXPECTED_CAMERA_CROP_IDS[model],
                "original_camera_replay": {
                    "witness_artifact": jobs.replay.file_identity(
                        original, label=f"{model} original"
                    )
                },
                "generated_decoded_crop": {
                    "witness_artifact": jobs.replay.file_identity(
                        generated, label=f"{model} generated"
                    )
                },
            })
            path = root / jobs.MODEL_CONTRACT_NAMES[model]
            jobs.queue.immutable_json(path, value)
            result[model] = {
                "value": value,
                "identity": jobs.replay.file_identity(path, label=f"raw {model}"),
            }
        return result

    def test_terminal_contract_is_attempt003_six_gate_zero_science(self) -> None:
        contract = jobs._runtime_contract(jobs.REPOSITORY_ROOT)
        self.assertEqual(jobs.JOB_ID, "camera-crop-replay-witness-003")
        self.assertEqual(contract["job"]["job_id"], jobs.JOB_ID)
        self.assertEqual(contract["prior_diagnostic"]["job_id"], jobs.PRIOR_JOB_ID)
        self.assertEqual(
            contract["prior_diagnostic"]["required_child_exits"], {"N3": 0, "D1": 0}
        )
        self.assertEqual(contract["replay_runtime"]["explicit_prerequisite_receipt_count"], 5)
        self.assertEqual(contract["replay_runtime"]["total_external_receipt_gate_count"], 6)
        self.assertIs(contract["child_runtime"]["diagnostic_only"], False)
        self.assertIs(contract["child_runtime"]["publish_crop_contracts"], True)
        self.assertIs(contract["child_runtime"]["atomic_directory_publication"], True)
        self.assertEqual(contract["expected_outputs"]["publish_file_count"], 3)
        self.assertEqual(contract["science_counts"], jobs.zero_science_counts())
        self.assertIs(contract["success_receipt_safe_for_physical_alignment_input"], True)
        self.assertIs(contract["safe_to_release_confirmation"], False)

    def test_terminal_contract_rejects_json_scalar_type_confusion(self) -> None:
        path = jobs.REPOSITORY_ROOT / jobs.TERMINAL_CONTRACT_RELATIVE
        contract = jobs.replay.load_json(path, "terminal contract test fixture")
        mutations = (
            ("prior child exit bool", lambda value: value["prior_diagnostic"][
                "required_child_exits"
            ].__setitem__("N3", False)),
            ("science signed zero", lambda value: value["science_counts"].__setitem__(
                "model_runtime_loads", -0.0
            )),
            ("CPU bool as int", lambda value: value["job"].__setitem__("cpu_only", 1)),
            ("wall integer as float", lambda value: value["job"].__setitem__(
                "max_wall_seconds", 10800.0
            )),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                changed = copy.deepcopy(contract)
                mutate(changed)
                with mock.patch.object(jobs.replay, "load_json", return_value=changed):
                    with self.assertRaisesRegex(
                        jobs.CameraCropTerminalError, "changed"
                    ):
                        jobs._runtime_contract(jobs.REPOSITORY_ROOT)

    def test_source_contains_no_duplicate_literal_dictionary_keys(self) -> None:
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        duplicates = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = [
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            ]
            duplicates.extend((node.lineno, key) for key in set(keys) if keys.count(key) > 1)
        self.assertEqual(duplicates, [])

    def test_prior_receipt_requires_exact_intentional_two_child_diagnostic(self) -> None:
        receipt = self._prior_receipt()
        self.assertEqual(jobs._validate_prior_diagnostic_value(receipt), "a" * 40)

        changed = dict(receipt)
        changed["failure"] = dict(changed["failure"])
        changed["failure"]["child_exits"] = {"N3": 1, "D1": 0}
        changed = self._resign(changed)
        with self.assertRaisesRegex(jobs.CameraCropTerminalError, "completed two-child"):
            jobs._validate_prior_diagnostic_value(changed)

        changed = dict(receipt)
        changed["failure"] = dict(changed["failure"])
        changed["failure"]["message_sha256"] = "f" * 64
        changed = self._resign(changed)
        with self.assertRaisesRegex(jobs.CameraCropTerminalError, "completed two-child"):
            jobs._validate_prior_diagnostic_value(changed)

        changed = dict(receipt)
        changed["failure"] = dict(changed["failure"])
        changed["failure"]["child_diagnostics"] = dict(
            changed["failure"]["child_diagnostics"]
        )
        changed["failure"]["child_diagnostics"]["D1"] = dict(
            changed["failure"]["child_diagnostics"]["D1"]
        )
        changed["failure"]["child_diagnostics"]["D1"]["launch_receipt"] = None
        changed = self._resign(changed)
        with self.assertRaisesRegex(jobs.CameraCropTerminalError, "launch receipt"):
            jobs._validate_prior_diagnostic_value(changed)

        changed = dict(receipt)
        changed["failure"] = dict(changed["failure"])
        changed["failure"]["child_exits"] = {"N3": False, "D1": 0}
        changed = self._resign(changed)
        with self.assertRaisesRegex(jobs.CameraCropTerminalError, "completed two-child"):
            jobs._validate_prior_diagnostic_value(changed)

        changed = dict(receipt)
        changed["failure"] = dict(changed["failure"])
        changed["failure"]["child_diagnostics"] = dict(
            changed["failure"]["child_diagnostics"]
        )
        changed["failure"]["child_diagnostics"]["N3"] = dict(
            changed["failure"]["child_diagnostics"]["N3"]
        )
        changed["failure"]["child_diagnostics"]["N3"]["exit_code"] = False
        changed = self._resign(changed)
        with self.assertRaisesRegex(jobs.CameraCropTerminalError, "did not exit zero"):
            jobs._validate_prior_diagnostic_value(changed)

        changed = dict(receipt)
        changed["failure"] = dict(changed["failure"])
        changed["failure"]["child_diagnostics"] = dict(
            changed["failure"]["child_diagnostics"]
        )
        changed["failure"]["child_diagnostics"]["N3"] = dict(
            changed["failure"]["child_diagnostics"]["N3"]
        )
        changed["failure"]["child_diagnostics"]["N3"]["bounded_tail"] = dict(
            changed["failure"]["child_diagnostics"]["N3"]["bounded_tail"]
        )
        changed["failure"]["child_diagnostics"]["N3"]["bounded_tail"][
            "safe_structured_summary"
        ] = dict(
            changed["failure"]["child_diagnostics"]["N3"]["bounded_tail"][
                "safe_structured_summary"
            ]
        )
        changed["failure"]["child_diagnostics"]["N3"]["bounded_tail"][
            "safe_structured_summary"
        ]["raw_text"] = "secret"
        changed = self._resign(changed)
        with self.assertRaisesRegex(jobs.CameraCropTerminalError, "summary fields"):
            jobs._validate_prior_diagnostic_value(changed)

        changed = dict(receipt)
        changed["failure"] = dict(changed["failure"])
        changed["failure"]["child_diagnostics"] = dict(
            changed["failure"]["child_diagnostics"]
        )
        changed["failure"]["child_diagnostics"]["D1"] = dict(
            changed["failure"]["child_diagnostics"]["D1"]
        )
        changed["failure"]["child_diagnostics"]["D1"]["runtime_preflight"] = dict(
            changed["failure"]["child_diagnostics"]["D1"]["runtime_preflight"]
        )
        changed["failure"]["child_diagnostics"]["D1"]["runtime_preflight"][
            "release_authority"
        ] = True
        changed = self._resign(changed)
        with self.assertRaisesRegex(jobs.CameraCropTerminalError, "preflight fields"):
            jobs._validate_prior_diagnostic_value(changed)

        changed = copy.deepcopy(receipt)
        changed["failure"]["traceback_frames"] = [{
            "source": {"absolute_path": "/etc/shadow"},
            "line": 1,
            "function": "safe_function",
        }]
        changed["failure"]["traceback_frame_count"] = 1
        changed = self._resign(changed)
        with self.assertRaisesRegex(jobs.CameraCropTerminalError, "redacted"):
            jobs._validate_prior_diagnostic_value(changed)

        changed = copy.deepcopy(receipt)
        changed["failure"]["traceback_frames"] = [{
            "source": {"absolute_path": "/usr/../../etc/shadow"},
            "line": 1,
            "function": "safe_function",
        }]
        changed["failure"]["traceback_frame_count"] = 1
        changed = self._resign(changed)
        with self.assertRaisesRegex(jobs.CameraCropTerminalError, "absolute path"):
            jobs._validate_prior_diagnostic_value(changed)

        changed = copy.deepcopy(receipt)
        changed["failure"]["child_diagnostics"]["N3"]["bounded_tail"][
            "safe_structured_summary"
        ]["traceback_frames"] = [{
            "source": {"absolute_path": "/usr/../home/ali/.ssh/id_rsa"},
            "line": 1,
            "function": "safe_function",
        }]
        changed["failure"]["child_diagnostics"]["N3"]["bounded_tail"][
            "safe_structured_summary"
        ]["line_count"] = 1
        changed["failure"]["child_diagnostics"]["N3"]["bounded_tail"][
            "safe_structured_summary"
        ]["recognized_line_count"] = 1
        changed = self._resign(changed)
        with self.assertRaisesRegex(jobs.CameraCropTerminalError, "absolute path"):
            jobs._validate_prior_diagnostic_value(changed)

        changed = copy.deepcopy(receipt)
        changed["failure"]["traceback_frames"] = [{
            "source": {"absolute_path": "/usr/lib/python3/example.py"},
            "line": 1,
            "function": "client_secret",
        }]
        changed["failure"]["traceback_frame_count"] = 1
        changed = self._resign(changed)
        with self.assertRaisesRegex(jobs.CameraCropTerminalError, "redacted"):
            jobs._validate_prior_diagnostic_value(changed)

        changed = copy.deepcopy(receipt)
        changed["failure"]["child_diagnostics"]["N3"]["runtime_preflight"][
            "python"
        ]["resolved_file"]["path"] = "/etc/passwd"
        changed = self._resign(changed)
        with self.assertRaisesRegex(jobs.CameraCropTerminalError, "resolved Python path"):
            jobs._validate_prior_diagnostic_value(changed)

        changed = copy.deepcopy(receipt)
        changed["failure"]["child_diagnostics"]["N3"]["bounded_tail"][
            "raw_bytes"
        ] += 1
        changed = self._resign(changed)
        with self.assertRaisesRegex(jobs.CameraCropTerminalError, "log arithmetic"):
            jobs._validate_prior_diagnostic_value(changed)

        changed = copy.deepcopy(receipt)
        changed["failure"]["child_diagnostics"]["N3"]["bounded_tail"][
            "maximum_bytes"
        ] = float(jobs.CHILD_LOG_TAIL_BYTES)
        changed = self._resign(changed)
        with self.assertRaisesRegex(jobs.CameraCropTerminalError, "log evidence"):
            jobs._validate_prior_diagnostic_value(changed)

        changed = copy.deepcopy(receipt)
        changed["failure"]["child_diagnostics"]["N3"]["bounded_tail"][
            "truncated"
        ] = True
        changed = self._resign(changed)
        with self.assertRaisesRegex(jobs.CameraCropTerminalError, "log arithmetic"):
            jobs._validate_prior_diagnostic_value(changed)

        changed = copy.deepcopy(receipt)
        changed["completed_at_utc"] = "2026-99-99T99:99:99Z"
        changed = self._resign(changed)
        with self.assertRaisesRegex(jobs.CameraCropTerminalError, "timestamp"):
            jobs._validate_prior_diagnostic_value(changed)

        scalar_mutations = (
            ("diagnostic_only", 1),
            ("simulator_state_render_used", -0.0),
        )
        for key, replacement in scalar_mutations:
            changed = copy.deepcopy(receipt)
            changed[key] = replacement
            changed = self._resign(changed)
            with self.subTest(key=key), self.assertRaisesRegex(
                jobs.CameraCropTerminalError, f"receipt changed: {key}"
            ):
                jobs._validate_prior_diagnostic_value(changed)

        changed = copy.deepcopy(receipt)
        changed["science_counts"]["model_runtime_loads"] = -0.0
        changed = self._resign(changed)
        with self.assertRaisesRegex(jobs.CameraCropTerminalError, "science_counts"):
            jobs._validate_prior_diagnostic_value(changed)

        changed = copy.deepcopy(receipt)
        changed["failure"]["message_bytes"] = 120.0
        changed = self._resign(changed)
        with self.assertRaisesRegex(jobs.CameraCropTerminalError, "completed two-child"):
            jobs._validate_prior_diagnostic_value(changed)

    def test_prior_receipt_signature_and_extra_fields_fail_closed(self) -> None:
        receipt = self._prior_receipt()
        receipt["status"] = "passed"
        with self.assertRaisesRegex(Exception, "signature changed"):
            jobs._validate_prior_diagnostic_value(receipt)
        receipt = self._prior_receipt()
        receipt["invented_authority"] = True
        receipt = self._resign(receipt)
        with self.assertRaisesRegex(jobs.CameraCropTerminalError, "fields changed"):
            jobs._validate_prior_diagnostic_value(receipt)

    def test_wave_is_deterministic_descriptor_only_and_binds_all_six_receipts(self) -> None:
        implementation = self._implementation()
        prerequisites = {
            name: self._descriptor(f"/fetched/{name}.json", index + 10)
            for index, name in enumerate(sorted(jobs.PREREQUISITE_CLUSTER_PATHS))
        }
        inputs = {
            name: (Path(row["path"]), row["sha256"])
            for name, row in prerequisites.items()
        }
        prior = {
            "receipt": self._descriptor("/fetched/camera002.json", 99),
            "results_commit": "b" * 40,
            "study_commit": "a" * 40,
            "authenticated_source_blobs": {},
        }
        patches = (
            mock.patch.object(jobs, "_local_implementation", return_value=implementation),
            mock.patch.object(jobs, "_runtime_contract", return_value={}),
            mock.patch.object(
                jobs.diagnostic, "validate_prerequisites", return_value=prerequisites
            ),
            mock.patch.object(jobs, "validate_prior_diagnostic", return_value=prior),
        )
        with patches[0], patches[1], patches[2], patches[3]:
            first = jobs.build_wave(
                study_commit="c" * 40,
                prerequisite_inputs=inputs,
                prior_diagnostic_path=Path("/fetched/camera002.json"),
                prior_diagnostic_sha256=prior["receipt"]["sha256"],
                results_root=Path("/fetched/results"),
                results_commit="b" * 40,
            )
        self.assertEqual(first["schema_version"], jobs.WAVE_SCHEMA)
        self.assertIn("not_dispatched", first["status"])
        self.assertIs(first["descriptor_dispatched"], False)
        self.assertEqual(len(first["replay_prerequisite_receipts"]), 5)
        self.assertEqual(first["prior_diagnostic"], prior)
        descriptor = first["jobs"][0]
        self.assertEqual(descriptor["job_id"], jobs.JOB_ID)
        command = " ".join(descriptor["argv"])
        self.assertIn("camera_crop_witness_terminal_jobs.py run", command)
        self.assertIn(prior["receipt"]["sha256"], command)
        self.assertIn(str(prior["receipt"]["bytes"]), command)
        self.assertIn(prior["study_commit"], command)
        self.assertNotIn("kubectl", command)
        self.assertEqual(descriptor["publish_log_tail_bytes"], 0)
        self.assertEqual(first["science_counts"], jobs.zero_science_counts())
        self.assertIs(first["safe_to_release_confirmation"], False)

    def test_runtime_descriptor_rejects_wrong_role_and_duplicate_hash_name(self) -> None:
        implementation = self._implementation()
        args = argparse.Namespace(
            study_commit="b" * 40,
            job_id=jobs.JOB_ID,
            expected_role="wmf-forecast-0912-worker-05",
            prior_diagnostic_sha256="e" * 64,
            prior_diagnostic_bytes=100,
            prior_diagnostic_study_commit="a" * 40,
            implementation=[
                (name, value["sha256"]) for name, value in implementation.items()
            ],
        )
        with self.assertRaisesRegex(jobs.CameraCropTerminalError, "queue identity"):
            jobs._runtime_descriptor(args)
        args.expected_role = jobs.WORKER_ROLE
        name, value = next(iter(implementation.items()))
        args.implementation.append((name, value["sha256"]))
        with self.assertRaisesRegex(jobs.CameraCropTerminalError, "duplicate implementation"):
            jobs._runtime_descriptor(args)

    def test_terminal_children_are_cpu_offline_and_controller_contained(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        for text in (
            '"CUDA_VISIBLE_DEVICES": ""',
            '"HF_HUB_OFFLINE": "1"',
            '"TRANSFORMERS_OFFLINE": "1"',
            "start_new_session=False",
            "inherits_queue_wrapper_process_group",
            "os.getpgrp() == os.getpid()",
        ):
            self.assertIn(text, source)

    def test_second_terminal_child_launch_failure_reaps_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary).resolve() / "raw"
            raw.mkdir()
            context = SimpleNamespace(source_root=raw)
            n3 = self._FakeProcess(701)

            def child_command(*, model, context, implementation, raw):
                return (
                    [sys.executable, "unused.py", model],
                    raw / model,
                    raw / jobs.MODEL_CONTRACT_NAMES[model],
                    raw / f"{model.lower()}_replay.log",
                )

            calls = 0

            def popen(*args, **kwargs):
                nonlocal calls
                calls += 1
                self.assertIs(kwargs["start_new_session"], False)
                if calls == 1:
                    return n3
                raise OSError("private launch detail")

            with mock.patch.object(jobs, "_child_command", side_effect=child_command), mock.patch.object(
                jobs.diagnostic, "_runtime_environment_diagnostic", return_value={"safe": True}
            ), mock.patch.object(jobs.subprocess, "Popen", side_effect=popen), mock.patch.object(
                jobs.os, "getpid", return_value=700
            ), mock.patch.object(jobs.os, "getpgrp", return_value=700):
                with self.assertRaises(jobs.CameraCropTerminalChildError) as caught:
                    jobs._launch_children(context=context, implementation={}, raw=raw)
            error = caught.exception
            self.assertTrue(n3.terminated)
            self.assertGreaterEqual(n3.wait_count, 1)
            self.assertIs(error.diagnostics["N3"]["launched"], True)
            self.assertIs(error.diagnostics["D1"]["launched"], False)
            self.assertIs(error.orchestration_failure["cleanup"]["N3"]["reaped"], True)
            self.assertNotIn("private launch detail", str(error.orchestration_failure))

    def test_raw_contract_and_both_witness_arrays_precede_publication(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        launch = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_launch_children"
        )
        launch_text = ast.get_source_segment(source, launch)
        assert launch_text is not None
        self.assertIn("validate_camera_crop_contract", launch_text)
        self.assertIn("_validate_witness_artifacts", launch_text)
        run = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_job"
        )
        run_text = ast.get_source_segment(source, run)
        assert run_text is not None
        self.assertLess(run_text.index("_launch_children"), run_text.index("_publish_success"))

    def test_atomic_success_publication_contains_exactly_three_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job_dir = Path(temporary).resolve() / jobs.JOB_ID
            raw = job_dir / "raw/camera_crop_witness_terminal"
            publish = job_dir / "publish"
            raw.mkdir(parents=True)
            publish.mkdir()
            contracts = self._raw_contracts(raw)
            with mock.patch.object(jobs.replay, "validate_camera_crop_contract"):
                receipt = jobs._publish_success(
                    job_dir=job_dir, raw=raw, publish=publish, contracts=contracts
                )
            self.assertEqual(
                {path.name for path in publish.iterdir()},
                {jobs.SUCCESS_RECEIPT, *jobs.MODEL_CONTRACT_NAMES.values()},
            )
            self.assertFalse((raw / "terminal_publish_staging").exists())
            jobs._validate_success_receipt(receipt, job_dir=job_dir)
            self.assertIs(receipt["safe_for_physical_alignment_input"], True)
            self.assertIs(receipt["safe_to_release_confirmation"], False)
            self.assertIs(receipt["confirmation_released"], False)

    def test_second_staged_copy_failure_never_partially_publishes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job_dir = Path(temporary).resolve() / jobs.JOB_ID
            raw = job_dir / "raw/camera_crop_witness_terminal"
            publish = job_dir / "publish"
            raw.mkdir(parents=True)
            publish.mkdir()
            contracts = self._raw_contracts(raw)
            original = jobs.queue.immutable_bytes
            calls = 0

            def fail_second(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected terminal second-copy failure")
                return original(*args, **kwargs)

            with mock.patch.object(jobs.replay, "validate_camera_crop_contract"), mock.patch.object(
                jobs.queue, "immutable_bytes", side_effect=fail_second
            ):
                with self.assertRaisesRegex(OSError, "second-copy"):
                    jobs._publish_success(
                        job_dir=job_dir, raw=raw, publish=publish, contracts=contracts
                    )
            self.assertEqual(list(publish.iterdir()), [])
            self.assertTrue((raw / "terminal_publish_staging").is_dir())
            self.assertEqual(len(list((raw / "terminal_publish_staging").iterdir())), 1)

    def test_post_commit_fsync_error_cannot_turn_published_success_into_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job_dir = Path(temporary).resolve() / jobs.JOB_ID
            raw = job_dir / "raw/camera_crop_witness_terminal"
            publish = job_dir / "publish"
            raw.mkdir(parents=True)
            publish.mkdir()
            contracts = self._raw_contracts(raw)
            calls = []

            def fsync(path):
                calls.append(Path(path))
                if len(calls) == 4:
                    raise OSError("post-commit fsync injection")

            with mock.patch.object(jobs.replay, "validate_camera_crop_contract"), mock.patch.object(
                jobs, "_fsync_directory", side_effect=fsync
            ):
                receipt = jobs._publish_success(
                    job_dir=job_dir, raw=raw, publish=publish, contracts=contracts
                )
            self.assertEqual(receipt["status"], "passed")
            self.assertEqual(
                {path.name for path in publish.iterdir()},
                {jobs.SUCCESS_RECEIPT, *jobs.MODEL_CONTRACT_NAMES.values()},
            )
            self.assertGreaterEqual(calls.count(raw), 2)
            self.assertGreaterEqual(calls.count(job_dir), 2)

    def test_failure_removes_allowed_partial_publish_and_signs_zero_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            control = Path(temporary).resolve() / "control"
            job_dir = control / "jobs" / jobs.JOB_ID
            publish = job_dir / "publish"
            publish.mkdir(parents=True)
            (publish / jobs.MODEL_CONTRACT_NAMES["N3"]).write_bytes(b"partial")
            context = SimpleNamespace(study_commit="d" * 40, role=jobs.WORKER_ROLE)
            with mock.patch.object(jobs, "CONTROL_ROOT", control):
                jobs._write_failure(job_dir, context, RuntimeError("private detail"))
            self.assertEqual({path.name for path in publish.iterdir()}, {jobs.FAILURE_RECEIPT})
            receipt = jobs.replay.load_json(
                publish / jobs.FAILURE_RECEIPT, "terminal failure receipt"
            )
            jobs.queue.verify_signed_document(receipt, "terminal failure receipt")
            self.assertEqual(receipt["science_counts"], jobs.zero_science_counts())
            self.assertIs(receipt["safe_for_physical_alignment_input"], False)
            self.assertIs(receipt["safe_to_release_confirmation"], False)
            self.assertIs(receipt["confirmation_released"], False)
            self.assertNotIn("private detail", str(receipt))

    def test_success_receipt_is_exactly_accepted_by_alignment_consumer(self) -> None:
        job_dir = jobs.CONTROL_ROOT / "jobs" / jobs.JOB_ID
        outputs = {}
        for ordinal, model in enumerate(("N3", "D1"), start=1):
            outputs[model] = {
                "path": str(job_dir / "publish" / jobs.MODEL_CONTRACT_NAMES[model]),
                "sha256": f"{ordinal:064x}",
                "bytes": ordinal + 100,
                "schema_version": jobs.replay.SCHEMA,
                "model_id": model,
                "camera_crop_id": jobs.EXPECTED_CAMERA_CROP_IDS[model],
                "payload_sha256": f"{ordinal + 10:064x}",
            }
        receipt = jobs._success_receipt(job_dir=job_dir, outputs=outputs)
        normalized_job, normalized_outputs = alignment._validate_camera_job(
            receipt,
            file_identity={"sha256": "f" * 64, "bytes": 999},
        )
        self.assertEqual(normalized_job["path"], str(job_dir / "publish" / jobs.SUCCESS_RECEIPT))
        self.assertEqual(set(normalized_outputs), {"N3", "D1"})
        self.assertEqual(normalized_outputs["N3"]["camera_crop_id"], outputs["N3"]["camera_crop_id"])

    def test_alignment_compatibility_fails_if_release_or_job_identity_broadens(self) -> None:
        job_dir = jobs.CONTROL_ROOT / "jobs" / jobs.JOB_ID
        outputs = {
            model: {
                "path": str(job_dir / "publish" / jobs.MODEL_CONTRACT_NAMES[model]),
                "sha256": f"{ordinal:064x}",
                "bytes": ordinal + 100,
                "schema_version": jobs.replay.SCHEMA,
                "model_id": model,
                "camera_crop_id": jobs.EXPECTED_CAMERA_CROP_IDS[model],
                "payload_sha256": f"{ordinal + 10:064x}",
            }
            for ordinal, model in enumerate(("N3", "D1"), start=1)
        }
        receipt = jobs._success_receipt(job_dir=job_dir, outputs=outputs)
        changed = dict(receipt)
        changed["confirmation_released"] = True
        changed = self._resign(changed)
        with self.assertRaisesRegex(Exception, "camera terminal receipt changed"):
            alignment._validate_camera_job(
                changed, file_identity={"sha256": "f" * 64, "bytes": 999}
            )

        changed = dict(receipt)
        changed["job_id"] = jobs.PRIOR_JOB_ID
        changed["job_dir"] = str(jobs.CONTROL_ROOT / "jobs" / jobs.PRIOR_JOB_ID)
        changed = self._resign(changed)
        with self.assertRaisesRegex(Exception, "diagnostic/nonterminal"):
            alignment._validate_camera_job(
                changed, file_identity={"sha256": "f" * 64, "bytes": 999}
            )

    def test_witness_path_or_symlink_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            original = root / "original_camera_crop.npy"
            generated = root / "generated_decoded_crop.npy"
            original.write_bytes(b"original")
            generated.write_bytes(b"generated")
            value = {
                "original_camera_replay": {
                    "witness_artifact": jobs.replay.file_identity(
                        original, label="original"
                    )
                },
                "generated_decoded_crop": {
                    "witness_artifact": jobs.replay.file_identity(
                        generated, label="generated"
                    )
                },
            }
            jobs._validate_witness_artifacts("N3", value, root)
            value["generated_decoded_crop"]["witness_artifact"] = dict(
                value["generated_decoded_crop"]["witness_artifact"]
            )
            value["generated_decoded_crop"]["witness_artifact"]["path"] = str(original)
            with self.assertRaisesRegex(jobs.CameraCropTerminalError, "path changed"):
                jobs._validate_witness_artifacts("N3", value, root)


if __name__ == "__main__":
    unittest.main()
