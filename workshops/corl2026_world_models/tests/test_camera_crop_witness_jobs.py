from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/forecast_layout/camera_crop_witness_jobs.py"
)
SPEC = importlib.util.spec_from_file_location("camera_crop_witness_jobs", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
jobs = importlib.util.module_from_spec(SPEC)
import sys

sys.modules[SPEC.name] = jobs
SPEC.loader.exec_module(jobs)

CLUSTER_QUEUE_PATH = (
    Path(__file__).resolve().parents[1]
    / "execution/20260912/autonomy/cluster_queue.py"
)
CLUSTER_SPEC = importlib.util.spec_from_file_location(
    "camera_crop_cluster_queue", CLUSTER_QUEUE_PATH
)
assert CLUSTER_SPEC is not None and CLUSTER_SPEC.loader is not None
cluster_queue = importlib.util.module_from_spec(CLUSTER_SPEC)
sys.modules[CLUSTER_SPEC.name] = cluster_queue
CLUSTER_SPEC.loader.exec_module(cluster_queue)


class CameraCropWitnessJobTests(unittest.TestCase):
    class _FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid
            self.returncode = None
            self.terminated = False
            self.killed = False
            self.wait_count = 0

        def poll(self):
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def kill(self) -> None:
            self.killed = True
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

    def _inputs(self, root: Path) -> dict[str, tuple[Path, str]]:
        runtime = jobs._runtime_contract(jobs.REPOSITORY_ROOT)
        return {
            name: (root / f"{name}.json", expected["sha256"])
            for name, expected in runtime["prerequisite_receipts"].items()
        }

    def test_contract_and_filenames_are_canonical_worker06_zero_science(self) -> None:
        contract = jobs._runtime_contract(jobs.REPOSITORY_ROOT)
        self.assertEqual(contract["job"]["job_id"], jobs.JOB_ID)
        self.assertEqual(jobs.JOB_ID, "camera-crop-replay-witness-002")
        self.assertEqual(contract["prior_attempt"]["job_id"], jobs.PRIOR_JOB_ID)
        self.assertEqual(contract["prior_attempt"]["status"], "technical_invalid")
        self.assertIs(contract["prior_attempt"]["preserved"], True)
        self.assertIs(contract["child_runtime"]["diagnostic_only"], True)
        self.assertIs(contract["child_runtime"]["publish_crop_contracts"], False)
        self.assertEqual(contract["job"]["worker_role"], "wmf-forecast-0912-worker-06")
        self.assertEqual(contract["job"]["publish_log_tail_bytes"], 0)
        self.assertEqual(jobs.SUCCESS_RECEIPT, "camera_crop_witness_job_receipt.json")
        self.assertEqual(
            jobs.MODEL_CONTRACT_NAMES,
            {
                "N3": "n3_camera_crop_contract.json",
                "D1": "d1_camera_crop_contract.json",
            },
        )
        self.assertEqual(contract["science_counts"], jobs.zero_science_counts())
        self.assertIs(contract["safe_to_release_confirmation"], False)
        self.assertIs(contract["confirmation_released"], False)

        mutated = dict(contract)
        mutated["confirmation_released"] = True
        with mock.patch.object(jobs.replay, "load_json", return_value=mutated):
            with self.assertRaisesRegex(jobs.CameraCropQueueError, "authority"):
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

    def test_wave_is_deterministic_receipt_gated_and_never_dispatched(self) -> None:
        implementation = self._implementation()
        prerequisites = {
            name: {
                "path": f"/fetched/{name}.json",
                "bytes": index + 1,
                "sha256": f"{index + 20:064x}",
            }
            for index, name in enumerate(sorted(jobs.PREREQUISITE_CLUSTER_PATHS))
        }
        inputs = {
            name: (Path(value["path"]), value["sha256"])
            for name, value in prerequisites.items()
        }
        with mock.patch.object(jobs, "_local_implementation", return_value=implementation), mock.patch.object(
            jobs, "validate_prerequisites", return_value=prerequisites
        ):
            first = jobs.build_wave(study_commit="a" * 40, prerequisite_inputs=inputs)
            second = jobs.build_wave(study_commit="a" * 40, prerequisite_inputs=inputs)
        self.assertEqual(first, second)
        self.assertEqual(first["schema_version"], jobs.WAVE_SCHEMA)
        self.assertIn("not_dispatched", first["status"])
        self.assertIs(first["diagnostic_only"], True)
        self.assertEqual(first["science_counts"], jobs.zero_science_counts())
        self.assertIs(first["safe_to_release_confirmation"], False)
        descriptor = first["jobs"][0]
        self.assertEqual(descriptor["job_id"], jobs.JOB_ID)
        self.assertEqual(descriptor["role"], "wmf-forecast-0912-worker-06")
        self.assertEqual(descriptor["argv"][0], "/usr/bin/python3")
        self.assertEqual(descriptor["publish_log_tail_bytes"], 0)
        self.assertEqual(
            cluster_queue.normalize_job(descriptor)["publish_log_tail_bytes"], 0
        )
        command = " ".join(descriptor["argv"])
        self.assertIn("camera_crop_witness_jobs.py run", command)
        self.assertNotIn("kubectl", command)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", command)
        for value in implementation.values():
            self.assertIn(value["sha256"], command)
        self.assertEqual(
            first["expected_outputs"],
            {"technical_invalid_diagnostic_receipt": jobs.FAILURE_RECEIPT},
        )

    def test_zero_outer_log_budget_prevents_queue_snapshot_traceback_leak(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary).resolve()
            job_dir = state / "jobs" / jobs.JOB_ID
            job_dir.mkdir(parents=True)
            descriptor = {
                "job_id": jobs.JOB_ID,
                "source_commit": "a" * 40,
                "publish_log_tail_bytes": jobs.PUBLISH_LOG_TAIL_BYTES,
            }
            result = {
                "status": "failed",
                "returncode": 1,
                "stderr": {"sha256": "b" * 64},
                "stdout": {"sha256": "c" * 64},
            }
            (job_dir / "descriptor.json").write_text(json.dumps(descriptor), encoding="utf-8")
            (job_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
            secret = "OPENAI_API_KEY=sk-proj-outer-wrapper-leak"
            (job_dir / "stderr.log").write_text(secret, encoding="utf-8")
            (job_dir / "stdout.log").write_text(secret, encoding="utf-8")
            snapshot = cluster_queue.snapshot(state)
            self.assertEqual(jobs.PUBLISH_LOG_TAIL_BYTES, 0)
            self.assertNotIn("log_tails", snapshot["jobs"][0])
            self.assertNotIn("sk-proj", str(snapshot))

    def test_builder_requires_exactly_five_unique_explicit_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            inputs = self._inputs(root)
            inputs.pop("d1_generation")
            with self.assertRaisesRegex(jobs.CameraCropQueueError, "exactly five"):
                jobs.validate_prerequisites(inputs)

            inputs = self._inputs(root)
            duplicate = root / "same.json"
            first, second = sorted(inputs)[:2]
            inputs[first] = (duplicate, inputs[first][1])
            inputs[second] = (duplicate, inputs[second][1])
            runtime = jobs._runtime_contract(jobs.REPOSITORY_ROOT)
            with mock.patch.object(jobs, "_runtime_contract", return_value=runtime), mock.patch.object(
                jobs.replay, "file_identity", return_value={}
            ), mock.patch.object(jobs.replay, "load_json", return_value={}), mock.patch.object(
                jobs, "_validate_prerequisite_value"
            ):
                with self.assertRaisesRegex(jobs.CameraCropQueueError, "ambiguous"):
                    jobs.validate_prerequisites(inputs)

    def test_prerequisite_validation_rejects_false_scientific_status(self) -> None:
        runtime = jobs._runtime_contract(jobs.REPOSITORY_ROOT)
        expected = runtime["prerequisite_receipts"]["n3_generation"]
        value = {
            "schema_version": expected["schema_version"],
            "status": "passed",
            "qualified": False,
            "generation_request_count": 6,
            "robot_episode_count": 0,
            "study_id": jobs.STUDY_ID,
        }
        with self.assertRaisesRegex(jobs.CameraCropQueueError, "passed zero-policy"):
            jobs._validate_prerequisite_value("n3_generation", value, expected)

    def test_runtime_descriptor_rejects_wrong_role_and_duplicate_hash_name(self) -> None:
        implementation = self._implementation()
        args = argparse.Namespace(
            study_commit="b" * 40,
            job_id=jobs.JOB_ID,
            expected_role="wmf-forecast-0912-worker-05",
            implementation=[(name, value["sha256"]) for name, value in implementation.items()],
        )
        with self.assertRaisesRegex(jobs.CameraCropQueueError, "queue identity"):
            jobs._runtime_descriptor(args)

        name, value = next(iter(implementation.items()))
        args.expected_role = jobs.WORKER_ROLE
        args.implementation.append((name, value["sha256"]))
        with self.assertRaisesRegex(jobs.CameraCropQueueError, "duplicate implementation"):
            jobs._runtime_descriptor(args)

    def test_child_environment_is_explicitly_cpu_only_and_detached(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn('"CUDA_VISIBLE_DEVICES": ""', source)
        self.assertIn('"PYTHONUNBUFFERED": "1"', source)
        self.assertIn("start_new_session=False", source)
        self.assertIn("inherits_queue_wrapper_process_group", source)
        self.assertIn("os.getpgrp() == os.getpid()", source)
        self.assertIn("child_launches.json", source)

    def test_failure_after_first_contract_copy_leaves_no_partial_publish_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            raw = root / "raw"
            publish = root / "publish"
            raw.mkdir()
            publish.mkdir()
            contracts = {}
            for model in ("N3", "D1"):
                path = raw / jobs.MODEL_CONTRACT_NAMES[model]
                path.write_bytes((model + "-contract\n").encode())
                contracts[model] = {
                    "identity": jobs.replay.file_identity(path, label=f"{model} raw"),
                    "value": {
                        "camera_crop_id": f"{model.lower()}-crop",
                        "payload_sha256": "a" * 64,
                    },
                }
            original = jobs.queue.immutable_bytes
            calls = 0

            def fail_second(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected second-copy failure")
                return original(*args, **kwargs)

            with mock.patch.object(jobs.queue, "immutable_bytes", side_effect=fail_second):
                with self.assertRaisesRegex(OSError, "second-copy"):
                    jobs._publish_contracts(publish, contracts)
            self.assertEqual(list(publish.iterdir()), [])
            self.assertTrue(all(Path(row["identity"]["path"]).is_file() for row in contracts.values()))

    def test_bounded_child_logs_are_authenticated_structural_and_secret_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            log_paths = {}
            runtime_diagnostics = {}
            exits = {"N3": 1, "D1": 7}
            for model in ("N3", "D1"):
                path = root / f"{model}.log"
                preflight = {
                    "schema_version": "wmf-camera-crop-child-runtime-preflight-v1",
                    "event": "runtime_preflight",
                    "phase": "after_replay_failure",
                    "model_id": model,
                    "python": {
                        "lexical_path": "/usr/bin/python3",
                        "resolved_path": "/usr/bin/python3.11",
                    },
                    "module_specs": {
                        "torch": {
                            "found": True,
                            "origin": "/usr/lib/python3/torch/__init__.py",
                            "search_locations": ["/usr/lib/python3/torch"],
                        }
                    },
                    "path_environment": {"PYTHONPATH": [], "PYTHONHOME": [], "VIRTUAL_ENV": []},
                    "cuda_visible_devices_empty": True,
                }
                path.write_bytes(
                    b"x" * (jobs.CHILD_LOG_TAIL_BYTES + 17)
                    + b"\n"
                    + jobs.queue.compact_bytes(preflight)
                    + b'\n  File "/usr/lib/python3/test.py", line 42, in load\n'
                    + b"ModuleNotFoundError: No module named 'torchvision'\n"
                    + b"Authorization: Bearer-secret\n"
                    + b"OPENAI_API_KEY=sk-proj-do-not-publish\n"
                    + b"AWS_SECRET_ACCESS_KEY=do-not-publish\n"
                    + b"HF_TOKEN=hf_do_not_publish\n"
                    + b"SLACK_BOT_TOKEN=xoxb-do-not-publish\n"
                    + b"client_secret=do-not-publish\n"
                    + b"github_pat_abcdefghijklmnopqrstuvwxyz123456\n"
                    + b"/data/users/ali/vla_wam/private/task/deploy_key\n"
                )
                log_paths[model] = path
                runtime_diagnostics[model] = {"safe": model}
            diagnostics = jobs._child_diagnostics(
                log_paths=log_paths,
                exits=exits,
                runtime_diagnostics=runtime_diagnostics,
                launches={},
                started=set(),
            )
            for model in ("N3", "D1"):
                row = diagnostics[model]
                self.assertEqual(row["exit_code"], exits[model])
                self.assertEqual(
                    row["full_log"]["sha256"],
                    jobs.queue.sha256_file(root / f"{model}.log"),
                )
                tail = (root / f"{model}.log").read_bytes()[-jobs.CHILD_LOG_TAIL_BYTES:]
                self.assertLessEqual(row["bounded_tail"]["raw_bytes"], jobs.CHILD_LOG_TAIL_BYTES)
                self.assertEqual(
                    row["bounded_tail"]["raw_sha256"], jobs.queue.sha256_bytes(tail)
                )
                self.assertGreater(
                    row["bounded_tail"]["partial_leading_line_dropped_bytes"], 0
                )
                summary = row["bounded_tail"]["safe_structured_summary"]
                self.assertEqual(len(summary["preflight_events"]), 1)
                self.assertEqual(summary["traceback_frames"][0]["line"], 42)
                self.assertEqual(
                    summary["terminal_errors"][0]["missing_module"], "torchvision"
                )
                self.assertIs(summary["arbitrary_text_published"], False)
                serialized = str(row)
                for secret in (
                    "Bearer-secret", "sk-proj", "AWS_SECRET", "hf_do", "xoxb-",
                    "client_secret", "github_pat_", "deploy_key",
                ):
                    self.assertNotIn(secret, serialized)
                self.assertIs(row["argv_published"], False)
                self.assertIs(row["environment_values_published"], False)
                self.assertEqual(row["runtime_preflight"], {"safe": model})
                self.assertIs(row["launched"], False)

    def test_runtime_environment_diagnostic_is_path_only_and_private_safe(self) -> None:
        environment = {
            "PYTHONPATH": (
                "/exact/source:/data/users/ali/vla_wam/private/task/key:relative:"
                "/usr/lib/sk-proj-never-publish:/opt/client_secret/value"
            ),
            "PYTHONHOME": "/exact/home",
            "VIRTUAL_ENV": "/exact/venv",
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "GH_TOKEN": "must-never-be-copied",
        }
        value = jobs._runtime_environment_diagnostic(Path(sys.executable), environment)
        self.assertIn(
            "redacted_value_sha256", value["path_environment"]["PYTHONPATH"][0]
        )
        self.assertNotIn("private", str(value["path_environment"]["PYTHONPATH"][1]))
        self.assertNotIn("relative", str(value["path_environment"]["PYTHONPATH"][2]))
        self.assertNotIn("sk-proj", str(value))
        self.assertNotIn("client_secret", str(value))
        self.assertNotIn("GH_TOKEN", str(value))
        self.assertNotIn("must-never", str(value))
        self.assertIs(value["argv_published"], False)
        self.assertIs(value["arbitrary_environment_values_published"], False)

    def test_child_failure_receipt_binds_diagnostics_and_zero_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            control = Path(temporary).resolve() / "control"
            job_dir = control / "jobs" / jobs.JOB_ID
            (job_dir / "publish").mkdir(parents=True)
            diagnostics = {
                model: {
                    "model_id": model,
                    "exit_code": 1,
                    "full_log": {"path": f"/retained/{model}.log", "bytes": 1, "sha256": "a" * 64},
                    "bounded_tail": {
                        "maximum_bytes": jobs.CHILD_LOG_TAIL_BYTES,
                        "source_offset_bytes": 0,
                        "raw_bytes": 1,
                        "raw_sha256": "b" * 64,
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
                    "launch_receipt": None,
                    "runtime_preflight": {},
                }
                for model in ("N3", "D1")
            }
            error = jobs.CameraCropChildError({"N3": 1, "D1": 1}, diagnostics)
            context = SimpleNamespace(study_commit="d" * 40, role=jobs.WORKER_ROLE)
            with mock.patch.object(jobs, "CONTROL_ROOT", control):
                try:
                    raise error
                except jobs.CameraCropChildError as caught:
                    jobs._write_failure(job_dir, context, caught)
            receipt = jobs.replay.load_json(
                job_dir / "publish" / jobs.FAILURE_RECEIPT, "test failure receipt"
            )
            jobs.queue.verify_signed_document(receipt, "test failure receipt")
            self.assertEqual(receipt["failure"]["child_diagnostics"], diagnostics)
            self.assertEqual(receipt["failure"]["child_exits"], {"N3": 1, "D1": 1})
            self.assertEqual(receipt["science_counts"], jobs.zero_science_counts())
            self.assertIs(receipt["diagnostic_only"], True)
            self.assertIs(receipt["safe_to_release_confirmation"], False)
            self.assertIs(receipt["confirmation_released"], False)

    def test_generic_failure_receipt_never_embeds_exception_text_or_traceback_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            control = Path(temporary).resolve() / "control"
            job_dir = control / "jobs" / jobs.JOB_ID
            (job_dir / "publish").mkdir(parents=True)
            context = SimpleNamespace(study_commit="d" * 40, role=jobs.WORKER_ROLE)
            secret = "OPENAI_API_KEY=sk-proj-never-sign client_secret=never-sign"
            with mock.patch.object(jobs, "CONTROL_ROOT", control):
                try:
                    raise RuntimeError(secret)
                except RuntimeError as error:
                    jobs._write_failure(job_dir, context, error)
            payload = (job_dir / "publish" / jobs.FAILURE_RECEIPT).read_text(
                encoding="utf-8"
            )
            self.assertNotIn("sk-proj", payload)
            self.assertNotIn("client_secret", payload)
            receipt = jobs.replay.load_json(
                job_dir / "publish" / jobs.FAILURE_RECEIPT, "safe generic failure"
            )
            jobs.queue.verify_signed_document(receipt, "safe generic failure")
            self.assertIs(receipt["failure"]["exception_message_published"], False)
            self.assertIs(receipt["failure"]["traceback_source_text_published"], False)

    def test_second_child_launch_failure_reaps_first_and_persists_truthful_launch(self) -> None:
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
                raise OSError("OPENAI_API_KEY=sk-proj-launch-secret")

            with mock.patch.object(jobs, "_child_command", side_effect=child_command), mock.patch.object(
                jobs, "_runtime_environment_diagnostic", return_value={"safe": True}
            ), mock.patch.object(jobs.subprocess, "Popen", side_effect=popen), mock.patch.object(
                jobs.os, "getpid", return_value=700
            ), mock.patch.object(jobs.os, "getpgrp", return_value=700):
                with self.assertRaises(jobs.CameraCropChildError) as caught:
                    jobs._launch_children(context=context, implementation={}, raw=raw)

            error = caught.exception
            self.assertTrue(n3.terminated)
            self.assertGreaterEqual(n3.wait_count, 1)
            self.assertIs(error.diagnostics["N3"]["launched"], True)
            self.assertIs(error.diagnostics["D1"]["launched"], False)
            self.assertIsNotNone(error.diagnostics["N3"]["launch_receipt"])
            self.assertIsNone(error.diagnostics["D1"]["launch_receipt"])
            self.assertIs(
                error.orchestration_failure["cleanup"]["N3"]["reaped"], True
            )
            self.assertTrue(
                (raw / "child_launch_receipts" / "n3_launch.json").is_file()
            )
            launch = jobs.replay.load_json(
                raw / "child_launch_receipts" / "n3_launch.json", "N3 launch"
            )
            jobs.queue.verify_signed_document(launch, "N3 launch")
            self.assertEqual(launch["wrapper_pid"], 700)
            self.assertEqual(launch["process_group_id"], 700)
            self.assertIs(launch["start_new_session"], False)
            self.assertIs(launch["inherits_queue_wrapper_process_group"], True)
            self.assertIs(launch["confirmation_released"], False)
            self.assertNotIn("sk-proj", str(error.orchestration_failure))

    def test_launch_receipt_write_failure_reaps_started_child_and_does_not_lie(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary).resolve() / "raw"
            raw.mkdir()
            context = SimpleNamespace(source_root=raw)
            n3 = self._FakeProcess(801)

            def child_command(*, model, context, implementation, raw):
                return (
                    [sys.executable, "unused.py", model],
                    raw / model,
                    raw / jobs.MODEL_CONTRACT_NAMES[model],
                    raw / f"{model.lower()}_replay.log",
                )

            with mock.patch.object(jobs, "_child_command", side_effect=child_command), mock.patch.object(
                jobs, "_runtime_environment_diagnostic", return_value={"safe": True}
            ), mock.patch.object(jobs.subprocess, "Popen", return_value=n3), mock.patch.object(
                jobs.queue,
                "immutable_json",
                side_effect=OSError("AWS_SECRET_ACCESS_KEY=receipt-secret"),
            ), mock.patch.object(jobs.os, "getpid", return_value=800), mock.patch.object(
                jobs.os, "getpgrp", return_value=800
            ):
                with self.assertRaises(jobs.CameraCropChildError) as caught:
                    jobs._launch_children(context=context, implementation={}, raw=raw)

            error = caught.exception
            self.assertTrue(n3.terminated)
            self.assertGreaterEqual(n3.wait_count, 1)
            self.assertIs(error.diagnostics["N3"]["launched"], True)
            self.assertIsNone(error.diagnostics["N3"]["launch_receipt"])
            self.assertIs(error.diagnostics["D1"]["launched"], False)
            self.assertIs(
                error.orchestration_failure["cleanup"]["N3"]["reaped"], True
            )
            self.assertNotIn("receipt-secret", str(error.orchestration_failure))

    def test_run_job_is_diagnostic_only_and_cannot_publish_contracts(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        run = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_job"
        )
        text = ast.get_source_segment(source, run)
        assert text is not None
        self.assertNotIn("_publish_contracts", text)
        self.assertNotIn("SUCCESS_RECEIPT", text)
        self.assertIn('runtime["child_runtime"]["diagnostic_only"] is True', text)
        self.assertIn('runtime["child_runtime"]["publish_crop_contracts"] is False', text)
        self.assertIn("CameraCropChildError", text)


if __name__ == "__main__":
    unittest.main()
