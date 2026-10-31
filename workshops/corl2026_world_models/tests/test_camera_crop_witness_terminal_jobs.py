from __future__ import annotations

import argparse
import ast
import copy
import io
import importlib.util
from pathlib import Path
import stat
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

CLUSTER_QUEUE_PATH = (
    WORKSHOP_ROOT / "execution/20260912/autonomy/cluster_queue.py"
)
CLUSTER_QUEUE_SPEC = importlib.util.spec_from_file_location(
    "terminal_camera_cluster_queue_consumer", CLUSTER_QUEUE_PATH
)
assert CLUSTER_QUEUE_SPEC is not None and CLUSTER_QUEUE_SPEC.loader is not None
cluster_queue = importlib.util.module_from_spec(CLUSTER_QUEUE_SPEC)
sys.modules[CLUSTER_QUEUE_SPEC.name] = cluster_queue
CLUSTER_QUEUE_SPEC.loader.exec_module(cluster_queue)


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

    @staticmethod
    def _git(root: Path, *arguments: str, binary: bool = False):
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
        )
        return result.stdout if binary else result.stdout.decode("utf-8").strip()

    def _init_git_repo(self, root: Path) -> str:
        root.mkdir(parents=True, exist_ok=True)
        self._git(root, "init", "-q")
        self._git(root, "config", "user.email", "camera-terminal-test@example.invalid")
        self._git(root, "config", "user.name", "Camera Terminal Test")
        self._git(root, "add", "-A")
        self._git(root, "commit", "-q", "-m", "fixture")
        return self._git(root, "rev-parse", "HEAD")

    def _commit_all(self, root: Path, message: str) -> str:
        self._git(root, "add", "-A")
        self._git(root, "commit", "-q", "-m", message)
        return self._git(root, "rev-parse", "HEAD")

    def _control_fixture(self, root: Path, *, jobs_value=None):
        queue_path = root / jobs.CONTROL_QUEUE_RELATIVE
        queue_path.parent.mkdir(parents=True)
        queue_value = {
            "schema_version": jobs.QUEUE_MANIFEST_SCHEMA,
            "namespace": jobs.NAMESPACE,
            "shutdown": False,
            "jobs": [] if jobs_value is None else jobs_value,
        }
        queue_path.write_bytes(jobs.queue.canonical_bytes(queue_value))
        implementation_path = root / "impl.py"
        implementation_payload = b"print('pinned implementation')\n"
        implementation_path.write_bytes(implementation_payload)
        commit = self._init_git_repo(root)
        implementation = {
            "impl": {
                "path": str(jobs.REPOSITORY_ROOT / "impl.py"),
                "bytes": len(implementation_payload),
                "sha256": jobs.queue.sha256_bytes(implementation_payload),
            }
        }
        return commit, queue_path, implementation, queue_value

    def _queue_auth_fixture(self, base: Path) -> dict:
        control_root = base / "control"
        control_commit, queue_path, implementation, _queue = self._control_fixture(
            control_root
        )
        results_root = base / "results"
        values = self._synthetic_prerequisite_values()
        expected = {}
        inputs = {}
        for name, value in values.items():
            path = results_root / jobs.PREREQUISITE_RESULTS_RELATIVES[name]
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = jobs.queue.canonical_bytes(value)
            path.write_bytes(payload)
            digest = jobs.queue.sha256_bytes(payload)
            inputs[name] = (path, digest)
            expected[name] = {
                "bytes": len(payload),
                "sha256": digest,
                "schema_version": value["schema_version"],
                "model_id": value["model_id"],
                "status": value["status"],
                "decision": value["decision"],
            }
        terminal_contract = control_root / jobs.TERMINAL_CONTRACT_RELATIVE
        terminal_contract.parent.mkdir(parents=True, exist_ok=True)
        terminal_contract.write_bytes(jobs.queue.canonical_bytes({}))
        replay_contract = control_root / jobs.REPLAY_CONTRACT_RELATIVE
        replay_contract.parent.mkdir(parents=True, exist_ok=True)
        replay_contract.write_bytes(jobs.queue.canonical_bytes({
            "prerequisite_receipts": expected,
        }))
        control_commit = self._commit_all(
            control_root, "add authenticated runtime contracts"
        )
        prior = self._prior_receipt()
        prior["study_commit"] = control_commit
        prior = self._resign(prior)
        prior_path = results_root / jobs.PRIOR_DIAGNOSTIC_RESULTS_RELATIVE
        prior_path.parent.mkdir(parents=True, exist_ok=True)
        prior_payload = jobs.queue.canonical_bytes(prior)
        prior_path.write_bytes(prior_payload)
        results_commit = self._init_git_repo(results_root)
        return {
            "control_root": control_root,
            "control_commit": control_commit,
            "queue_path": queue_path,
            "implementation": implementation,
            "results_root": results_root,
            "results_commit": results_commit,
            "inputs": inputs,
            "expected": expected,
            "prior_path": prior_path,
            "prior_sha256": jobs.queue.sha256_bytes(prior_payload),
            "source_pin": {Path("impl.py"): implementation["impl"]["sha256"]},
        }

    @staticmethod
    def _synthetic_prerequisite_values() -> dict[str, dict]:
        return {
            "n3_source_audit": {
                "schema_version": "source-audit-v1",
                "study_id": jobs.STUDY_ID,
                "model_id": "N3",
                "status": "passed",
                "decision": "qualified",
                "source": {"commit": "411d25b2e35bc441126f48c44a4b93e1c0564274"},
            },
            "d1_source_audit": {
                "schema_version": "source-audit-v1",
                "study_id": jobs.STUDY_ID,
                "model_id": "D1",
                "status": "passed",
                "decision": "qualified",
                "source": {"commit": "ab790c198fbce33503358efbbd4187ce9a89adf3"},
            },
            "n3_generation": {
                "schema_version": "n3-generation-v1",
                "study_id": jobs.STUDY_ID,
                "model_id": "N3",
                "status": "passed",
                "decision": "qualified",
                "qualified": True,
                "generation_request_count": 6,
                "robot_episode_count": 0,
            },
            "d1_generation": {
                "schema_version": "d1-generation-v1",
                "study_id": jobs.STUDY_ID,
                "model_id": "D1",
                "status": "passed",
                "decision": "qualified",
                "model_returned_action_executed": False,
                "selected_request": {"decoded_rgb_shape": [9, 352, 640, 3]},
            },
            "d1_qualification": {
                "schema_version": "d1-qualification-v1",
                "study_id": jobs.STUDY_ID,
                "model_id": "D1",
                "status": "passed",
                "decision": "qualified",
                "behavioral_episode_count": 0,
                "generation_request_count": 6,
            },
        }

    def _built_wave(self, study_commit: str = "c" * 40) -> dict:
        implementation = jobs._local_implementation()
        prerequisites = {
            name: self._descriptor(f"/fetched/{name}.json", index + 10)
            for index, name in enumerate(sorted(jobs.PREREQUISITE_CLUSTER_PATHS))
        }
        inputs = {
            name: (Path(row["path"]), row["sha256"])
            for name, row in prerequisites.items()
        }
        prior_path = (
            Path("/fetched/results") / jobs.PRIOR_DIAGNOSTIC_RESULTS_RELATIVE
        )
        prior = {
            "receipt": self._descriptor(str(prior_path), 99),
            "results_commit": "b" * 40,
            "study_commit": "a" * 40,
            "authenticated_source_blobs": {
                str(path): {"sha256": digest, "bytes": index + 1}
                for index, (path, digest) in enumerate(
                    jobs.PRIOR_SOURCE_PINS.items()
                )
            },
        }
        with mock.patch.object(
            jobs, "_local_implementation", return_value=implementation
        ), mock.patch.object(
            jobs, "_runtime_contract", return_value={}
        ), mock.patch.object(
            jobs.diagnostic, "validate_prerequisites", return_value=prerequisites
        ), mock.patch.object(
            jobs, "validate_prior_diagnostic", return_value=prior
        ):
            return jobs.build_wave(
                study_commit=study_commit,
                prerequisite_inputs=inputs,
                prior_diagnostic_path=prior_path,
                prior_diagnostic_sha256=prior["receipt"]["sha256"],
                results_root=Path("/fetched/results"),
                results_commit="b" * 40,
            )

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

    def test_queue_candidate_is_exact_and_directly_consumable_without_mutation(self) -> None:
        wave = self._built_wave()
        current_queue = {
            "schema_version": jobs.QUEUE_MANIFEST_SCHEMA,
            "namespace": jobs.NAMESPACE,
            "shutdown": False,
            "jobs": [],
        }
        original_wave = copy.deepcopy(wave)
        original_queue = copy.deepcopy(current_queue)

        with mock.patch.object(
            jobs,
            "_authenticate_control_state",
            return_value=copy.deepcopy(current_queue),
        ) as authenticate_control, mock.patch.object(
            jobs,
            "_authenticate_results_inputs",
            return_value=(
                copy.deepcopy(wave["replay_prerequisite_receipts"]),
                copy.deepcopy(wave["prior_diagnostic"]),
            ),
        ) as authenticate_results:
            candidate = jobs.compose_queue_candidate(
                wave=wave,
                expected_study_commit="c" * 40,
                results_root=Path("/fetched/results"),
                control_root=Path("/control"),
                control_commit="c" * 40,
                queue_path=Path("/control") / jobs.CONTROL_QUEUE_RELATIVE,
            )

        self.assertEqual(set(candidate), jobs.QUEUE_MANIFEST_KEYS)
        self.assertEqual(candidate["schema_version"], "wmf-cluster-queue-v1")
        self.assertEqual(candidate["namespace"], cluster_queue.NAMESPACE)
        self.assertIs(candidate["shutdown"], False)
        self.assertEqual(len(candidate["jobs"]), 1)
        self.assertEqual(candidate["jobs"][0], wave["jobs"][0])
        normalized = cluster_queue.normalize_job(candidate["jobs"][0])
        self.assertEqual(normalized["job_id"], jobs.JOB_ID)
        self.assertEqual(normalized["source_commit"], "c" * 40)
        self.assertEqual(normalized["publish_log_tail_bytes"], 0)
        self.assertEqual(wave, original_wave)
        self.assertEqual(current_queue, original_queue)
        authenticate_control.assert_called_once_with(
            control_root=Path("/control"),
            control_commit="c" * 40,
            queue_path=Path("/control") / jobs.CONTROL_QUEUE_RELATIVE,
            implementation=wave["implementation"],
            terminal_runtime=mock.ANY,
            replay_runtime=mock.ANY,
        )
        authenticate_results.assert_called_once_with(
            results_root=Path("/fetched/results"),
            results_commit=wave["prior_diagnostic"]["results_commit"],
            prerequisites=wave["replay_prerequisite_receipts"],
            prior=wave["prior_diagnostic"],
            expected_prerequisites=mock.ANY,
            source_root=Path("/control"),
        )

    def test_queue_candidate_rejects_stale_nonempty_shutdown_and_malformed_inputs(self) -> None:
        wave = self._built_wave()
        empty = {
            "schema_version": jobs.QUEUE_MANIFEST_SCHEMA,
            "namespace": jobs.NAMESPACE,
            "shutdown": False,
            "jobs": [],
        }

        authenticated = copy.deepcopy(wave["replay_prerequisite_receipts"])
        with mock.patch.object(
            jobs, "_authenticate_control_state", return_value=copy.deepcopy(empty)
        ), mock.patch.object(
            jobs,
            "_authenticate_results_inputs",
            return_value=(authenticated, copy.deepcopy(wave["prior_diagnostic"])),
        ):
            with self.assertRaisesRegex(jobs.CameraCropTerminalError, "stale"):
                jobs.compose_queue_candidate(
                    wave=wave,
                    expected_study_commit="d" * 40,
                    results_root=Path("/fetched/results"),
                    control_root=Path("/control"),
                    control_commit="d" * 40,
                    queue_path=Path("/control") / jobs.CONTROL_QUEUE_RELATIVE,
                )

            nonempty = copy.deepcopy(empty)
            nonempty["jobs"] = [wave["jobs"][0]]
            with mock.patch.object(
                jobs, "_authenticate_control_state", return_value=nonempty
            ), self.assertRaisesRegex(jobs.CameraCropTerminalError, "exactly empty"):
                    jobs.compose_queue_candidate(
                        wave=wave,
                        expected_study_commit="c" * 40,
                        results_root=Path("/fetched/results"),
                        control_root=Path("/control"),
                        control_commit="c" * 40,
                        queue_path=Path("/control") / jobs.CONTROL_QUEUE_RELATIVE,
                    )

            shutdown = copy.deepcopy(empty)
            shutdown["shutdown"] = True
            with mock.patch.object(
                jobs, "_authenticate_control_state", return_value=shutdown
            ), self.assertRaisesRegex(jobs.CameraCropTerminalError, "shutdown"):
                    jobs.compose_queue_candidate(
                        wave=wave,
                        expected_study_commit="c" * 40,
                        results_root=Path("/fetched/results"),
                        control_root=Path("/control"),
                        control_commit="c" * 40,
                        queue_path=Path("/control") / jobs.CONTROL_QUEUE_RELATIVE,
                    )

            malformed = copy.deepcopy(wave)
            malformed["descriptor_dispatched"] = True
            with self.assertRaisesRegex(jobs.CameraCropTerminalError, "authority boundary"):
                jobs.compose_queue_candidate(
                    wave=malformed,
                    expected_study_commit="c" * 40,
                    results_root=Path("/fetched/results"),
                    control_root=Path("/control"),
                    control_commit="c" * 40,
                    queue_path=Path("/control") / jobs.CONTROL_QUEUE_RELATIVE,
                )

            malformed = copy.deepcopy(wave)
            malformed["jobs"][0]["publish_log_tail_bytes"] = 1
            with self.assertRaisesRegex(jobs.CameraCropTerminalError, "descriptor changed"):
                jobs.compose_queue_candidate(
                    wave=malformed,
                    expected_study_commit="c" * 40,
                    results_root=Path("/fetched/results"),
                    control_root=Path("/control"),
                    control_commit="c" * 40,
                    queue_path=Path("/control") / jobs.CONTROL_QUEUE_RELATIVE,
                )

            malformed_queue = dict(empty, observed_control_commit="e" * 40)
            with mock.patch.object(
                jobs, "_authenticate_control_state", return_value=malformed_queue
            ), self.assertRaisesRegex(jobs.CameraCropTerminalError, "fields changed"):
                    jobs.compose_queue_candidate(
                        wave=wave,
                        expected_study_commit="c" * 40,
                        results_root=Path("/fetched/results"),
                        control_root=Path("/control"),
                        control_commit="c" * 40,
                        queue_path=Path("/control") / jobs.CONTROL_QUEUE_RELATIVE,
                    )

            for field, replacement in (
                ("sha256", "f" * 64),
                ("path", "/different/source.py"),
            ):
                malformed = copy.deepcopy(wave)
                name = next(iter(malformed["implementation"]))
                malformed["implementation"][name][field] = replacement
                with self.subTest(implementation_field=field), self.assertRaisesRegex(
                    jobs.CameraCropTerminalError, "implementation bytes"
                ):
                    jobs.compose_queue_candidate(
                        wave=malformed,
                        expected_study_commit="c" * 40,
                        results_root=Path("/fetched/results"),
                        control_root=Path("/control"),
                        control_commit="c" * 40,
                        queue_path=Path("/control") / jobs.CONTROL_QUEUE_RELATIVE,
                    )

            for field, replacement in (
                ("sha256", "e" * 64),
                ("path", "/fetched/different-prerequisite.json"),
            ):
                malformed = copy.deepcopy(wave)
                name = next(iter(malformed["replay_prerequisite_receipts"]))
                malformed["replay_prerequisite_receipts"][name][field] = replacement
                with self.subTest(prerequisite_field=field), self.assertRaisesRegex(
                    jobs.CameraCropTerminalError, "prerequisite identities"
                ):
                    jobs.compose_queue_candidate(
                        wave=malformed,
                        expected_study_commit="c" * 40,
                        results_root=Path("/fetched/results"),
                        control_root=Path("/control"),
                        control_commit="c" * 40,
                        queue_path=Path("/control") / jobs.CONTROL_QUEUE_RELATIVE,
                    )

            prior_mutations = (
                ("sha256", "d" * 64),
                (
                    "path",
                    str(
                        Path("/different/results")
                        / jobs.PRIOR_DIAGNOSTIC_RESULTS_RELATIVE
                    ),
                ),
                ("authenticated_source_bytes", 999999),
            )
            for field, replacement in prior_mutations:
                malformed = copy.deepcopy(wave)
                if field == "authenticated_source_bytes":
                    name = next(iter(
                        malformed["prior_diagnostic"]["authenticated_source_blobs"]
                    ))
                    malformed["prior_diagnostic"]["authenticated_source_blobs"][name][
                        "bytes"
                    ] = replacement
                else:
                    malformed["prior_diagnostic"]["receipt"][field] = replacement
                with self.subTest(prior_field=field), self.assertRaisesRegex(
                    jobs.CameraCropTerminalError, "diagnostic identity"
                ):
                    jobs.compose_queue_candidate(
                        wave=malformed,
                        expected_study_commit="c" * 40,
                        results_root=Path("/fetched/results"),
                        control_root=Path("/control"),
                        control_commit="c" * 40,
                        queue_path=Path("/control") / jobs.CONTROL_QUEUE_RELATIVE,
                    )

    def test_control_checkout_authentication_is_exact_and_index_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "control"
            commit, queue_path, implementation, expected = self._control_fixture(root)
            index_path = root / ".git/index"
            index_before = (
                index_path.read_bytes(),
                index_path.stat().st_mtime_ns,
                index_path.stat().st_ino,
            )
            queue_before = (
                queue_path.read_bytes(),
                queue_path.stat().st_mtime_ns,
                queue_path.stat().st_ino,
            )
            real_run = subprocess.run
            observed_git_calls = []

            def guarded_run(*args, **kwargs):
                observed_git_calls.append(tuple(args[0]))
                self.assertEqual(kwargs["env"]["GIT_OPTIONAL_LOCKS"], "0")
                self.assertEqual(kwargs["env"]["GIT_NO_LAZY_FETCH"], "1")
                for name in (
                    "GIT_DIR", "GIT_WORK_TREE",
                    "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
                    "GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0",
                ):
                    self.assertNotIn(name, kwargs["env"])
                self.assertRegex(
                    kwargs["env"]["GIT_INDEX_FILE"], r"^/proc/self/fd/[0-9]+$"
                )
                self.assertRegex(
                    kwargs["env"]["GIT_COMMON_DIR"], r"^/proc/self/fd/[0-9]+$"
                )
                for name in (
                    "PATH", "HOME", "XDG_CONFIG_HOME", "LD_PRELOAD",
                    "LD_LIBRARY_PATH", "PYTHONPATH",
                ):
                    self.assertNotIn(name, kwargs["env"])
                self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
                self.assertTrue(args[0][0].startswith("/proc/self/fd/"))
                self.assertTrue(any(
                    item.startswith("--git-dir=/proc/self/fd/") for item in args[0]
                ))
                self.assertTrue(any(
                    item.startswith("--work-tree=/proc/self/fd/") for item in args[0]
                ))
                self.assertIn("core.fsmonitor=false", args[0])
                self.assertIn("core.hooksPath=/dev/null", args[0])
                return real_run(*args, **kwargs)

            with mock.patch.object(
                jobs, "_source_paths",
                side_effect=lambda source_root: {"impl": Path(source_root) / "impl.py"},
            ), mock.patch.object(
                jobs.subprocess, "run", side_effect=guarded_run
            ), mock.patch.dict(jobs.os.environ, {
                "GIT_DIR": str(root / "poisoned-git-dir"),
                "GIT_WORK_TREE": str(root / "poisoned-work-tree"),
                "GIT_INDEX_FILE": str(root / "poisoned-index"),
                "GIT_OBJECT_DIRECTORY": str(root / "poisoned-objects"),
                "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(root / "poisoned-alternates"),
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "core.worktree",
                "GIT_CONFIG_VALUE_0": str(root / "poisoned-config-work-tree"),
            }, clear=False):
                authenticated = jobs._authenticate_control_state(
                    control_root=root,
                    control_commit=commit,
                    queue_path=queue_path,
                    implementation=implementation,
                )
            index_after = (
                index_path.read_bytes(),
                index_path.stat().st_mtime_ns,
                index_path.stat().st_ino,
            )
            self.assertEqual(authenticated, expected)
            self.assertEqual(index_after, index_before)
            self.assertEqual(
                (queue_path.read_bytes(), queue_path.stat().st_mtime_ns,
                 queue_path.stat().st_ino),
                queue_before,
            )
            self.assertGreaterEqual(len(observed_git_calls), 7)
            changed_implementation = copy.deepcopy(implementation)
            changed_implementation["impl"]["sha256"] = "f" * 64
            with mock.patch.object(
                jobs, "_source_paths",
                side_effect=lambda source_root: {"impl": Path(source_root) / "impl.py"},
            ), self.assertRaisesRegex(
                jobs.CameraCropTerminalError, "differs from the control commit"
            ):
                jobs._authenticate_control_state(
                    control_root=root,
                    control_commit=commit,
                    queue_path=queue_path,
                    implementation=changed_implementation,
                )

            queue_path.chmod(queue_path.stat().st_mode | stat.S_IXUSR)
            with mock.patch.object(
                jobs, "_source_paths",
                side_effect=lambda source_root: {"impl": Path(source_root) / "impl.py"},
            ), self.assertRaisesRegex(
                jobs.CameraCropTerminalError, "executable mode"
            ):
                jobs._authenticate_control_state(
                    control_root=root,
                    control_commit=commit,
                    queue_path=queue_path,
                    implementation=implementation,
                )

    def test_git_boundary_neutralizes_hostile_helpers_and_ambient_process_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "control"
            commit, queue_path, implementation, expected = self._control_fixture(root)
            marker = base / "helper-ran"
            helper = base / "hostile-fsmonitor"
            helper.write_text(
                "#!/bin/sh\n/usr/bin/touch " + str(marker) + "\nexit 0\n",
                encoding="utf-8",
            )
            helper.chmod(0o755)
            (root / ".gitattributes").write_text(
                "impl.py filter=evil\n", encoding="utf-8"
            )
            commit = self._commit_all(root, "hostile clean-filter attribute")
            self._git(root, "config", "core.fsmonitor", str(helper))
            self._git(root, "config", "filter.evil.clean", str(helper))
            self._git(root, "config", "filter.evil.required", "true")
            hostile_bin = base / "bin"
            hostile_bin.mkdir()
            hostile_git_marker = base / "path-git-ran"
            hostile_git = hostile_bin / "git"
            hostile_git.write_text(
                "#!/bin/sh\n/usr/bin/touch " + str(hostile_git_marker) + "\nexit 99\n",
                encoding="utf-8",
            )
            hostile_git.chmod(0o755)
            hostile_home = base / "home"
            hostile_home.mkdir()
            (hostile_home / ".gitconfig").write_text(
                "[core]\n\tfsmonitor = " + str(helper) + "\n",
                encoding="utf-8",
            )
            hostile_xdg = base / "xdg"
            (hostile_xdg / "git").mkdir(parents=True)
            (hostile_xdg / "git/config").write_text(
                "[core]\n\tfsmonitor = " + str(helper) + "\n",
                encoding="utf-8",
            )
            index = root / ".git/index"
            before = (index.read_bytes(), index.stat().st_mtime_ns)
            source_paths = lambda source_root: {
                "impl": Path(source_root) / "impl.py"
            }
            with mock.patch.object(
                jobs, "_source_paths", side_effect=source_paths
            ), mock.patch.dict(jobs.os.environ, {
                "PATH": str(hostile_bin),
                "HOME": str(hostile_home),
                "XDG_CONFIG_HOME": str(hostile_xdg),
                "LD_PRELOAD": str(base / "hostile-loader.so"),
                "LD_LIBRARY_PATH": str(base),
                "PYTHONPATH": str(base / "hostile-python"),
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "core.fsmonitor",
                "GIT_CONFIG_VALUE_0": str(helper),
            }, clear=False):
                authenticated = jobs._authenticate_control_state(
                    control_root=root,
                    control_commit=commit,
                    queue_path=queue_path,
                    implementation=implementation,
                )
            self.assertEqual(authenticated, expected)
            self.assertFalse(marker.exists())
            self.assertFalse(hostile_git_marker.exists())
            self.assertEqual((index.read_bytes(), index.stat().st_mtime_ns), before)

    def test_parsed_runtime_contracts_must_equal_held_control_git_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "control"
            terminal_value = {"contract": {"count": 1, "enabled": False}}
            replay_value = {"prerequisite_receipts": {"gate": ["sha", 7]}}
            for relative, value in (
                (jobs.TERMINAL_CONTRACT_RELATIVE, terminal_value),
                (jobs.REPLAY_CONTRACT_RELATIVE, replay_value),
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(jobs.queue.canonical_bytes(value))
            commit = self._init_git_repo(root)
            with jobs._held_git_checkout(root, "runtime contract checkout") as checkout:
                jobs._authenticate_runtime_contract_values_in_checkout(
                    checkout=checkout,
                    control_commit=commit,
                    terminal_runtime=terminal_value,
                    replay_runtime=replay_value,
                )
                forged_terminal = copy.deepcopy(terminal_value)
                forged_terminal["contract"]["count"] = 1.0
                with self.assertRaisesRegex(
                    jobs.CameraCropTerminalError, "semantics differ"
                ):
                    jobs._authenticate_runtime_contract_values_in_checkout(
                        checkout=checkout,
                        control_commit=commit,
                        terminal_runtime=forged_terminal,
                        replay_runtime=replay_value,
                    )
                forged_replay = copy.deepcopy(replay_value)
                forged_replay["invented_gate"] = True
                with self.assertRaisesRegex(
                    jobs.CameraCropTerminalError, "semantics differ"
                ):
                    jobs._authenticate_runtime_contract_values_in_checkout(
                        checkout=checkout,
                        control_commit=commit,
                        terminal_runtime=terminal_value,
                        replay_runtime=forged_replay,
                    )

    def test_git_boundary_supports_committed_linked_worktree_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            main = base / "main"
            main.mkdir()
            (main / "tracked.txt").write_text("linked worktree payload\n", encoding="utf-8")
            commit = self._init_git_repo(main)
            linked = base / "linked"
            self._git(main, "worktree", "add", "-q", str(linked), "HEAD")
            pointer = linked / ".git"
            pointer_before = (
                pointer.read_bytes(), pointer.stat().st_mtime_ns, pointer.stat().st_ino,
            )
            with jobs._held_git_checkout(linked, "linked test checkout") as checkout:
                admin_files = [
                    checkout.admin_path / "commondir",
                    checkout.admin_path / "gitdir",
                    checkout.admin_path / "index",
                ]
                before = {
                    str(path): (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_ino)
                    for path in admin_files
                }
                jobs._require_clean_checkout(checkout, commit, "linked test checkout")
                self.assertEqual(
                    jobs._git_blob(
                        checkout, commit, Path("tracked.txt"), "linked fixture"
                    ),
                    b"linked worktree payload\n",
                )
            self.assertEqual(
                (pointer.read_bytes(), pointer.stat().st_mtime_ns, pointer.stat().st_ino),
                pointer_before,
            )
            self.assertEqual(
                {
                    str(path): (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_ino)
                    for path in admin_files
                },
                before,
            )

    def test_git_boundary_uses_held_root_and_linked_admin_during_swaps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            original = base / "original"
            original.mkdir()
            (original / "value.txt").write_text("original\n", encoding="utf-8")
            original_commit = self._init_git_repo(original)
            hostile = base / "hostile"
            hostile.mkdir()
            (hostile / "value.txt").write_text("hostile\n", encoding="utf-8")
            hostile_commit = self._init_git_repo(hostile)
            self.assertNotEqual(original_commit, hostile_commit)
            checkout = jobs._open_git_checkout(original, "root-swap checkout")
            parked = base / "parked-original"
            real_run = subprocess.run

            def root_swap_run(*args, **kwargs):
                original.rename(parked)
                hostile.rename(original)
                try:
                    return real_run(*args, **kwargs)
                finally:
                    original.rename(hostile)
                    parked.rename(original)

            try:
                with mock.patch.object(
                    jobs.subprocess, "run", side_effect=root_swap_run
                ):
                    observed = jobs._run_git(checkout, "rev-parse", "HEAD")
                self.assertEqual(observed.strip(), original_commit)
            finally:
                checkout.close()

            main = base / "main"
            main.mkdir()
            (main / "tracked.txt").write_text("main\n", encoding="utf-8")
            main_commit = self._init_git_repo(main)
            linked = base / "linked"
            other = base / "other"
            self._git(main, "worktree", "add", "-q", str(linked), "HEAD")
            self._git(main, "worktree", "add", "-q", str(other), "HEAD")
            linked_checkout = jobs._open_git_checkout(linked, "admin-swap checkout")
            other_pointer = (other / ".git").read_text(encoding="utf-8")
            other_admin = Path(other_pointer.removeprefix("gitdir: ").strip())
            linked_admin = linked_checkout.admin_path
            parked_admin = linked_admin.with_name(linked_admin.name + "-parked")

            def admin_swap_run(*args, **kwargs):
                linked_admin.rename(parked_admin)
                other_admin.rename(linked_admin)
                try:
                    return real_run(*args, **kwargs)
                finally:
                    linked_admin.rename(other_admin)
                    parked_admin.rename(linked_admin)

            try:
                with mock.patch.object(
                    jobs.subprocess, "run", side_effect=admin_swap_run
                ):
                    observed = jobs._run_git(linked_checkout, "rev-parse", "HEAD")
                self.assertEqual(observed.strip(), main_commit)
            finally:
                linked_checkout.close()

            pointer = linked / ".git"
            saved_pointer = linked / ".git-authenticated"
            hostile_pointer = linked / ".git-hostile"
            hostile_pointer.write_text(
                "gitdir: " + str(other_admin) + "\n", encoding="utf-8"
            )
            pointer_checkout = jobs._open_git_checkout(linked, "pointer-swap checkout")

            def pointer_swap_run(*args, **kwargs):
                pointer.rename(saved_pointer)
                hostile_pointer.rename(pointer)
                try:
                    return real_run(*args, **kwargs)
                finally:
                    pointer.rename(hostile_pointer)
                    saved_pointer.rename(pointer)

            try:
                with mock.patch.object(
                    jobs.subprocess, "run", side_effect=pointer_swap_run
                ), self.assertRaisesRegex(
                    jobs.CameraCropTerminalError, "administrative (?:path|pointer)"
                ):
                    jobs._run_git(pointer_checkout, "rev-parse", "HEAD")
            finally:
                pointer_checkout.close()

    def test_reserved_queue_read_and_recheck_stay_on_held_checkout_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            original = base / "control"
            commit, queue_path, implementation, clean_queue = self._control_fixture(
                original
            )
            alternate = base / "alternate-clean"
            self._git(original, "worktree", "add", "-q", str(alternate), "HEAD")
            alternate_queue = alternate / jobs.CONTROL_QUEUE_RELATIVE
            self.assertEqual(
                alternate_queue.read_bytes(), jobs.queue.canonical_bytes(clean_queue)
            )

            dirty_queue = copy.deepcopy(clean_queue)
            dirty_queue["jobs"] = [{"held-dirty": True}]
            dirty_payload = jobs.queue.canonical_bytes(dirty_queue)
            queue_path.write_bytes(dirty_payload)
            checkout = jobs._open_git_checkout(original, "reserved-root-swap checkout")
            parked = base / "parked-control"
            real_read = jobs._read_checkout_regular_file_once
            observed_held_payload = []

            def swapped_reserved_read(held, relative, *args, **kwargs):
                original.rename(parked)
                alternate.rename(original)
                try:
                    result = real_read(held, relative, *args, **kwargs)
                    observed_held_payload.append(result[0])
                    jobs._require_checkout_regular_file_identity(
                        held,
                        relative,
                        result[2],
                        "reserved queue during named-root swap",
                    )
                    return result
                finally:
                    original.rename(alternate)
                    parked.rename(original)

            try:
                with mock.patch.object(
                    jobs,
                    "_source_paths",
                    side_effect=lambda root: {"impl": Path(root) / "impl.py"},
                ), mock.patch.object(
                    jobs,
                    "_read_checkout_regular_file_once",
                    side_effect=swapped_reserved_read,
                ), self.assertRaisesRegex(
                    jobs.CameraCropTerminalError,
                    "control queue bytes differ",
                ):
                    jobs._authenticate_control_state_in_checkout(
                        checkout=checkout,
                        control_commit=commit,
                        queue_path=queue_path,
                        implementation=implementation,
                    )
            finally:
                checkout.close()
            self.assertEqual(observed_held_payload, [dirty_payload])

    def test_control_checkout_rejects_stale_other_traversal_and_symlink_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "control"
            commit, queue_path, implementation, _expected = self._control_fixture(root)
            patches = mock.patch.object(
                jobs, "_source_paths",
                side_effect=lambda source_root: {"impl": Path(source_root) / "impl.py"},
            )
            with patches:
                with self.assertRaisesRegex(
                    jobs.CameraCropTerminalError, "exact tracked"
                ):
                    jobs._authenticate_control_state(
                        control_root=root,
                        control_commit=commit,
                        queue_path=root / "synthetic-empty.json",
                        implementation=implementation,
                    )
                nested_base = queue_path.parent
                with self.assertRaisesRegex(
                    jobs.CameraCropTerminalError, "Git administrative path|exact supplied"
                ):
                    jobs._authenticate_control_state(
                        control_root=nested_base,
                        control_commit=commit,
                        queue_path=nested_base / jobs.CONTROL_QUEUE_RELATIVE,
                        implementation=implementation,
                    )
                traversal = root / "queue-parent" / ".." / jobs.CONTROL_QUEUE_RELATIVE
                with self.assertRaisesRegex(
                    jobs.CameraCropTerminalError, "lexically normalized"
                ):
                    jobs._authenticate_control_state(
                        control_root=root,
                        control_commit=commit,
                        queue_path=traversal,
                        implementation=implementation,
                    )
                parent_link = base / "linked-parent"
                parent_link.symlink_to(base, target_is_directory=True)
                with self.assertRaisesRegex(
                    jobs.CameraCropTerminalError, "contains a symlink"
                ):
                    jobs._authenticate_control_state(
                        control_root=parent_link / "control",
                        control_commit=commit,
                        queue_path=parent_link / "control" / jobs.CONTROL_QUEUE_RELATIVE,
                        implementation=implementation,
                    )
                root_link = base / "control-link"
                root_link.symlink_to(root, target_is_directory=True)
                with self.assertRaisesRegex(
                    jobs.CameraCropTerminalError, "contains a symlink"
                ):
                    jobs._authenticate_control_state(
                        control_root=root_link,
                        control_commit=commit,
                        queue_path=root_link / jobs.CONTROL_QUEUE_RELATIVE,
                        implementation=implementation,
                    )

                (root / "stale-marker").write_text("new commit\n", encoding="utf-8")
                new_commit = self._commit_all(root, "advance")
                self.assertNotEqual(new_commit, commit)
                with self.assertRaisesRegex(
                    jobs.CameraCropTerminalError, "exact supplied commit"
                ):
                    jobs._authenticate_control_state(
                        control_root=root,
                        control_commit=commit,
                        queue_path=queue_path,
                        implementation=implementation,
                    )

                target = root / "queue-target.json"
                target.write_bytes(jobs.queue.canonical_bytes({
                    "schema_version": jobs.QUEUE_MANIFEST_SCHEMA,
                    "namespace": jobs.NAMESPACE,
                    "shutdown": False,
                    "jobs": [],
                }))
                queue_path.unlink()
                queue_path.symlink_to(target)
                symlink_commit = self._commit_all(root, "track final symlink")
                with self.assertRaisesRegex(
                    jobs.CameraCropTerminalError, "symlink"
                ):
                    jobs._authenticate_control_state(
                        control_root=root,
                        control_commit=symlink_commit,
                        queue_path=queue_path,
                        implementation=implementation,
                    )

    def test_real_tracked_nonempty_control_queue_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "control"
            commit, queue_path, implementation, _expected = self._control_fixture(
                root, jobs_value=[{"synthetic": "occupied"}]
            )
            with mock.patch.object(
                jobs, "_source_paths",
                side_effect=lambda source_root: {"impl": Path(source_root) / "impl.py"},
            ):
                authenticated = jobs._authenticate_control_state(
                    control_root=root,
                    control_commit=commit,
                    queue_path=queue_path,
                    implementation=implementation,
                )
            with self.assertRaisesRegex(
                jobs.CameraCropTerminalError, "exactly empty"
            ):
                jobs._validate_empty_queue_template(authenticated)

    def test_single_read_rejects_fifo_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fifo = Path(temporary).resolve() / "receipt.json"
            jobs.os.mkfifo(fifo)
            with self.assertRaisesRegex(
                jobs.CameraCropTerminalError, "regular file"
            ):
                jobs._read_regular_file_once(
                    fifo, "FIFO receipt", maximum_bytes=1024
                )

    def test_queue_builder_rejects_fifo_and_path_swap_during_gate_read(self) -> None:
        def invoke(fixture, read_patch):
            source_paths = lambda root: {"impl": Path(root) / "impl.py"}
            with mock.patch.object(
                jobs, "REPOSITORY_ROOT", fixture["control_root"]
            ), mock.patch.object(
                jobs, "_source_paths", side_effect=source_paths
            ), mock.patch.object(
                jobs, "PRIOR_SOURCE_PINS", fixture["source_pin"]
            ), mock.patch.object(
                jobs, "_runtime_contract", return_value={}
            ), mock.patch.object(
                jobs.diagnostic, "_runtime_contract",
                return_value={"prerequisite_receipts": fixture["expected"]},
            ), read_patch:
                return jobs.build_queue_candidate(
                    study_commit=fixture["control_commit"],
                    prerequisite_inputs=fixture["inputs"],
                    prior_diagnostic_path=fixture["prior_path"],
                    prior_diagnostic_sha256=fixture["prior_sha256"],
                    results_root=fixture["results_root"],
                    results_commit=fixture["results_commit"],
                    control_root=fixture["control_root"],
                    control_commit=fixture["control_commit"],
                    queue_path=fixture["queue_path"],
                )

        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._queue_auth_fixture(Path(temporary).resolve())
            first_name = sorted(fixture["inputs"])[0]
            target = fixture["inputs"][first_name][0]
            target_relative = jobs.PREREQUISITE_RESULTS_RELATIVES[first_name]
            real_read = jobs._read_checkout_regular_file_once
            changed = False

            def fifo_swap(checkout, relative, *args, **kwargs):
                nonlocal changed
                if Path(relative) == target_relative and not changed:
                    changed = True
                    target.unlink()
                    jobs.os.mkfifo(target)
                return real_read(checkout, relative, *args, **kwargs)

            with self.assertRaisesRegex(
                jobs.CameraCropTerminalError, "regular file"
            ):
                invoke(
                    fixture,
                    mock.patch.object(
                        jobs, "_read_checkout_regular_file_once",
                        side_effect=fifo_swap,
                    ),
                )

        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._queue_auth_fixture(Path(temporary).resolve())
            first_name = sorted(fixture["inputs"])[0]
            target = fixture["inputs"][first_name][0]
            target_inode = target.stat().st_ino
            backup = target.with_name(target.name + ".held-original")
            real_os_read = jobs.os.read
            changed = False

            def swapping_read(descriptor, count):
                nonlocal changed
                block = real_os_read(descriptor, count)
                if (
                    block
                    and not changed
                    and jobs.os.fstat(descriptor).st_ino == target_inode
                ):
                    changed = True
                    target.rename(backup)
                    target.write_bytes(b"{}\n")
                return block

            with self.assertRaisesRegex(
                jobs.CameraCropTerminalError, "changed while it was read|path changed"
            ):
                invoke(
                    fixture,
                    mock.patch.object(jobs.os, "read", side_effect=swapping_read),
                )

    def test_prior_source_blob_authentication_uses_real_git_without_index_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "source"
            payloads = {
                Path("source/a.py"): b"print('source a')\n",
                Path("source/b.json"): b'{"source":"b"}\n',
            }
            for relative, payload in payloads.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            commit = self._init_git_repo(root)
            pins = {
                relative: jobs.queue.sha256_bytes(payload)
                for relative, payload in payloads.items()
            }
            index_path = root / ".git/index"
            before = (index_path.read_bytes(), index_path.stat().st_mtime_ns)
            with mock.patch.object(jobs, "PRIOR_SOURCE_PINS", pins):
                authenticated = jobs._validate_prior_source_commit(root, commit)
            self.assertEqual(
                authenticated,
                {
                    str(relative): {
                        "sha256": pins[relative], "bytes": len(payload),
                    }
                    for relative, payload in payloads.items()
                },
            )
            self.assertEqual(
                (index_path.read_bytes(), index_path.stat().st_mtime_ns), before
            )
            tree_oid = self._git(root, "rev-parse", f"{commit}^{{tree}}")
            self.assertRegex(tree_oid, r"^[0-9a-f]{40}$")
            with mock.patch.object(
                jobs, "PRIOR_SOURCE_PINS", pins
            ), mock.patch.object(
                jobs, "_git_blob",
                side_effect=AssertionError("blob read preceded commit-object gate"),
            ) as blob_read, self.assertRaisesRegex(
                jobs.CameraCropTerminalError, "not a commit object"
            ):
                jobs._validate_prior_source_commit(root, tree_oid)
            self.assertEqual(blob_read.call_count, 0)

    def test_control_commit_must_equal_intended_study_commit(self) -> None:
        wave = self._built_wave()
        with self.assertRaisesRegex(
            jobs.CameraCropTerminalError, "control commit differs"
        ):
            jobs.compose_queue_candidate(
                wave=wave,
                expected_study_commit="c" * 40,
                results_root=Path("/results"),
                control_root=Path("/control"),
                control_commit="d" * 40,
                queue_path=Path("/control") / jobs.CONTROL_QUEUE_RELATIVE,
            )

    def test_queue_cli_relationships_fail_before_any_builder_or_read(self) -> None:
        control = Path("/control")
        results = Path("/results")
        base = [
            "build-wave",
            "--study-commit", "a" * 40,
            "--results-root", str(results),
            "--results-commit", "b" * 40,
            "--prior-diagnostic-receipt",
            str(results / jobs.PRIOR_DIAGNOSTIC_RESULTS_RELATIVE),
            "--prior-diagnostic-receipt-sha256", "c" * 64,
            "--queue-template", str(control / jobs.CONTROL_QUEUE_RELATIVE),
            "--control-root", str(control),
            "--control-commit", "a" * 40,
        ]
        for name in sorted(jobs.PREREQUISITE_RESULTS_RELATIVES):
            flag = name.replace("_", "-")
            base.extend([
                f"--{flag}",
                str(results / jobs.PREREQUISITE_RESULTS_RELATIVES[name]),
                f"--{flag}-sha256", "d" * 64,
            ])

        def replaced(arguments, flag, value):
            result = list(arguments)
            result[result.index(flag) + 1] = value
            return result

        def removed(arguments, flag):
            result = list(arguments)
            index = result.index(flag)
            del result[index:index + 2]
            return result

        first_name = sorted(jobs.PREREQUISITE_RESULTS_RELATIVES)[0]
        first_flag = "--" + first_name.replace("_", "-")
        cases = [
            base + ["--output", "/tmp/forbidden.json"],
            removed(base, "--control-root"),
            removed(base, "--control-commit"),
            replaced(base, "--control-commit", "e" * 40),
            replaced(base, "--queue-template", "/control/other.json"),
            replaced(base, "--prior-diagnostic-receipt", "/results/other.json"),
            replaced(base, "--prior-diagnostic-receipt-sha256", "not-a-digest"),
            replaced(base, first_flag, "/results/other-prerequisite.json"),
            removed(base, "--queue-template"),
        ]
        sentinels = [
            mock.patch.object(
                jobs, "_builder_inputs", side_effect=AssertionError("inputs read")
            ),
            mock.patch.object(
                jobs, "build_wave", side_effect=AssertionError("legacy builder used")
            ),
            mock.patch.object(
                jobs, "build_queue_candidate",
                side_effect=AssertionError("queue builder used"),
            ),
            mock.patch.object(
                jobs, "_read_regular_file_once",
                side_effect=AssertionError("filesystem read"),
            ),
            mock.patch.object(
                jobs, "_read_checkout_regular_file_once",
                side_effect=AssertionError("held-checkout filesystem read"),
            ),
            mock.patch.object(
                jobs, "_open_git_checkout", side_effect=AssertionError("Git opened")
            ),
            mock.patch.object(
                jobs.diagnostic, "validate_prerequisites",
                side_effect=AssertionError("legacy prerequisite read"),
            ),
            mock.patch.object(
                jobs, "validate_prior_diagnostic",
                side_effect=AssertionError("legacy prior read"),
            ),
        ]
        entered = [patcher.start() for patcher in sentinels]
        try:
            for arguments in cases:
                with self.subTest(arguments=arguments), self.assertRaises(Exception):
                    jobs.main(arguments)
            for sentinel in entered:
                self.assertEqual(sentinel.call_count, 0)
        finally:
            for patcher in reversed(sentinels):
                patcher.stop()

    def test_results_checkout_rehashes_and_deep_validates_all_six_exact_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "results"
            values = self._synthetic_prerequisite_values()
            payloads = {}
            for name, value in values.items():
                path = root / jobs.PREREQUISITE_RESULTS_RELATIVES[name]
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = jobs.queue.canonical_bytes(value)
                path.write_bytes(payload)
                payloads[name] = payload
            prior_path = root / jobs.PRIOR_DIAGNOSTIC_RESULTS_RELATIVE
            prior_path.parent.mkdir(parents=True, exist_ok=True)
            prior_payload = jobs.queue.canonical_bytes(self._prior_receipt())
            prior_path.write_bytes(prior_payload)
            commit = self._init_git_repo(root)

            def identity(path: Path) -> dict:
                payload = path.read_bytes()
                return {
                    "path": str(path),
                    "bytes": len(payload),
                    "sha256": jobs.queue.sha256_bytes(payload),
                }

            prerequisites = {
                name: identity(root / relative)
                for name, relative in jobs.PREREQUISITE_RESULTS_RELATIVES.items()
            }
            expected = {
                name: {
                    "bytes": len(payloads[name]),
                    "sha256": jobs.queue.sha256_bytes(payloads[name]),
                    "schema_version": values[name]["schema_version"],
                    "model_id": values[name]["model_id"],
                    "status": values[name]["status"],
                    "decision": values[name]["decision"],
                }
                for name in values
            }
            source_blobs = {
                str(path): {"sha256": digest, "bytes": index + 1}
                for index, (path, digest) in enumerate(jobs.PRIOR_SOURCE_PINS.items())
            }
            prior = {
                "receipt": identity(prior_path),
                "results_commit": commit,
                "study_commit": "a" * 40,
                "authenticated_source_blobs": source_blobs,
            }
            index_path = root / ".git/index"
            index_before = (index_path.read_bytes(), index_path.stat().st_mtime_ns)
            real_deep_validator = jobs.diagnostic._validate_prerequisite_value
            with mock.patch.object(
                jobs,
                "_validate_prior_source_commit_in_checkout",
                return_value=copy.deepcopy(source_blobs),
            ), mock.patch.object(
                jobs.diagnostic,
                "_validate_prerequisite_value",
                wraps=real_deep_validator,
            ) as deep_validator:
                authenticated, authenticated_prior = jobs._authenticate_results_inputs(
                    results_root=root,
                    results_commit=commit,
                    prerequisites=prerequisites,
                    prior=prior,
                    expected_prerequisites=expected,
                    source_root=root,
                )
            self.assertEqual(authenticated, prerequisites)
            self.assertEqual(authenticated_prior, prior)
            self.assertEqual(deep_validator.call_count, 5)
            self.assertEqual(
                (index_path.read_bytes(), index_path.stat().st_mtime_ns),
                index_before,
            )

            changed_hash = copy.deepcopy(prerequisites)
            changed_hash["n3_generation"]["sha256"] = "f" * 64
            with mock.patch.object(
                jobs, "_validate_prior_source_commit_in_checkout", return_value=source_blobs
            ), self.assertRaisesRegex(
                jobs.CameraCropTerminalError, "supplied SHA-256 changed"
            ):
                jobs._authenticate_results_inputs(
                    results_root=root,
                    results_commit=commit,
                    prerequisites=changed_hash,
                    prior=prior,
                    expected_prerequisites=expected,
                    source_root=root,
                )

            changed_path = copy.deepcopy(prerequisites)
            changed_path["n3_generation"]["path"] = str(
                root / jobs.PREREQUISITE_RESULTS_RELATIVES["d1_generation"]
            )
            with mock.patch.object(
                jobs, "_validate_prior_source_commit_in_checkout", return_value=source_blobs
            ), self.assertRaisesRegex(
                jobs.CameraCropTerminalError, "exact results artifact"
            ):
                jobs._authenticate_results_inputs(
                    results_root=root,
                    results_commit=commit,
                    prerequisites=changed_path,
                    prior=prior,
                    expected_prerequisites=expected,
                    source_root=root,
                )

            changed_prior_hash = copy.deepcopy(prior)
            changed_prior_hash["receipt"]["sha256"] = "e" * 64
            with mock.patch.object(
                jobs, "_validate_prior_source_commit_in_checkout", return_value=source_blobs
            ), self.assertRaisesRegex(
                jobs.CameraCropTerminalError, "diagnostic identity changed"
            ):
                jobs._authenticate_results_inputs(
                    results_root=root,
                    results_commit=commit,
                    prerequisites=prerequisites,
                    prior=changed_prior_hash,
                    expected_prerequisites=expected,
                    source_root=root,
                )

            changed_prior_path = copy.deepcopy(prior)
            changed_prior_path["receipt"]["path"] = str(
                root / jobs.PREREQUISITE_RESULTS_RELATIVES["d1_qualification"]
            )
            with mock.patch.object(
                jobs, "_validate_prior_source_commit_in_checkout", return_value=source_blobs
            ), self.assertRaisesRegex(
                jobs.CameraCropTerminalError, "exact results artifact"
            ):
                jobs._authenticate_results_inputs(
                    results_root=root,
                    results_commit=commit,
                    prerequisites=prerequisites,
                    prior=changed_prior_path,
                    expected_prerequisites=expected,
                    source_root=root,
                )

            bad_generation = copy.deepcopy(values["n3_generation"])
            bad_generation["qualified"] = False
            bad_payload = jobs.queue.canonical_bytes(bad_generation)
            generation_path = root / jobs.PREREQUISITE_RESULTS_RELATIVES["n3_generation"]
            generation_path.write_bytes(bad_payload)
            bad_commit = self._commit_all(root, "semantic prerequisite mutation")
            bad_expected = copy.deepcopy(expected)
            bad_expected["n3_generation"]["bytes"] = len(bad_payload)
            bad_expected["n3_generation"]["sha256"] = jobs.queue.sha256_bytes(bad_payload)
            bad_prerequisites = copy.deepcopy(prerequisites)
            bad_prerequisites["n3_generation"] = identity(generation_path)
            bad_prior = copy.deepcopy(prior)
            bad_prior["results_commit"] = bad_commit
            with mock.patch.object(
                jobs, "_validate_prior_source_commit_in_checkout", return_value=source_blobs
            ), self.assertRaisesRegex(
                jobs.diagnostic.CameraCropQueueError, "zero-policy probe"
            ):
                jobs._authenticate_results_inputs(
                    results_root=root,
                    results_commit=bad_commit,
                    prerequisites=bad_prerequisites,
                    prior=bad_prior,
                    expected_prerequisites=bad_expected,
                    source_root=root,
                )

            generation_path.write_bytes(payloads["n3_generation"])
            changed_receipt = self._prior_receipt()
            changed_receipt["status"] = "passed"
            changed_receipt = self._resign(changed_receipt)
            prior_path.write_bytes(jobs.queue.canonical_bytes(changed_receipt))
            receipt_commit = self._commit_all(root, "semantic receipt mutation")
            receipt_prerequisites = copy.deepcopy(prerequisites)
            receipt_prior = copy.deepcopy(prior)
            receipt_prior["receipt"] = identity(prior_path)
            receipt_prior["results_commit"] = receipt_commit
            with mock.patch.object(
                jobs, "_validate_prior_source_commit_in_checkout", return_value=source_blobs
            ), self.assertRaisesRegex(
                jobs.CameraCropTerminalError, "receipt changed: status"
            ):
                jobs._authenticate_results_inputs(
                    results_root=root,
                    results_commit=receipt_commit,
                    prerequisites=receipt_prerequisites,
                    prior=receipt_prior,
                    expected_prerequisites=expected,
                    source_root=root,
                )

    def test_cli_composes_through_real_control_and_results_checkouts_stdout_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            control_root = base / "control"
            control_commit, queue_path, implementation, _queue_value = (
                self._control_fixture(control_root)
            )
            results_root = base / "results"
            values = self._synthetic_prerequisite_values()
            payloads = {}
            for name, value in values.items():
                path = results_root / jobs.PREREQUISITE_RESULTS_RELATIVES[name]
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = jobs.queue.canonical_bytes(value)
                path.write_bytes(payload)
                payloads[name] = payload
            prior_path = results_root / jobs.PRIOR_DIAGNOSTIC_RESULTS_RELATIVE
            prior_path.parent.mkdir(parents=True, exist_ok=True)
            prior_value = self._prior_receipt()
            prior_value["study_commit"] = control_commit
            prior_value = self._resign(prior_value)
            prior_path.write_bytes(jobs.queue.canonical_bytes(prior_value))
            results_commit = self._init_git_repo(results_root)

            def identity(path: Path) -> dict:
                payload = path.read_bytes()
                return {
                    "path": str(path), "bytes": len(payload),
                    "sha256": jobs.queue.sha256_bytes(payload),
                }

            prerequisites = {
                name: identity(results_root / relative)
                for name, relative in jobs.PREREQUISITE_RESULTS_RELATIVES.items()
            }
            expected = {
                name: {
                    "bytes": len(payloads[name]),
                    "sha256": jobs.queue.sha256_bytes(payloads[name]),
                    "schema_version": values[name]["schema_version"],
                    "model_id": values[name]["model_id"],
                    "status": values[name]["status"],
                    "decision": values[name]["decision"],
                }
                for name in values
            }
            terminal_contract = control_root / jobs.TERMINAL_CONTRACT_RELATIVE
            terminal_contract.parent.mkdir(parents=True, exist_ok=True)
            terminal_contract.write_bytes(jobs.queue.canonical_bytes({}))
            replay_contract = control_root / jobs.REPLAY_CONTRACT_RELATIVE
            replay_contract.parent.mkdir(parents=True, exist_ok=True)
            replay_contract.write_bytes(jobs.queue.canonical_bytes({
                "prerequisite_receipts": expected,
            }))
            control_commit = self._commit_all(
                control_root, "add authenticated runtime contracts"
            )
            prior_value["study_commit"] = control_commit
            prior_value = self._resign(prior_value)
            prior_path.write_bytes(jobs.queue.canonical_bytes(prior_value))
            results_commit = self._commit_all(
                results_root, "bind prior receipt to control commit"
            )
            source_pin = {Path("impl.py"): implementation["impl"]["sha256"]}
            source_blobs = {
                "impl.py": {
                    "sha256": implementation["impl"]["sha256"],
                    "bytes": implementation["impl"]["bytes"],
                }
            }
            prior = {
                "receipt": identity(prior_path),
                "results_commit": results_commit,
                "study_commit": control_commit,
                "authenticated_source_blobs": source_blobs,
            }
            source_paths = lambda root: {"impl": Path(root) / "impl.py"}
            with mock.patch.object(jobs, "_source_paths", side_effect=source_paths):
                descriptor = jobs._job_descriptor(
                    study_commit=control_commit,
                    implementation=implementation,
                    prior=prior,
                )
            wave = {
                "schema_version": jobs.WAVE_SCHEMA,
                "namespace": jobs.NAMESPACE,
                "study_id": jobs.STUDY_ID,
                "source_commit": control_commit,
                "status": jobs.WAVE_STATUS,
                "jobs": [descriptor],
                "implementation": implementation,
                "replay_prerequisite_receipts": prerequisites,
                "prior_diagnostic": prior,
                "expected_outputs": {
                    "published_camera_crop_contracts": jobs.MODEL_CONTRACT_NAMES,
                    "success_receipt": jobs.SUCCESS_RECEIPT,
                    "publish_file_count": 3,
                },
                "science_counts": jobs.zero_science_counts(),
                "simulator_state_render_used": False,
                "whole_frame_identity": False,
                "safe_for_physical_alignment_input_only_after_runtime_success": True,
                "safe_to_release_confirmation": False,
                "confirmation_released": False,
                "descriptor_dispatched": False,
                "claim_boundary": jobs.WAVE_CLAIM_BOUNDARY,
            }
            control_index = control_root / ".git/index"
            results_index = results_root / ".git/index"
            before = {
                "control_index": (
                    control_index.read_bytes(), control_index.stat().st_mtime_ns,
                ),
                "results_index": (
                    results_index.read_bytes(), results_index.stat().st_mtime_ns,
                ),
                "queue": (queue_path.read_bytes(), queue_path.stat().st_mtime_ns),
            }
            arguments = [
                "build-wave",
                "--study-commit", control_commit,
                "--results-root", str(results_root),
                "--results-commit", results_commit,
                "--prior-diagnostic-receipt", str(prior_path),
                "--prior-diagnostic-receipt-sha256", prior["receipt"]["sha256"],
                "--queue-template", str(queue_path),
                "--control-root", str(control_root),
                "--control-commit", control_commit,
            ]
            for name in sorted(prerequisites):
                flag = name.replace("_", "-")
                arguments.extend([
                    f"--{flag}", prerequisites[name]["path"],
                    f"--{flag}-sha256", prerequisites[name]["sha256"],
                ])
            sink = SimpleNamespace(buffer=io.BytesIO())
            real_read = jobs._read_checkout_regular_file_once
            real_os_open = jobs.os.open
            real_os_read = jobs.os.read
            observed_reads = []
            single_read_paths = {
                results_root / relative
                for relative in jobs.PREREQUISITE_RESULTS_RELATIVES.values()
            } | {prior_path, queue_path}
            low_level_opens = {path: 0 for path in single_read_paths}
            expected_low_level_bytes = {
                path: path.stat().st_size for path in single_read_paths
            }
            low_level_read_chunks = {path: 0 for path in single_read_paths}
            low_level_read_bytes = {path: 0 for path in single_read_paths}

            def counting_read(checkout, relative, *args, **kwargs):
                observed_reads.append(checkout.root / Path(relative))
                return real_read(checkout, relative, *args, **kwargs)

            def counting_open(path, flags, mode=0o777, *, dir_fd=None):
                raw = jobs.os.fsdecode(path)
                if dir_fd is None:
                    candidate = Path(raw)
                    if not candidate.is_absolute():
                        candidate = Path(jobs.os.path.abspath(raw))
                else:
                    parent = Path(jobs.os.readlink(f"/proc/self/fd/{dir_fd}"))
                    candidate = parent / raw
                candidate = Path(jobs.os.path.normpath(str(candidate)))
                if candidate in low_level_opens:
                    low_level_opens[candidate] += 1
                return real_os_open(path, flags, mode, dir_fd=dir_fd)

            def counting_os_read(descriptor, count):
                try:
                    candidate = Path(
                        jobs.os.readlink(f"/proc/self/fd/{descriptor}")
                    )
                except OSError:
                    candidate = None
                payload = real_os_read(descriptor, count)
                if candidate in low_level_read_bytes and payload:
                    low_level_read_chunks[candidate] += 1
                    low_level_read_bytes[candidate] += len(payload)
                return payload

            with mock.patch.object(
                jobs, "build_wave", side_effect=AssertionError("legacy builder used")
            ) as build, mock.patch.object(
                jobs.diagnostic, "validate_prerequisites",
                side_effect=AssertionError("legacy prerequisite reader used"),
            ), mock.patch.object(
                jobs, "validate_prior_diagnostic",
                side_effect=AssertionError("legacy prior reader used"),
            ), mock.patch.object(
                jobs, "_source_paths", side_effect=source_paths
            ), mock.patch.object(
                jobs, "REPOSITORY_ROOT", control_root
            ), mock.patch.object(
                jobs, "PRIOR_SOURCE_PINS", source_pin
            ), mock.patch.object(
                jobs, "_runtime_contract", return_value={}
            ), mock.patch.object(
                jobs.diagnostic,
                "_runtime_contract",
                return_value={"prerequisite_receipts": expected},
            ), mock.patch.object(
                jobs, "_read_checkout_regular_file_once", side_effect=counting_read,
            ), mock.patch.object(
                jobs.os, "open", side_effect=counting_open,
            ), mock.patch.object(
                jobs.os, "read", side_effect=counting_os_read,
            ), mock.patch.object(jobs.sys, "stdout", sink):
                self.assertEqual(jobs.main(arguments), 0)
            candidate = jobs._parse_json_bytes(
                sink.buffer.getvalue(), "stdout queue candidate"
            )
            self.assertEqual(candidate["jobs"], [descriptor])
            self.assertIs(candidate["shutdown"], False)
            self.assertEqual(
                cluster_queue.normalize_job(candidate["jobs"][0])["job_id"],
                jobs.JOB_ID,
            )
            self.assertEqual(build.call_count, 0)
            for relative in jobs.PREREQUISITE_RESULTS_RELATIVES.values():
                self.assertEqual(observed_reads.count(results_root / relative), 1)
            self.assertEqual(observed_reads.count(prior_path), 1)
            self.assertEqual(low_level_opens, {
                path: 1 for path in single_read_paths
            })
            self.assertEqual(low_level_read_chunks, {
                path: 1 for path in single_read_paths
            })
            self.assertEqual(low_level_read_bytes, expected_low_level_bytes)
            self.assertEqual(
                {
                    "control_index": (
                        control_index.read_bytes(), control_index.stat().st_mtime_ns,
                    ),
                    "results_index": (
                        results_index.read_bytes(), results_index.stat().st_mtime_ns,
                    ),
                    "queue": (queue_path.read_bytes(), queue_path.stat().st_mtime_ns),
                },
                before,
            )

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
