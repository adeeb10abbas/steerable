from __future__ import annotations

import argparse
import ast
import importlib.util
from pathlib import Path
import tempfile
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


class CameraCropWitnessJobTests(unittest.TestCase):
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
        self.assertEqual(contract["job"]["worker_role"], "wmf-forecast-0912-worker-06")
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
        self.assertEqual(first["science_counts"], jobs.zero_science_counts())
        self.assertIs(first["safe_to_release_confirmation"], False)
        descriptor = first["jobs"][0]
        self.assertEqual(descriptor["job_id"], jobs.JOB_ID)
        self.assertEqual(descriptor["role"], "wmf-forecast-0912-worker-06")
        self.assertEqual(descriptor["argv"][0], "/usr/bin/python3")
        command = " ".join(descriptor["argv"])
        self.assertIn("camera_crop_witness_jobs.py run", command)
        self.assertNotIn("kubectl", command)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", command)
        for value in implementation.values():
            self.assertIn(value["sha256"], command)

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
        self.assertIn("start_new_session=True", source)
        self.assertIn("os.killpg", source)
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

    def test_success_receipt_code_keeps_all_authority_flags_false(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        run = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_job"
        )
        text = ast.get_source_segment(source, run)
        assert text is not None
        self.assertIn('"simulator_state_render_used": False', text)
        self.assertIn('"whole_frame_identity": False', text)
        self.assertIn('"safe_to_release_confirmation": False', text)
        self.assertIn('"confirmation_released": False', text)


if __name__ == "__main__":
    unittest.main()
