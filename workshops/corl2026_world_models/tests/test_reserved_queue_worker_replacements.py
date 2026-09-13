import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


WORKSHOP = Path(__file__).resolve().parents[1]
MODULE = WORKSHOP / "scripts/generate_reserved_queue_worker_replacements.py"
MANIFEST = (
    WORKSHOP
    / "execution/20260912/autonomy/queue_workers_pod_uid_00_d1.json"
)
spec = importlib.util.spec_from_file_location("reserved_worker_replacements", MODULE)
reserved_worker_replacements = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reserved_worker_replacements)


class ReservedQueueWorkerReplacementTests(unittest.TestCase):
    def _source_job(self, path, name):
        source = json.loads(path.read_text())
        matches = [
            item
            for item in source["items"]
            if item.get("kind") == "Job"
            and item.get("metadata", {}).get("name") == name
        ]
        self.assertEqual(len(matches), 1)
        return matches[0]

    def test_source_manifests_have_the_frozen_reviewed_identities(self):
        for source_path, expected_digest in (
            (path, reserved_worker_replacements.SOURCE_SHA256[path.name])
            for path, _ in reserved_worker_replacements.TARGETS
        ):
            with self.subTest(source=source_path.name):
                self.assertEqual(
                    hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    expected_digest,
                )

    def test_only_the_two_target_jobs_are_emitted(self):
        value = reserved_worker_replacements.build_manifest()
        self.assertEqual(value["apiVersion"], "v1")
        self.assertEqual(value["kind"], "List")
        self.assertEqual(
            [(item["kind"], item["metadata"]["name"]) for item in value["items"]],
            [
                ("Job", "wmf-forecast-0912-worker-00"),
                ("Job", "wmf-forecast-0912-worker-d1-00"),
            ],
        )
        self.assertNotIn("Service", {item["kind"] for item in value["items"]})

    def test_each_job_diff_is_exactly_annotations_and_appended_pod_uid(self):
        value = reserved_worker_replacements.build_manifest()
        actual_by_name = {item["metadata"]["name"]: item for item in value["items"]}

        for source_path, target_name in reserved_worker_replacements.TARGETS:
            with self.subTest(worker=target_name):
                expected = self._source_job(source_path, target_name)
                actual = copy.deepcopy(actual_by_name[target_name])

                self.assertEqual(
                    actual["metadata"]["annotations"].pop(
                        reserved_worker_replacements.POD_UID_ANNOTATION
                    ),
                    reserved_worker_replacements.POD_UID_ANNOTATION_VALUE,
                )
                self.assertEqual(
                    actual["spec"]["template"]["metadata"]["annotations"].pop(
                        reserved_worker_replacements.POD_UID_ANNOTATION
                    ),
                    reserved_worker_replacements.POD_UID_ANNOTATION_VALUE,
                )
                environment = actual["spec"]["template"]["spec"]["containers"][0][
                    "env"
                ]
                self.assertEqual(
                    environment.pop(), reserved_worker_replacements.POD_UID_ENV
                )
                self.assertFalse(any(row.get("name") == "POD_UID" for row in environment))
                if target_name == "wmf-forecast-0912-worker-d1-00":
                    self.assertFalse(
                        any(row.get("name") == "NVIDIA_VISIBLE_DEVICES" for row in environment)
                    )

                # Removing the three deliberate additions must reproduce every
                # byte-decoded source field, including labels, resources,
                # bootstrap digest, admission deadline and command topology.
                self.assertEqual(actual, expected)

    def test_checked_in_manifest_is_exactly_reproducible(self):
        self.assertEqual(
            json.loads(MANIFEST.read_text()),
            reserved_worker_replacements.build_manifest(),
        )

    def test_source_digest_mismatch_fails_closed(self):
        path, _ = reserved_worker_replacements.TARGETS[0]
        with mock.patch.dict(
            reserved_worker_replacements.SOURCE_SHA256,
            {path.name: "0" * 64},
        ):
            with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
                reserved_worker_replacements.build_manifest()

    def test_cli_refuses_to_overwrite_a_different_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.json"
            output.write_text("different\n")
            argv = [str(MODULE), "--output", str(output)]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaises(FileExistsError):
                    reserved_worker_replacements.main()


if __name__ == "__main__":
    unittest.main()
