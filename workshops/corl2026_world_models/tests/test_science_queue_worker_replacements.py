import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


MODULE = (
    Path(__file__).resolve().parents[1]
    / "scripts/generate_science_queue_worker_replacements.py"
)
MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "execution/20260912/autonomy/science_queue_workers_pod_uid_050609.json"
)
spec = importlib.util.spec_from_file_location("science_worker_replacements", MODULE)
science_worker_replacements = importlib.util.module_from_spec(spec)
spec.loader.exec_module(science_worker_replacements)


class ScienceQueueWorkerReplacementTests(unittest.TestCase):
    def test_exact_targeted_workers_bind_pod_uid_and_retain_frozen_topology(self):
        digest = "a" * 64
        deadline = 1789868932
        value = science_worker_replacements.build_manifest(
            digest,
            deadline,
            [9, 5, 6],
        )
        expected_names = [
            "wmf-forecast-0912-worker-05",
            "wmf-forecast-0912-worker-06",
            "wmf-forecast-0912-worker-09",
        ]
        self.assertEqual(value["apiVersion"], "v1")
        self.assertEqual(value["kind"], "List")
        self.assertEqual(
            [item["metadata"]["name"] for item in value["items"]],
            expected_names,
        )
        for item, expected_name in zip(value["items"], expected_names):
            self.assertEqual(item["kind"], "Job")
            self.assertEqual(item["metadata"]["namespace"], "211247-prod")
            self.assertEqual(item["metadata"]["labels"]["user"], "ali")
            self.assertEqual(item["metadata"]["labels"]["wmf-role"], "worker")
            self.assertEqual(item["metadata"]["annotations"]["wmf-bootstrap-sha256"], digest)
            self.assertEqual(
                item["metadata"]["annotations"][
                    science_worker_replacements.POD_UID_ANNOTATION
                ],
                science_worker_replacements.POD_UID_ANNOTATION_VALUE,
            )

            template = item["spec"]["template"]
            pod = template["spec"]
            container = pod["containers"][0]
            pod_uid_rows = [row for row in container["env"] if row.get("name") == "POD_UID"]
            self.assertEqual(
                pod_uid_rows,
                [science_worker_replacements.POD_UID_ENV],
            )
            self.assertNotIn("value", pod_uid_rows[0])
            self.assertEqual(
                template["metadata"]["annotations"][
                    science_worker_replacements.POD_UID_ANNOTATION
                ],
                science_worker_replacements.POD_UID_ANNOTATION_VALUE,
            )
            self.assertNotIn(
                "NVIDIA_VISIBLE_DEVICES",
                {row["name"] for row in container["env"]},
            )
            self.assertEqual(container["resources"]["requests"]["nvidia.com/gpu"], 1)
            self.assertEqual(container["resources"]["limits"]["nvidia.com/gpu"], 1)
            self.assertEqual(
                pod["nodeSelector"]["nvidia.com/gpu.product"],
                "NVIDIA-B200",
            )
            self.assertFalse(pod["automountServiceAccountToken"])
            self.assertEqual(
                pod["volumes"][0]["persistentVolumeClaim"]["claimName"],
                "211247-prod-pvc",
            )
            worker_id_flag = container["args"].index("--worker-id")
            role_flag = container["args"].index("--role")
            self.assertEqual(container["args"][worker_id_flag + 1], expected_name)
            self.assertEqual(container["args"][role_flag + 1], expected_name)
            self.assertIn(str(deadline), container["args"])

    def test_selection_is_explicit_unique_and_excludes_reserved_workers(self):
        invalid_selections = (
            [],
            [0],
            [1],
            [2],
            [3],
            [4],
            [32],
            [-1],
            [5, 5],
            [True],
            ["5"],
        )
        for selection in invalid_selections:
            with self.subTest(selection=selection):
                with self.assertRaises(ValueError):
                    science_worker_replacements.build_manifest(
                        "a" * 64,
                        1789868932,
                        selection,
                    )

    def test_manifest_is_order_independent(self):
        first = science_worker_replacements.build_manifest(
            "a" * 64,
            1789868932,
            [5, 6, 9],
        )
        second = science_worker_replacements.build_manifest(
            "a" * 64,
            1789868932,
            [9, 5, 6],
        )
        self.assertEqual(first, second)

    def test_checked_in_manifest_is_exactly_reproducible(self):
        expected = science_worker_replacements.build_manifest(
            "c9fe3d2da0b889eae2bb6cff4c8c149583e4293b366840134e14dcb07a18ba22",
            1789868932,
            [5, 6, 9],
        )
        self.assertEqual(json.loads(MANIFEST.read_text()), expected)

    def test_cli_refuses_to_overwrite_a_different_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.json"
            output.write_text("different\n")
            argv = [
                str(MODULE),
                "--bootstrap-sha256",
                "a" * 64,
                "--admission-deadline-unix",
                "1789868932",
                "--worker-index",
                "5",
                "--output",
                str(output),
            ]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaises(FileExistsError):
                    science_worker_replacements.main()


if __name__ == "__main__":
    unittest.main()
