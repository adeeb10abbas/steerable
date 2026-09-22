import json
import multiprocessing
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from experiments.workshops.spatial_grounding_v1.cluster_probe import persist, probe, wait_for


def run_role(root, role):
    with patch("socket.gethostname", return_value=f"test-pod-{role}"):
        probe(Path(root), role, 10)


class ClusterProbeTests(unittest.TestCase):
    def test_publication_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            persist(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                persist(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text()), {"value": 1})
            self.assertEqual([p.name for p in path.parent.iterdir()], ["receipt.json"])

    def test_timeout_is_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TimeoutError):
                wait_for(Path(directory) / "missing.json", time.monotonic())

    def test_exclusion_and_crash_recovery(self):
        context = multiprocessing.get_context("fork")
        with tempfile.TemporaryDirectory() as directory:
            processes = [
                context.Process(target=run_role, args=(directory, role))
                for role in ("owner", "contender")
            ]
            try:
                for process in processes:
                    process.start()
                for process in processes:
                    process.join(timeout=15)
                    self.assertEqual(process.exitcode, 0)
                receipt = json.loads((Path(directory) / "lock_receipt.json").read_text())
                self.assertEqual(receipt["status"], "passed")
                self.assertNotEqual(receipt["owner_pod"], receipt["contender_pod"])
                self.assertEqual(receipt["owner_exitcode"], -9)
                self.assertEqual(receipt["model_requests"], 0)
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
