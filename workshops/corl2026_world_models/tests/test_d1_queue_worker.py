import importlib.util
from pathlib import Path
import tempfile
import unittest


MODULE = Path(__file__).resolve().parents[1] / "scripts/generate_d1_queue_worker.py"
spec = importlib.util.spec_from_file_location("d1_worker", MODULE)
d1_worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d1_worker)


class D1QueueWorkerTests(unittest.TestCase):
    def test_exact_two_b200_worker_and_service(self):
        digest = "a" * 64
        manifest = d1_worker.build_manifest(digest, 1789868932)
        self.assertEqual([item["kind"] for item in manifest["items"]], ["Job", "Service"])
        job, service = manifest["items"]
        container = job["spec"]["template"]["spec"]["containers"][0]
        self.assertEqual(container["resources"]["requests"]["nvidia.com/gpu"], 2)
        self.assertEqual(container["resources"]["limits"]["nvidia.com/gpu"], 2)
        self.assertEqual(job["spec"]["template"]["spec"]["nodeSelector"]["nvidia.com/gpu.product"], "NVIDIA-B200")
        self.assertFalse(job["spec"]["template"]["spec"]["automountServiceAccountToken"])
        self.assertIn("d1", container["args"])
        self.assertIn("1789868932", container["args"])
        self.assertEqual(service["spec"]["selector"]["batch.kubernetes.io/job-name"], d1_worker.NAME)
        self.assertEqual(service["spec"]["ports"][0]["port"], 18021)

    def test_digest_deadline_and_immutable_output_fail_closed(self):
        with self.assertRaises(ValueError):
            d1_worker.build_manifest("bad", 1789868932)
        with self.assertRaises(ValueError):
            d1_worker.build_manifest("a" * 64, 0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.json"
            output.write_text("different\n")
            with self.assertRaises(FileExistsError):
                old_argv = __import__("sys").argv
                try:
                    __import__("sys").argv = [
                        str(MODULE), "--bootstrap-sha256", "a" * 64,
                        "--admission-deadline-unix", "1789868932", "--output", str(output),
                    ]
                    d1_worker.main()
                finally:
                    __import__("sys").argv = old_argv


if __name__ == "__main__":
    unittest.main()
