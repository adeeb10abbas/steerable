import importlib.util
from pathlib import Path
import unittest


MODULE = Path(__file__).resolve().parents[1] / "scripts/generate_science_queue_workers.py"
spec = importlib.util.spec_from_file_location("science_workers", MODULE)
science_workers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(science_workers)


class ScienceQueueWorkerTests(unittest.TestCase):
    def test_exact_replacement_pool_uses_plugin_visibility(self):
        value = science_workers.build_manifest("a" * 64, 1789868932)
        self.assertEqual(len(value["items"]), 30)
        names = {item["metadata"]["name"] for item in value["items"]}
        self.assertNotIn("wmf-forecast-0912-worker-02", names)
        self.assertNotIn("wmf-forecast-0912-worker-03", names)
        self.assertIn("wmf-forecast-0912-worker-00", names)
        self.assertIn("wmf-forecast-0912-worker-31", names)
        for item in value["items"]:
            pod = item["spec"]["template"]["spec"]
            container = pod["containers"][0]
            environment = {row["name"]: row["value"] for row in container["env"]}
            self.assertNotIn("NVIDIA_VISIBLE_DEVICES", environment)
            self.assertEqual(container["resources"]["requests"]["nvidia.com/gpu"], 1)
            self.assertEqual(pod["nodeSelector"]["nvidia.com/gpu.product"], "NVIDIA-B200")
            self.assertFalse(pod["automountServiceAccountToken"])
            self.assertEqual(
                item["metadata"]["annotations"]["wmf-device-visibility"],
                "kubernetes-device-plugin-injected",
            )


if __name__ == "__main__":
    unittest.main()
