from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "deploy/k8s/v4_lane_bundle/scripts"
    / "run_online_correction_v4_lane_dispatch.py"
)
SPEC = importlib.util.spec_from_file_location("v4_lane_dispatch", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
dispatch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dispatch)


class LaneDispatchTests(unittest.TestCase):
    def test_runtime_working_directory_comes_from_released_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime_root = Path(tmp) / "study"
            marker = (
                runtime_root
                / "experiments/online_correction_v4"
                / "droid_contract.py"
            )
            marker.parent.mkdir(parents=True)
            marker.write_text("", encoding="utf-8")
            lock = Path(tmp) / "runtime-lock.json"
            lock.write_text(
                json.dumps(
                    {
                        "runner": {
                            "entrypoint": str(
                                runtime_root / "tools/run_online_correction_v4.py"
                            )
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                dispatch.resolve_runtime_working_directory(
                    {"runtime_lock_path": str(lock)}
                ),
                runtime_root,
            )

    def test_runtime_runner_replaces_staged_copy_after_hash_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime_runner = root / "study/tools/run_online_correction_v4.py"
            runtime_runner.parent.mkdir(parents=True)
            runtime_runner.write_text("print('runtime')\n", encoding="utf-8")
            staged_runner = root / "bindings/run_online_correction_v4.py"
            staged_runner.parent.mkdir()
            staged_runner.write_bytes(runtime_runner.read_bytes())
            resolved = dispatch.resolve_runtime_runner(
                {"runner_sha256": dispatch.sha256_file(runtime_runner)},
                root / "study",
                staged_runner,
            )
            self.assertEqual(resolved, runtime_runner)


if __name__ == "__main__":
    unittest.main()
