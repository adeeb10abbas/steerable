from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.derive_v4_lane_spec import derive_spec, parse_override


class DeriveV4LaneSpecTests(unittest.TestCase):
    def test_derives_nested_values_and_absolutizes_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_file = root / "bound.py"
            source_file.write_text("pass\n", encoding="utf-8")
            spec_path = root / "spec.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "lane_id": "old",
                        "policy": {
                            "gpu_product": "GPU-A",
                            "file_bindings": [
                                {"source": "bound.py", "path": "/bound.py"}
                            ],
                        },
                        "simulator": {"file_bindings": []},
                    }
                ),
                encoding="utf-8",
            )
            result = derive_spec(
                source_path=spec_path,
                overrides=[
                    'lane_id="new"',
                    'policy.gpu_product="GPU-B"',
                ],
            )
            self.assertEqual(result["lane_id"], "new")
            self.assertEqual(result["policy"]["gpu_product"], "GPU-B")
            self.assertEqual(
                result["policy"]["file_bindings"][0]["source"],
                str(source_file.resolve()),
            )

    def test_rejects_non_json_override(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be JSON"):
            parse_override("lane_id=not-json")


if __name__ == "__main__":
    unittest.main()
