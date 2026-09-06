from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.build_v4_second_stack_g5_receipt import build_receipt


class SecondStackG5Tests(unittest.TestCase):
    def test_complete_g3_and_g4_basis_selects_fallback(self) -> None:
        g3 = {
            "schema_version": "v4-second-stack-g3-scripted-aggregate-v1",
            "fixture_id": "second_stack",
            "passed": True,
            "observed_check_count": 112,
            "passed_check_count": 112,
        }
        g4 = {
            "schema_version": "v4-second-stack-g4-policy-session-receipt-v1",
            "fixture_id": "second_stack",
            "policy_id": "groot_n1_7_simplerenv_bridge",
            "passed": True,
            "checks": {
                "fresh_reset_exact_repeat_actions_equal": True,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            g3_path = root / "g3.json"
            g4_path = root / "g4.json"
            g3_path.write_text(json.dumps(g3), encoding="utf-8")
            g4_path.write_text(json.dumps(g4), encoding="utf-8")

            receipt = build_receipt(g3_path=g3_path, g4_path=g4_path)

        self.assertTrue(receipt["passed"])
        self.assertEqual(
            receipt["selected_prefix_mode"],
            "independent_natural_rollout_fallback",
        )
        self.assertTrue(all(receipt["checks"].values()))


if __name__ == "__main__":
    unittest.main()
