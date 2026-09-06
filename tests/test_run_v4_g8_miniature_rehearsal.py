from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tools.run_v4_g8_miniature_rehearsal import render_outcome_figure


class G8MiniatureRehearsalTests(unittest.TestCase):
    def test_outcome_figure_preserves_valid_failures(self) -> None:
        rows = [
            {"outcome": {"failure_label": "no_grasp"}},
            {"outcome": {"failure_label": "no_grasp"}},
            {"outcome": {"failure_label": "wrong_relation"}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "figure.svg"
            counts = render_outcome_figure(rows, output)
            self.assertEqual(
                counts,
                {"no_grasp": 2, "wrong_relation": 1},
            )
            svg = output.read_text(encoding="utf-8")
            self.assertIn("V4 G8 miniature rehearsal outcomes", svg)
            self.assertIn("no_grasp", svg)


if __name__ == "__main__":
    unittest.main()
