from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/cross_arena_geometry_v3e005"


def test_v3e005_registration_validator() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "tools/validate_v3e005.py")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "valid_registered" in completed.stdout


def test_v3e005_exact_grid_and_nested_scenes() -> None:
    rows = [json.loads(line) for line in (BASE / "queue.jsonl").read_text().splitlines()]
    assert len(rows) == 108
    assert len({row["cell_id"] for row in rows}) == 108
    for seed in range(9400, 9427):
        subset = [row for row in rows if row["environment_seed"] == seed]
        assert len(subset) == 4
        assert len({row["scene_id"] for row in subset}) == 1
        assert {(row["symmetry_level_s"], row["relation"]) for row in subset} == {
            (0.0, "left"), (0.0, "right"), (1.0, "left"), (1.0, "right")
        }


def test_v3e005_power_boundary_is_honest() -> None:
    registration = json.loads((BASE / "registration.json").read_text())
    h2 = registration["predictions"]["H2"]
    assert h2["binary"]["margin"] == 0.0
    assert "undefined" in h2["binary"]["status"]
    assert h2["requested_depth_m"]["mde80_n27"] > h2["requested_depth_m"]["margin"] / 2
    assert "underpowered" in h2["requested_depth_m"]["status"]


def test_v3e005_h4_is_hard_first_gate() -> None:
    registration = json.loads((BASE / "registration.json").read_text())
    h4 = registration["predictions"]["H4"]
    assert h4["threshold_m"] == 0.05
    assert h4["role"] == "positive_control_hard_gate_evaluated_first"
    assert registration["analysis"]["h4_must_be_compiled_and_recorded_before_h1_h3"] is True
