from pathlib import Path
import json
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_v3e002_registration_is_valid():
    subprocess.run([sys.executable, "tools/validate_v3e002.py"], cwd=ROOT, check=True)


def test_v3e002_queue_is_four_cells_per_seed():
    reg = json.loads((ROOT / "artifacts/vla_wam_shared_v3/phase_e/reference_controller_symmetry_v3e002/registration.json").read_text())
    assert len(reg["queue"]) == 108
    for seed in range(9400, 9427):
        assert sum(row["environment_seed"] == seed for row in reg["queue"]) == 4
