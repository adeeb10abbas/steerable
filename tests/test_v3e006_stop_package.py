from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parents[1]
BASE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006"


def test_v3e006_stop_validator() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "tools/validate_v3e006.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_v3e006_stopped_before_registration_or_inference() -> None:
    results = json.loads((BASE / "results/results.json").read_text(encoding="utf-8"))
    release = json.loads((BASE / "release_gate.json").read_text(encoding="utf-8"))
    assert results["status"] == "gate_failed_no_valid_candidate_stop_before_registration"
    assert not (BASE / "registration.json").exists()
    assert not (BASE / "queue.jsonl").exists()
    assert release["release_for_inference"] is False
    assert results["model_request_count"] == results["behavioral_episode_count"] == results["state_candidate_count"] == 0
