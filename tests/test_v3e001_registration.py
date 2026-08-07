from pathlib import Path
import json
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_v3e001_registration_is_valid():
    subprocess.run([sys.executable, "tools/validate_v3e001.py"], cwd=ROOT, check=True)


def test_v3e001_counts_and_prompts():
    reg = json.loads((ROOT / "artifacts/vla_wam_shared_v3/phase_e/fixed_observation_prompt_noise_v3e001/registration.json").read_text())
    assert reg["design"]["total_model_requests"] == 336
    assert reg["design"]["behavioral_episode_count"] == 0
    assert reg["prompts"]["left"] != reg["prompts"]["right"]
