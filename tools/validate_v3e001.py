#!/usr/bin/env python3
"""Fail-closed registration/output validator for V3-E001."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "artifacts/vla_wam_shared_v3/phase_e/fixed_observation_prompt_noise_v3e001/registration.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    reg = json.loads(REG.read_text(encoding="utf-8"))
    assert reg["schema_version"] == "vla-wam-shared-v3e001-registration-v1"
    assert reg["status"] == "registered_before_inference"
    assert reg["design"]["total_model_requests"] == 336
    assert reg["design"]["behavioral_episode_count"] == 0
    assert reg["policy_sampling_seeds"] == list(range(9400, 9427))
    assert set(reg["models"]) == {"pi05_current_stack_droid", "cosmos3_nano_policy_droid", "dreamzero_droid_action_cfg"}
    assert reg["prompts"]["left"] != reg["prompts"]["right"]
    for item in reg["parent_bindings"]:
        path = ROOT / item["path"]
        assert path.is_file(), path
        assert path.stat().st_size == item["bytes"], path
        assert sha256(path) == item["sha256"], path
    report = REG.parent / "results" / "compiled_results.json"
    if report.exists():
        value = json.loads(report.read_text(encoding="utf-8"))
        assert value.get("behavioral_episode_count") == 0
        assert value.get("model_request_count") == 336
    print(json.dumps({"status": "valid", "registration_sha256": sha256(REG), "requests": 336, "behavioral_episodes": 0}, indent=2))


if __name__ == "__main__":
    main()
