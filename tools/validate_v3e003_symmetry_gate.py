#!/usr/bin/env python3
"""Validate the model-blind E003 scene construction before inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.v3.phase_e.bilateral_symmetry_null_control_v3e003.contract import (  # noqa: E402
    B001_CELLS, B001_CELLS_SHA256, EQUIVALENCE_MARGIN, PROMPTS, SEEDS,
    SYMMETRIC_POSITIONS,
)

BASE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/bilateral_symmetry_null_control_v3e003"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    reg = json.loads((BASE / "registration.json").read_text())
    assert reg["status"] == "registered_before_inference"
    assert reg["model_request_count_before_registration"] == 0
    assert reg["behavioral_episode_count_before_registration"] == 0
    assert reg["equivalence_margins"] == EQUIVALENCE_MARGIN
    assert reg["design"]["matched_seeds"] == list(SEEDS)
    assert reg["design"]["prompts"] == PROMPTS
    assert len(reg["queue"]) == 54
    assert len({r["cell_id"] for r in reg["queue"]}) == 54
    assert {r["relation"] for r in reg["queue"]} == {"left", "right"}
    source = ROOT / B001_CELLS
    assert sha(source) == B001_CELLS_SHA256
    candidate = json.loads((BASE / "symmetry_gate/candidate.json").read_text())
    assert candidate["status"] == "model_blind_candidate_not_released_for_inference"
    assert candidate["model_request_count"] == 0 and candidate["behavioral_episode_count"] == 0
    assert candidate["exactly_one_bowl"] is True
    pos = candidate["positions_robot_base_m"]
    assert abs(pos["bowl"][1]) < 0.001
    assert abs(pos["rubiks_cube"][1]) < 0.001
    assert abs(pos["banana_left"][1] + pos["banana_right"][1]) < 0.001
    assert abs(pos["banana_left"][0] - pos["banana_right"][0]) < 0.001
    residual = candidate["symmetry_residual"]
    assert max(residual.values()) < 0.001
    assert candidate["robot_and_non_movable_geometry_unchanged"] is True
    print(json.dumps({"status": "valid", "behavioral_cells": 54, "model_requests": 0, "symmetry_residual_max_m": max(residual.values()), "registration_sha256": sha(BASE / "registration.json"), "candidate_sha256": sha(BASE / "symmetry_gate/candidate.json")}, indent=2))


if __name__ == "__main__":
    main()
