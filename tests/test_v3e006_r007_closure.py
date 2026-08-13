from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.validate_v3e006_r007_closure import gate_summary


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r007/results/results.json"


def test_compact_gate_summary_is_exact_and_mutations_differ() -> None:
    value = json.loads(RESULT.read_text(encoding="utf-8"))
    stage = value["candidate_attempts"][0]["stages"]["canonical_grasp"]
    synthetic = {
        "passed": stage["passed"],
        "normalized_state_sha256": stage["normalized_state_sha256"],
        "physics_gate": stage["physics_gate"],
        "ood_gate": stage["ood_gate"],
        "camera_evidence": {"passed": stage["camera_gate_passed"]},
        "companion_pose_gate": stage["companion_gate"],
        "base_link_to_eef_frame_identity": {"passed": stage["frame_identity_passed"]},
    }
    assert gate_summary(synthetic) == stage
    for key in ("physics_gate", "ood_gate", "normalized_state_sha256"):
        bad = deepcopy(synthetic)
        if key == "normalized_state_sha256":
            bad[key] = "0" * 64
        else:
            bad[key]["passed"] = not bad[key]["passed"]
        assert gate_summary(bad) != stage


def test_closure_binds_distinct_authoritative_and_zero_byte_attempts() -> None:
    value = json.loads(RESULT.read_text(encoding="utf-8"))
    assert value["authoritative_target_validation_receipt"]["sha256"] == "5b02a70eee847773a314f7a37da2f6b2575d04c0e095dcb90a9cda6e2327a476"
    assert value["failed_zero_byte_postexecution_receipt"]["bytes"] == 0
    assert value["postexecution_validator_amendment"]["sha256"] == "c34486692c4d7c451500e1925d731769f33a7fc14e1804c594ccadb2e3a9367a"
