from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004"


def load_validator():
    spec = importlib.util.spec_from_file_location("validate_v3e004", ROOT / "tools/validate_v3e004.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_registration_validator_passes():
    report = load_validator().validate_registration()
    assert report["queue_rows"] == 4096
    assert report["new_behavioral_cells"] == 4096


def test_registration_keeps_arena_boundary_and_power_boundary():
    registration = json.loads((BASE / "registration.json").read_text())
    assert registration["design"]["droid_and_robotwin_never_pooled"] is True
    rows = registration["power_registration"]["rows"]
    assert any(row["status"] == "underpowered_no_equivalence_claim" for row in rows)
    assert next(
        row for row in rows
        if row["model_id"] == "cosmos3_nano_policy_droid" and row["estimand"] == "binary_R_minus_L"
    )["strict_n"] is None


def test_s0_controls_are_linked_as_comparators_but_newly_measured():
    rows = [json.loads(line) for line in (BASE / "queue.jsonl").read_text().splitlines()]
    linked = [row for row in rows if "historical_control_comparator_cell_id" in row]
    assert len(linked) == 162
    assert {row["model_id"] for row in linked} == {
        "pi05_current_stack_droid",
        "cosmos3_nano_policy_droid",
        "dreamzero_droid_action_cfg",
    }
    assert all(row["symmetry_level_s"] == 0.0 for row in linked)
    assert all(row["execution_mode"] == "new_behavioral_episode" for row in rows)
