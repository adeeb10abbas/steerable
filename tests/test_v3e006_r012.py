from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.v3.phase_e.canonical_stage_localization_v3e006_r012.pinch_geometry import (
    validate_attachment_preflight_contract,
)
from experiments.v3.phase_e.canonical_stage_localization_v3e006_r012.predecessor_contract import (
    R011_RESULTS_SHA256,
    validate_r011_scene_sync_failure_closure,
)
from tools.validate_v3e006_r012 import ValidationError, validate_static


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r012"
R011 = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r011"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def function_ast(path: Path, name: str) -> str:
    node = next(
        row for row in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(row, ast.FunctionDef) and row.name == name
    )
    return ast.dump(node, include_attributes=False)


def test_real_r011_closure_and_mutations() -> None:
    value = load(R011 / "results/results.json")
    validate_r011_scene_sync_failure_closure(value)
    assert R011_RESULTS_SHA256 == "f6e869939d5003c175abb21c0544d2d02313fe00e7a0dd7831d60e0c7f192054"
    for key, replacement in (
        ("geometry_attachment_preflight_count", 0),
        ("candidate_pair_evaluation_count", 1),
        ("model_request_count", 1),
    ):
        bad = deepcopy(value); bad[key] = replacement
        with pytest.raises(ValueError):
            validate_r011_scene_sync_failure_closure(bad)
    bad = deepcopy(value); bad["raw_result"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_r011_scene_sync_failure_closure(bad)


def test_registration_schedule_are_finite_and_preserve_r011() -> None:
    registration = load(ART / "repair_registration.json")
    schedule = load(ART / "gates/candidate_schedule.json")
    prior = load(R011 / "gates/candidate_schedule.json")
    assert registration["predecessor_repair_amendment_id"] == "V3-E006-R011"
    assert registration["counts_at_registration"] == {
        "r012_geometry_attachment_preflights": 0,
        "r012_live_diagnostics": 0,
        "r012_live_candidate_evaluations": 0,
        "model_requests": 0,
        "behavioral_episodes": 0,
    }
    assert schedule["candidate_budget"] == schedule["diagnostic_budget"] == 4
    assert schedule["candidate_pairs"] == prior["candidate_pairs"]
    assert schedule["known_reachable_diagnostics"] == prior["known_reachable_diagnostics"]
    for key in (
        "pinch_geometry_contract", "joint_handoff_contract",
        "construction_lifecycle_contract", "residual_correction_contract",
    ):
        assert schedule[key] == prior[key]
    assert "scene_sync_source_bindings" not in schedule
    assert schedule["selection_rule"]["algorithm_version"] == (
        "r012-live-tensor-relative-bound-collision-pinch-first-passing-pair-v1"
    )


def test_live_tensor_contract_and_mutations() -> None:
    contract = load(ART / "gates/candidate_schedule.json")["geometry_attachment_preflight_contract"]
    validate_attachment_preflight_contract(contract)
    assert contract["dynamic_usd_world_state_used"] is False
    assert contract["physics_to_usd_sync_call_count"] == 0
    assert contract["dynamic_usd_world_bound_or_xform_query_count"] == 0
    for key, replacement in (
        ("dynamic_usd_world_state_used", True),
        ("physics_to_usd_sync_call_count", 1),
        ("pad_collision_center_separation_m_inclusive", [0.0, 1.0]),
        ("cube_aabb_dimension_m_each_inclusive", [0.0, 1.0]),
    ):
        bad = deepcopy(contract); bad[key] = replacement
        with pytest.raises(ValueError):
            validate_attachment_preflight_contract(bad)


def test_active_preflight_is_tensor_only_and_controller_is_unchanged() -> None:
    old = ROOT / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r011/state_repair_gate.py"
    new = ROOT / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r012/state_repair_gate.py"
    preflight = function_ast(new, "_run_geometry_attachment_preflight")
    for forbidden in (
        "ComputeWorldBound", "XformCache", "update_transformations", "omni.physx",
        "PhysicsSchemaTools", "UsdUtils", "SimulationContext", "carb.settings", "env.step",
    ):
        assert forbidden not in preflight
    assert "_live_pinch_bounds" in preflight
    for name in (
        "_collision_geometry_body_local", "_resolve_pinch_scene_geometry",
        "_live_pinch_bounds", "_pinch_geometry_materialize_and_gate",
        "_contact_forces", "_contact_coverage", "_fresh_reset_and_gate",
        "_finalize_unchanged_gates",
    ):
        assert function_ast(old, name) == function_ast(new, name).replace("R012", "R011")
    main = function_ast(new, "main")
    assert main.index("_run_geometry_attachment_preflight") < main.index("known_reachable_diagnostics")


def test_static_package_pre_source_gate() -> None:
    summary = validate_static(ROOT, source_gate_required=False)
    assert summary["passed"] is True
    assert summary["candidate_pair_count"] == summary["diagnostic_count"] == 4


def test_static_rejects_mutated_contract(tmp_path: Path) -> None:
    schedule_path = ART / "gates/candidate_schedule.json"
    original = schedule_path.read_bytes()
    schedule = json.loads(original)
    schedule["geometry_attachment_preflight_contract"]["physics_to_usd_sync_call_count"] = 1
    schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
    try:
        with pytest.raises((ValidationError, ValueError)):
            validate_static(ROOT, source_gate_required=False)
    finally:
        schedule_path.write_bytes(original)


def test_outer_accepts_registered_terminal_statuses() -> None:
    source = (ROOT / "tools/run_v3e006_r012_state_repair.py").read_text(encoding="utf-8")
    for status in (
        "r012_geometry_attachment_preflight_failed_candidates_not_evaluated",
        "r012_known_reachable_diagnostic_failed_candidates_not_evaluated",
        "r012_candidate_budget_exhausted_no_valid_state_pair",
        "passed_r012_state_repair_not_released_for_behavior",
    ):
        assert status in source
    assert "experiment/v3e006-r012-live-tensor-repair" in source
