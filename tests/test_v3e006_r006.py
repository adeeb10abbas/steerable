from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.v3.phase_e.canonical_stage_localization_v3e006_r006.predecessor_contract import (
    validate_r005_exhaustion_closure,
)
from tools.validate_v3e006_r006 import canonical_sha, function_ast, validate_static


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r006"
R005 = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r005"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_r005_real_exhaustion_closure_and_mutations() -> None:
    payload = load(R005 / "results/results.json")
    validate_r005_exhaustion_closure(payload)
    for key, value in (
        ("candidate_pair_evaluation_count", 3),
        ("diagnostics_all_passed", False),
        ("model_request_count", 1),
        ("accepted_candidate_rank", 1),
    ):
        bad = deepcopy(payload); bad[key] = value
        with pytest.raises(ValueError):
            validate_r005_exhaustion_closure(bad)
    bad = deepcopy(payload); bad["raw_result"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_r005_exhaustion_closure(bad)


def test_r006_registration_schedule_are_prospective_and_finite() -> None:
    registration = load(ARTIFACT / "repair_registration.json")
    schedule = load(ARTIFACT / "gates/candidate_schedule.json")
    assert registration["repair_amendment_id"] == "V3-E006-R006"
    assert registration["predecessor_repair_amendment_id"] == "V3-E006-R005"
    assert registration["counts_at_registration"] == {
        "r006_live_diagnostics": 0, "r006_live_candidate_evaluations": 0,
        "model_requests": 0, "behavioral_episodes": 0,
    }
    assert schedule["candidate_budget"] == schedule["diagnostic_budget"] == 4
    assert [row["candidate_rank"] for row in schedule["candidate_pairs"]] == [1, 2, 3, 4]
    assert [row["diagnostic_index_one_based"] for row in schedule["known_reachable_diagnostics"]] == [1, 2, 3, 4]
    assert schedule["schedule_canonical_sha256_without_this_field"] == canonical_sha({
        key: value for key, value in schedule.items()
        if key != "schedule_canonical_sha256_without_this_field"
    })


def test_r006_is_exact_r005_schedule_plus_equilibrium_contract() -> None:
    schedule = load(ARTIFACT / "gates/candidate_schedule.json")
    old = load(R005 / "gates/candidate_schedule.json")
    assert schedule["known_reachable_diagnostics"] == old["known_reachable_diagnostics"]
    assert schedule["candidate_pairs"] == old["candidate_pairs"]
    assert schedule["residual_correction_contract"] == old["residual_correction_contract"]
    assert schedule["construction_horizon_contract"] == old["construction_horizon_contract"]
    assert schedule["target_contact_and_rank_identity"] == old["target_contact_and_rank_identity"]
    assert schedule["unchanged_gate_bindings"] == old["unchanged_gate_bindings"]
    contract = schedule["joint_equilibrium_hold_contract"]
    assert canonical_sha(contract) == schedule["joint_equilibrium_hold_contract_sha256"]
    assert contract["settle_steps"] == 780
    assert contract["required_episode_length_buf_before_settle"] == 75
    assert contract["worst_case_materialization_steps"] == 855
    assert contract["fixed_margin_after_reset_and_settle_steps"] == 45


def test_runtime_has_one_target_write_and_no_cartesian_equilibrium_action() -> None:
    path = ROOT / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r006/state_repair_gate.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    funcs = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    step = ast.dump(funcs["_normal_joint_equilibrium_step"], include_attributes=False)
    materialize = ast.dump(funcs["_direct_materialize_and_gate"], include_attributes=False)
    assert "set_joint_position_target" not in step
    assert "process_action" not in step and "apply_action" not in step
    assert materialize.count("set_joint_position_target") == 1
    assert "_normal_joint_equilibrium_step" in materialize
    assert "780" in materialize and "770" in materialize
    assert "episode_length_before_equilibrium" in materialize
    assert "75" in materialize and "855" not in materialize  # 855 is observed, not hard-written.


def test_unchanged_scientific_gate_functions_are_exact_r005_ast() -> None:
    old = ROOT / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r005/state_repair_gate.py"
    new = ROOT / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r006/state_repair_gate.py"
    for name in (
        "_contact_forces", "_contact_coverage", "_reference_bounds", "_save_camera_evidence",
        "_companion_gate", "_fresh_reset_and_gate", "_finalize_unchanged_gates",
    ):
        assert function_ast(old, name) == function_ast(new, name)


def test_zero_model_and_no_scientific_gate_relaxation() -> None:
    source = (ROOT / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r006/state_repair_gate.py").read_text(encoding="utf-8")
    assert "requests.post" not in source
    assert "httpx" not in source
    assert "policy_server" not in source
    assert 'cube_midline_residual_m_strict": 0.001' not in source  # Threshold remains imported, never restated/edited.
    assert "settled_gate(settled" in source and "stage_ood(state" in source


def test_full_pre_source_validator_passes() -> None:
    result = validate_static(ROOT, require_source_gate=False)
    assert result["passed"] is True
    assert result["candidate_pair_count"] == result["diagnostic_count"] == 4
    assert result["joint_equilibrium_hold_contract_sha256"] == load(
        ARTIFACT / "gates/candidate_schedule.json"
    )["joint_equilibrium_hold_contract_sha256"]
