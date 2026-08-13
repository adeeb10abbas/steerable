from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.v3.phase_e.canonical_stage_localization_v3e006_r001 import candidate_schedule as schedule
from tools.validate_v3e006_r001 import validate_package
from tools.run_v3e006_r001_state_repair import E004_APP_LAUNCHER_ARGV, child_process_completed


ROOT = Path(__file__).resolve().parents[1]
REPAIR = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r001"


def _load(name: str) -> dict:
    return json.loads((REPAIR / name).read_text(encoding="utf-8"))


def test_static_prospective_package_passes_and_preserves_original() -> None:
    result = validate_package(ROOT)
    assert result["passed"] is True
    assert result["candidate_pairs"] == 8
    assert result["original_files_verified"] == 28


def test_rank_one_anchors_are_exact_and_direction_paired() -> None:
    frozen = _load("gates/candidate_schedule.json")
    assert [row["candidate_rank"] for row in frozen["candidate_pairs"]] == list(range(1, 9))
    first = frozen["candidate_pairs"][0]
    grasp = first["canonical_grasp"]
    carry = first["canonical_carry"]
    assert (grasp["environment_seed"], grasp["source_states"]["left"]["state_capture_index"], grasp["source_states"]["left"]["hdf5_index"], grasp["source_states"]["right"]["state_capture_index"], grasp["source_states"]["right"]["hdf5_index"]) == (9521, 30, 104, 31, 105)
    assert (carry["environment_seed"], carry["source_states"]["left"]["state_capture_index"], carry["source_states"]["left"]["hdf5_index"], carry["source_states"]["right"]["state_capture_index"], carry["source_states"]["right"]["hdf5_index"]) == (9442, 39, 113, 38, 112)


def test_schedule_contract_digest_and_zero_counts_are_frozen() -> None:
    registration = _load("repair_registration.json")
    encoded = json.dumps(registration["candidate_search"], allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    assert hashlib.sha256(encoded).hexdigest() == schedule.CANDIDATE_SEARCH_SHA256
    assert registration["candidate_search"]["maximum_candidate_pairs"] == schedule.MAXIMUM_CANDIDATE_PAIRS == 8
    assert registration["counts_at_registration"] == {
        "behavioral_episodes": 0,
        "model_requests": 0,
        "repair_candidate_evaluations": 0,
    }


def test_stage_enumeration_uses_registered_all_qualifying_rows() -> None:
    import numpy as np

    steps = [{"object_grabbed": value} for value in (False, True, True, True, True, True, True)]
    cube = np.zeros((7, 3), dtype=np.float64)
    cube[:, 2] = [0.0, 0.01, 0.02, 0.025, 0.03, 0.045, 0.055]
    assert schedule._stage_indices(steps, cube, stage="canonical_grasp", offset=0) == [3, 4]
    assert schedule._stage_indices(steps, cube, stage="canonical_carry", offset=0) == [5, 6]


def test_build_rejects_mutated_candidate_search_contract(tmp_path: Path) -> None:
    registration = _load("repair_registration.json")
    registration["candidate_search"]["maximum_candidate_pairs"] = 9
    bad = tmp_path / "registration.json"
    bad.write_text(json.dumps(registration), encoding="utf-8")
    with pytest.raises(schedule.CandidateScheduleError, match="candidate algorithm|candidate budget|candidate-search"):
        schedule.build(
            registration=bad,
            ood_freeze=ROOT / registration["frozen_inputs"]["ood_freeze"]["path"],
            original_closure_binding=REPAIR / "gates/original_v3e006_closure_binding.json",
            original_study_root=ROOT,
            output=tmp_path / "out.json",
        )


def test_outer_launcher_uses_exact_working_e004_runtime_flags() -> None:
    assert E004_APP_LAUNCHER_ARGV == (
        "--headless",
        "--device", "cuda:0",
        "--num-envs", "1",
        "--num-runs", "1",
        "--renderer", "realtime",
        "--rendering-type", "balanced",
        "--video-mode", "viewport",
        "--instruction-type", "default",
        "--disable-subtask",
        "--kit_args=--/rtx/verifyDriverVersion/enabled=false",
    )
    assert "--rendering_mode" not in E004_APP_LAUNCHER_ARGV


def test_outer_never_classifies_a_failure_report_as_completed(tmp_path: Path) -> None:
    failure = tmp_path / "state_construction_failure.json"
    failure.write_text("{}", encoding="utf-8")
    result = tmp_path / "state_repair_result.json"
    result.write_text("{}", encoding="utf-8")
    failed_payload = {
        "status": "infrastructure_invalid_r001_state_repair",
        "passed": False,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "repair_candidate_evaluation_count": 1,
    }
    assert child_process_completed(0, failure, failed_payload) is False
    passed_payload = {
        "status": "passed_r001_state_repair_not_released_for_behavior",
        "passed": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "repair_candidate_evaluation_count": 2,
    }
    assert child_process_completed(0, result, passed_payload) is True
    assert child_process_completed(1, result, passed_payload) is False


def test_lifecycle_repair_is_prospective_zero_count_and_science_frozen() -> None:
    repair = _load("gates/runtime_lifecycle_repair_v1.json")
    assert repair["status"] == (
        "prospectively_frozen_after_infrastructure_invalid_attempt01_before_any_retry_or_model_request"
    )
    assert repair["counts_at_freeze"] == {
        "model_requests": 0,
        "behavioral_episodes": 0,
        "accepted_state_candidates": 0,
        "completed_candidate_pairs": 0,
        "infrastructure_invalid_search_attempts": 1,
    }
    assert repair["frozen_scientific_contracts"]["candidate_schedule_sha256"] == (
        "022b9b65b58758a192dd77ac32a79d463f8638c9be162a7f6b1e3d5270d9d04f"
    )
    assert repair["prospective_runtime_only_repair"]["maximum_new_environment_instances"] == 16
    assert repair["frozen_scientific_contracts"]["state_contract_sha256"] == (
        "2476b28d2867c1b87f477fd5f89e545616be00d860d4144f8cbdb70af10f3c18"
    )
    assert repair["frozen_scientific_contracts"]["ood_reference_sha256"] == (
        "4df1ebf0061096a74b5eccd10b2a144e840f52fd50469b8bdae9369d1696fd04"
    )
    source = (
        ROOT
        / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r001/state_repair_gate.py"
    ).read_text(encoding="utf-8")
    stage_loop = source.index('for stage_name in ("canonical_grasp", "canonical_carry"):')
    create = source.index("stage_env, cfg = create_env(", stage_loop)
    close = source.index("stage_env.close()", create)
    assert stage_loop < create < close
    assert "_record_termination(" in source
    failure_write = source.rindex("failure_path = _write_failure(exc)")
    nonzero_exit = source.rindex("os._exit(1)")
    assert failure_write < nonzero_exit


def test_attempt01_is_retained_only_as_zero_count_infrastructure() -> None:
    rows = [
        json.loads(line)
        for line in (REPAIR / "infrastructure_attempts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["attempt_id"] == "v3e006-r001-state-repair-attempt01"
    assert row["model_request_count"] == row["behavioral_episode_count"] == 0
    assert row["state_candidate_count"] == row["completed_candidate_pair_count"] == 0
    assert row["behavioral_denominator_included"] is row["candidate_denominator_included"] is False
    assert row["harness_misclassification"]["scientific_completion"] is False
