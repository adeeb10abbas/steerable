from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.v3.phase_e.canonical_stage_localization_v3e006_r001 import candidate_schedule as schedule
from tools.validate_v3e006_r001 import validate_package


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
