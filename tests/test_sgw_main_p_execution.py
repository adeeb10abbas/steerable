import json
from pathlib import Path

import pytest

from experiments.workshops.spatial_grounding_v1.main_p_execution import load_plan, publish_progress, wait_json
from experiments.workshops.spatial_grounding_v1.paper_engineering import record


ROOT = Path("artifacts/workshops/spatial_grounding_v1")


def make_plan(tmp_path):
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}")
    plan = {
        "direction": record(ROOT / "main_p_execution_20260923dm.json"),
        "assignment": record(ROOT / "main_n3_lat_p_assignment_20260923.json"),
        "fixed_input_completion": record(ROOT / "infrastructure/n3-fixed-input-20260923cw/completed-20260923dk/completion.json"),
        "candidate": record(candidate),
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return path, candidate


def test_main_p_retains_six_original_cells_and_unscored_timing(tmp_path):
    path, candidate = make_plan(tmp_path)
    _, cells = load_plan(path)
    assert len(cells) == 6
    assert [cell.row["prompt_id"] for cell in cells] == [
        "LAT-I-POS", "LAT-D-POS", "LAT-C-POS", "LAT-C-NEG", "LAT-D-NEG", "LAT-I-NEG",
    ]
    assert all(cell.row["sampling_seed"] == 2026092201 for cell in cells)
    assert all(cell.row["prediction_time_mapping_status"] == "unqualified" for cell in cells)
    assert all(cell.stage == "P" and cell.family == "LAT" and cell.model == "N3" for cell in cells)
    candidate.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="bytes differ"):
        load_plan(path)


def test_wait_stops_on_peer_failure_without_retry(tmp_path):
    (tmp_path / "failure-simulator.json").write_text('{"error":"native failure"}')
    with pytest.raises(RuntimeError, match="peer failed"):
        wait_json(tmp_path / "ready.json", tmp_path)


def test_progress_advances_through_all_six_cells_without_exclusive_create_failure(tmp_path):
    path, _ = make_plan(tmp_path)
    _, cells = load_plan(path)
    completed = []
    for cell in cells:
        publish_progress(tmp_path, cell.cell_id, completed)
        completed.append({"cell_id": cell.cell_id, "status": "valid_model_failure"})
        publish_progress(tmp_path, cell.cell_id, completed)
        assert json.loads((tmp_path / "progress.json").read_bytes())["completed_cells"] == completed
    publish_progress(tmp_path, None, completed)
    assert json.loads((tmp_path / "progress.json").read_bytes()) == {
        "state": "complete", "current_cell": None, "completed_cells": completed,
    }


def make_resume_plan(tmp_path):
    path, _ = make_plan(tmp_path)
    plan, cells = load_plan(path)
    prior = tmp_path / "prior-plan.json"
    prior.write_bytes(path.read_bytes())
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"cell_id": cells[0].cell_id, "status": "valid_model_failure"}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    pointer = tmp_path / f"{cells[0].cell_id}.complete.json"
    pointer.write_text(json.dumps({
        "result": record(result), "manifest_path": str(manifest),
        "manifest_sha256": record(manifest)["sha256"],
    }))
    plan["resume"] = {
        "prior_plan": record(prior),
        "completed": [{
            "cell_id": cells[0].cell_id, "status": "valid_model_failure",
            "pointer": record(pointer),
        }],
        "attempt_ids": {cell.cell_id: "attempt-002" for cell in cells[1:]},
        "consumed_policy_requests": 15,
    }
    path.write_text(json.dumps(plan))
    return path, plan, result


def test_resume_preserves_failure_and_only_runs_five_remaining_cells(tmp_path):
    path, _, result = make_resume_plan(tmp_path)
    plan, cells = load_plan(path)
    assert len(cells) == 5
    assert cells[0].cell_id == "LAT-P01-N3-D-POS"
    assert all(cell.row["attempt_id"] == "attempt-002" for cell in cells)
    assert plan["resume"]["completed"][0]["status"] == "valid_model_failure"
    result.write_text("{}")
    with pytest.raises(ValueError, match="completed evidence changed"):
        load_plan(path)


@pytest.mark.parametrize("change", ["non_prefix", "request_overrun"])
def test_resume_rejects_skipped_cells_or_extra_request_allocation(tmp_path, change):
    path, plan, _ = make_resume_plan(tmp_path)
    if change == "non_prefix":
        plan["resume"]["completed"][0]["cell_id"] = "LAT-P01-N3-D-POS"
    else:
        plan["resume"]["consumed_policy_requests"] = 16
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="prefix|request allocation"):
        load_plan(path)
