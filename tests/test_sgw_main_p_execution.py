import json
from pathlib import Path

import pytest

from experiments.workshops.spatial_grounding_v1.main_p_execution import load_plan, wait_json
from experiments.workshops.spatial_grounding_v1.paper_engineering import record


ROOT = Path("artifacts/workshops/spatial_grounding_v1")


def test_main_p_retains_six_original_cells_and_unscored_timing(tmp_path):
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
