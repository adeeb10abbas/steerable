"""Tests for the post-physical-gate Nano V3-B005 queue."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/vla_wam_shared_v3/phase_b/nano_lateral_sweep_v3b005"


def test_queue_has_complete_matched_blocks_and_balanced_latin_positions() -> None:
    rows = [json.loads(line) for line in (BASE / "nano_lateral_v3b005_cells.jsonl").read_text().splitlines()]
    assert len(rows) == 210
    assert len({row["cell_id"] for row in rows}) == 210
    for seed in range(9500, 9515):
        block = [row for row in rows if row["environment_seed"] == seed]
        assert len(block) == 14
        assert {(row["level_index"], row["relation"]) for row in block} == {
            (level, relation) for level in range(7) for relation in ("left", "right")
        }
        assert {row["execution_order_index_within_seed"] for row in block} == set(range(1, 15))
    for level in range(7):
        for relation in ("left", "right"):
            first_fourteen = [
                row["execution_order_index_within_seed"] for row in rows
                if row["environment_seed"] < 9514
                and row["level_index"] == level and row["relation"] == relation
            ]
            assert set(first_fourteen) == set(range(1, 15))


def test_manifest_keeps_behavior_unreleased_after_zero_request_gate() -> None:
    manifest = json.loads((BASE / "nano_lateral_v3b005_manifest.json").read_text())
    assert manifest["counts"]["registered_cells"] == 210
    assert manifest["physical_gate"]["model_request_count"] == 0
    assert manifest["physical_gate"]["behavioral_episode_count"] == 0
    assert manifest["behavioral_release"] is False
