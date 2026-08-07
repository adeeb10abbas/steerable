"""Static tests for the DreamZero V3-B003 runtime contract."""

from pathlib import Path

from experiments.v3.dreamzero_phase_b.compile_cell import (
    _normalize_frozen_failure_stage,
)
from experiments.v3.dreamzero_phase_b.contract import load_cells


ROOT = Path(__file__).resolve().parents[1]


def test_registered_cells_are_complete_matched_seed_blocks() -> None:
    cells = load_cells(ROOT)
    assert len(cells) == 108
    assert {cell.seed for cell in cells} == set(range(9400, 9427))
    for seed in range(9400, 9427):
        block = [cell for cell in cells if cell.seed == seed]
        assert len(block) == 4
        assert {(cell.arm, cell.relation) for cell in block} == {
            ("control", "left"),
            ("control", "right"),
            ("position_mirrored", "left"),
            ("position_mirrored", "right"),
        }
        assert {cell.row["execution_order_index_within_seed"] for cell in block} == {
            1, 2, 3, 4
        }


def test_registered_noise_seed_is_constant_while_pair_labels_match() -> None:
    for cell in load_cells(ROOT):
        assert cell.row["effective_model_noise_seed"] == 1140
        assert cell.row["registered_sampling_seed_label"] == cell.seed


def test_initial_reset_sample_does_not_mark_behavior_as_executed() -> None:
    source = (
        ROOT / "experiments" / "v3" / "dreamzero_phase_b" / "robolab_bridge.py"
    ).read_text()
    assert "def _sample(self, action_step: int, *, persist: bool = True)" in source
    assert "self._sample(0, persist=False)" in source


def test_queue_restores_the_validated_isaac_and_glvnd_library_path() -> None:
    source = (
        ROOT / "experiments" / "v3" / "dreamzero_phase_b" / "queue.py"
    ).read_text()
    assert "/data/users/ali/glvnd/lib" in source
    assert '"LD_LIBRARY_PATH": FROZEN_LD_LIBRARY_PATH' in source


def test_phase_b_bridge_keeps_legacy_stage_separate_from_v3_taxonomy() -> None:
    source = (
        ROOT / "experiments" / "v3" / "dreamzero_phase_b" / "robolab_bridge.py"
    ).read_text()
    assert 'failure_stage = "success"' in source
    assert 'failure_category = "correct"' in source
    assert '"frozen_failure_stage": failure_stage' in source
    assert '"failure_taxonomy": failure_category' in source


def test_compiler_recovers_known_taxonomy_stage_mixup_from_retained_samples() -> None:
    capture = {
        "frozen_failure_stage": "correct",
        "requested_success": True,
        "requested_relation": "right",
        "samples": [
            {"object_xyz": [0.3, -0.2, 0.1], "reference_xyz": [0.4, 0.0, 0.1]}
        ],
    }
    normalized = _normalize_frozen_failure_stage(capture)
    assert normalized["frozen_failure_stage"] == "success"
    assert capture["frozen_failure_stage"] == "correct"


def test_compiler_preserves_already_valid_legacy_stage() -> None:
    capture = {"frozen_failure_stage": "no_object_interaction"}
    assert _normalize_frozen_failure_stage(capture) is capture
