"""Static tests for the DreamZero V3-B003 runtime contract."""

from pathlib import Path

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
