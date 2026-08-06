from __future__ import annotations

from pathlib import Path

import pytest

from experiments.v3.cosmos_nano_lateral_sweep.calibration_design import (
    CONTROL_BOWL_Y_M,
    CalibrationDesignError,
    dense_candidates,
    neutral_under_frozen_cones,
    select_largest_radius,
    seven_levels,
    xy_aabb_separation_m,
)


def test_dense_grid_and_largest_radius_are_exactly_prespecified() -> None:
    scan = dense_candidates(
        lower_y_m=CONTROL_BOWL_Y_M - 0.125,
        upper_y_m=CONTROL_BOWL_Y_M + 0.125,
    )
    radius_mm, levels = select_largest_radius(scan)
    assert radius_mm == 120
    assert levels == seven_levels(120)
    assert [round((y - CONTROL_BOWL_Y_M) * 1000) for y in levels] == [
        -120, -80, -40, 0, 40, 80, 120
    ]


def test_release_fails_closed_below_minimum_radius() -> None:
    scan = dense_candidates(
        lower_y_m=CONTROL_BOWL_Y_M - 0.085,
        upper_y_m=CONTROL_BOWL_Y_M + 0.085,
    )
    with pytest.raises(CalibrationDesignError, match="radius"):
        select_largest_radius(scan)


def test_original_zero_centering_was_analytically_infeasible() -> None:
    cube_x = 0.303364634513855
    cube_y = 0.12396888434886932
    bowl_x = 0.44258353114128113
    assert neutral_under_frozen_cones(
        cube_x_m=cube_x, cube_y_m=cube_y, bowl_x_m=bowl_x, bowl_y_m=0.0
    )
    assert not neutral_under_frozen_cones(
        cube_x_m=cube_x, cube_y_m=cube_y, bowl_x_m=bowl_x, bowl_y_m=-0.09
    )
    assert neutral_under_frozen_cones(
        cube_x_m=cube_x,
        cube_y_m=cube_y,
        bowl_x_m=bowl_x,
        bowl_y_m=CONTROL_BOWL_Y_M + 0.09,
    )


def test_xy_gap_is_zero_on_overlap_and_positive_when_separated() -> None:
    assert xy_aabb_separation_m((0, 0), (1, 1), (0.5, 0.5), (2, 2)) == 0
    assert xy_aabb_separation_m((0, 0), (1, 1), (1.3, 1.4), (2, 2)) == pytest.approx(0.5)


def test_live_driver_forces_and_records_each_fresh_physical_reset() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "experiments/v3/cosmos_nano_lateral_sweep/model_blind_lateral_calibration.py"
    ).read_text()
    assert "def fresh_physical_reset" in source
    assert "counter.zero_()" in source
    assert '"episode_length_buf_after_reset": after_reset' in source
    assert '"pre_teleport_positions_robot_base_m": pre_teleport_positions' in source
