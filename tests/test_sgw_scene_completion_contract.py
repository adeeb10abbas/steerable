import importlib.util
import importlib

import pytest


def api():
    name = 'experiments.workshops.spatial_grounding_v1.scene_completion_contract'
    assert importlib.util.find_spec(name), 'Completion runner must preserve the DIST plate in resets and scoring'
    return importlib.import_module(name)


def test_distance_retains_both_anchors_and_serialized_reset_order():
    m = api()
    assert m.scored_names('DIST') == ('rubiks_cube', 'bowl', 'plate')
    value = {'family': 'DIST', 'object_poses': {'plate': 3, 'bowl': 2, 'rubiks_cube': 1}}
    result = m.restore_scored_order(value)
    assert list(result['object_poses']) == ['rubiks_cube', 'bowl', 'plate']
    assert result['object_poses']['plate'] == 3
    with pytest.raises(ValueError, match='plate'):
        m.restore_scored_order({'family': 'DIST', 'object_poses': {'bowl': 2, 'rubiks_cube': 1}})


def test_measured_distance_goal_targets_bind_matching_contact_surfaces():
    m = api()
    objects = {'rubiks_cube': {'geometric_center_env_local_xyz_m': [.46, -.1, .12],
                              'bbox_env_local_min_xyz_m': [.43, -.13, .09]},
               'dist_bowl_landing_support': {'geometric_center_env_local_xyz_m': [.46, .06, .07],
                                             'bbox_env_local_max_xyz_m': [.54, .14, .09]},
               'dist_plate_landing_support': {'geometric_center_env_local_xyz_m': [.46, -.26, .07],
                                              'bbox_env_local_max_xyz_m': [.54, -.18, .09]}}
    goals = m.measured_goal_supports('DIST', objects)
    assert goals['near_bowl']['cube_center_env_local_xyz_m'] == pytest.approx([.46, .06, .12])
    assert goals['near_plate']['cube_center_env_local_xyz_m'] == pytest.approx([.46, -.26, .12])
    assert goals['near_plate']['contact_sensor_id'] == 'rubiks_cube__dist_plate_landing_support'


def test_side_measurements_use_world_axes_without_changing_goal_meaning():
    m = api()
    def obj(y): return {'geometric_center_env_local_xyz_m': [.5, y, .12]}
    assert m.measured_side('LAT', {'rubiks_cube': obj(.18)}) == 'left'
    assert m.measured_side('DIST', {'bowl': obj(.1), 'plate': obj(-.3)}) == 'left'
    assert m.measured_side('HEIGHT', {'height_upper_support': obj(-.3), 'height_lower_support': obj(.1)}) == 'right'
    with pytest.raises(ValueError): m.measured_side('LAT', {'rubiks_cube': obj(0)})
