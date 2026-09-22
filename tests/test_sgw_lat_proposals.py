from experiments.workshops.spatial_grounding_v1.lat_proposals import propose_lat_layouts


def test_proposals_are_bounded_neutral_and_unqualified():
    workspace = {
        "measurement_schema_version": "sgw-01-lat-measured-workspace-v2",
        "eef_position_env_local_xyz_m": [0.3, 0, 0.4],
        "objects": {
            "table": {"bbox_env_local_min_xyz_m": [0, -1, 0], "bbox_env_local_max_xyz_m": [1, 1, 0.1]},
            "rubiks_cube": {"bbox_env_local_min_xyz_m": [0, 0, 0], "bbox_env_local_max_xyz_m": [.1, .1, .1], "root_position_env_local_xyz_m": [.4, 0, .1], "root_quaternion_world_wxyz": [1, 0, 0, 0], "geometric_center_offset_root_local_xyz_m": [0, .02, 0], "com_position_env_local_xyz_m": [.4, 0, .1]},
            "bowl": {"bbox_env_local_min_xyz_m": [0, 0, 0], "bbox_env_local_max_xyz_m": [.2, .2, .1], "root_position_env_local_xyz_m": [.5, 0, .1], "root_quaternion_world_wxyz": [1, 0, 0, 0], "geometric_center_offset_root_local_xyz_m": [0, 0, 0], "com_position_env_local_xyz_m": [.5, 0, .1]},
        },
    }
    rows = propose_lat_layouts(workspace, seed=4, count=3)
    assert len(rows) == 3
    assert all(row["status"].startswith("proposed_unqualified") for row in rows)
    assert all(abs(row["object_root_poses"]["rubiks_cube"]["position_m"][1] + .02 - row["object_root_poses"]["bowl"]["position_m"][1]) < 1e-9 for row in rows)
