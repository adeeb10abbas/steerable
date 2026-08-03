from steerable_bridge.join import (
    build_conservative_join,
    derive_lerobot_episode_records,
    derive_steering_task_records,
    iter_frame_records,
    validate_joined_trajectories,
)


def _fixture():
    commands = {
        "7": {
            "Reach cup": ["put cup in sink", "Reach cup", "Approach the cup"],
            "Place cup": ["put cup in sink", "Place cup", "Put the cup down"],
        }
    }
    trajectory_map = {
        "traj_idx_to_key": {
            "7": [
                "/data/numpy_256/bridge_data_v2/site/task/train/out.npy",
                2,
            ]
        }
    }
    episodes = [{"episode_index": 3, "tasks": ["Put cup in sink."], "length": 2}]
    steps = {"7": {"0": "Reach cup", "1": "Place cup", "2": "Place cup", "3": "Place cup"}}
    return commands, trajectory_map, episodes, steps


def test_join_never_depends_on_raw_index_equality() -> None:
    commands, trajectory_map, episodes, _ = _fixture()
    steering = derive_steering_task_records(commands, trajectory_map)
    lerobot = derive_lerobot_episode_records(episodes)
    joins, issues, counts = build_conservative_join(steering, lerobot)
    assert not issues
    assert counts["one_to_one_joined_episodes"] == 1
    assert joins[0]["lerobot_episode_index"] == 3
    assert joins[0]["steering_trajectory_id"] == 7
    assert joins[0]["raw_indices_equal"] is False


def test_direct_frame_mapping_requires_plus_two_sidecar_and_valid_pools() -> None:
    commands, trajectory_map, episodes, steps = _fixture()
    steering = derive_steering_task_records(commands, trajectory_map)
    joins, _, _ = build_conservative_join(
        steering, derive_lerobot_episode_records(episodes)
    )
    validations, issues = validate_joined_trajectories(joins, steps, commands)
    assert not issues
    assert validations[0]["validation_status"] == "eligible"
    assert validations[0]["annotation_minus_episode_length"] == 2
    frames = list(iter_frame_records(validations, steps, commands))
    assert [row["annotation_timestep"] for row in frames] == [0, 1]
    assert [row["raw_bridge_observation_index"] for row in frames] == [1, 2]
    assert [row["semantic_segment_index"] for row in frames] == [0, 1]
