from steerable_bridge.manifests import (
    build_master_manifest,
    materialize_views,
    validate_views,
)


def test_four_views_keep_robot_rows_identical(tmp_path) -> None:
    frames = [
        {
            "steering_trajectory_id": 7,
            "lerobot_episode_index": 3,
            "frame_index": index,
            "annotation_timestep": index,
            "raw_bridge_observation_index": index + 1,
            "task_instruction": "put the cup in the sink",
            "task_id": "task_1",
            "subtask_id": "subtask_1" if index < 2 else "subtask_2",
            "subtask_text": "reach for the cup" if index < 2 else "place cup in sink",
            "semantic_segment_index": 0 if index < 2 else 1,
            "source_collection": "bridge_data_v2",
            "task_family": "put_cup_in_sink",
            "observation_ref": f"video#frame={index}",
            "action_ref": f"actions#row={index}",
        }
        for index in range(4)
    ]
    master = build_master_manifest(frames, {7: "train"}, seed=11)
    views, _ = materialize_views(master, tmp_path, seed=11)
    result = validate_views(master, views, seed=11)
    assert result["all_structural_assertions_pass"] is True
    assert result["scientific_language_conditions_pass"] is False
    assert result["training_ready"] is False
    assert {value["reference_hash"] for value in result["conditions"].values()} == {
        result["master_reference_hash"]
    }


def test_validator_rejects_corrupted_language_view(tmp_path) -> None:
    frames = [
        {
            "steering_trajectory_id": 7,
            "lerobot_episode_index": 3,
            "frame_index": index,
            "annotation_timestep": index,
            "raw_bridge_observation_index": index + 1,
            "task_instruction": "put the cup in the sink",
            "task_id": "task_1",
            "subtask_id": "subtask_1",
            "subtask_text": "reach for the cup",
            "semantic_segment_index": 0,
            "source_collection": "bridge_data_v2",
            "task_family": "put_cup_in_sink",
            "observation_ref": f"video#frame={index}",
            "action_ref": f"actions#row={index}",
        }
        for index in range(2)
    ]
    master = build_master_manifest(frames, {7: "train"}, seed=11)
    views, _ = materialize_views(master, tmp_path, seed=11)
    corrupted = views["B_task_paraphrases"].copy()
    corrupted["heldout_instruction_pool"] = "[]"
    corrupted["selected_instruction"] = "wrong"
    corrupted["condition"] = "A_task_canonical"
    views["B_task_paraphrases"] = corrupted
    result = validate_views(master, views, seed=11)
    assert result["all_structural_assertions_pass"] is False
    checks = result["conditions"]["B_task_paraphrases"]["checks"]
    assert checks["condition_metadata_exact"] is False
    assert checks["expected_heldout_pool_size"] is False
    assert checks["selected_instruction_is_training_partition"] is False


def test_validator_requires_exactly_all_four_views(tmp_path) -> None:
    frames = [
        {
            "steering_trajectory_id": 7,
            "lerobot_episode_index": 3,
            "frame_index": 0,
            "annotation_timestep": 0,
            "raw_bridge_observation_index": 1,
            "task_instruction": "put the cup in the sink",
            "task_id": "task_1",
            "subtask_id": "subtask_1",
            "subtask_text": "reach for the cup",
            "semantic_segment_index": 0,
            "source_collection": "bridge_data_v2",
            "task_family": "put_cup_in_sink",
            "observation_ref": "video#frame=0",
            "action_ref": "actions#row=0",
        }
    ]
    master = build_master_manifest(frames, {7: "train"}, seed=11)
    views, _ = materialize_views(master, tmp_path, seed=11)
    del views["A_task_canonical"]
    result = validate_views(master, views, seed=11)
    assert result["condition_key_set_exact"] is False
    assert result["missing_condition_keys"] == ["A_task_canonical"]
    assert result["all_structural_assertions_pass"] is False
