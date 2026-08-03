from steerable_bridge.splits import balance_rows, nested_split, stratified_split


def test_stratified_split_is_exact_deterministic_and_leakage_free() -> None:
    rows = [
        {
            "steering_trajectory_id": index,
            "lerobot_episode_index": 1000 + index,
            "source_collection": "bridge_data_v1" if index % 2 else "bridge_data_v2",
            "episode_length": 20 + index % 7,
            "semantic_segment_count": 3 + index % 4,
        }
        for index in range(30)
    ]
    first = stratified_split(rows, {"train": 12, "validation": 4, "test": 4}, seed=9)
    second = stratified_split(rows, {"train": 12, "validation": 4, "test": 4}, seed=9)
    assert {key: len(value) for key, value in first.items()} == {
        "train": 12,
        "validation": 4,
        "test": 4,
    }
    assert first == second
    ids = [row["steering_trajectory_id"] for values in first.values() for row in values]
    assert len(ids) == len(set(ids))


def test_original_capture_groups_do_not_cross_splits_and_pilot_is_nested() -> None:
    rows = [
        {
            "steering_trajectory_id": index,
            "lerobot_episode_index": 1000 + index,
            "source_collection": "bridge_data_v2",
            "task_family": f"family_{index % 3}",
            "normalized_task": f"task number {index}",
            "original_bridge_path": f"/capture/{index // 3}/out.npy",
            "episode_length": 20 + index % 7,
            "semantic_segment_count": 3 + index % 4,
        }
        for index in range(180)
    ]
    target = stratified_split(
        rows, {"train": 60, "validation": 20, "test": 20}, seed=13
    )
    membership: dict[str, set[str]] = {}
    for split, values in target.items():
        for row in values:
            membership.setdefault(row["original_bridge_path"], set()).add(split)
    assert all(len(splits) == 1 for splits in membership.values())

    pilot = nested_split(
        target, {"train": 20, "validation": 5, "test": 5}, seed=14
    )
    target_roles = {
        row["steering_trajectory_id"]: split
        for split, values in target.items()
        for row in values
    }
    assert all(
        target_roles[row["steering_trajectory_id"]] == split
        for split, values in pilot.items()
        for row in values
    )


def test_balance_rows_reports_source_specific_means() -> None:
    rows = {
        "train": [
            {
                "source_collection": "a",
                "episode_length": 10,
                "semantic_segment_count": 3,
            },
            {
                "source_collection": "b",
                "episode_length": 30,
                "semantic_segment_count": 7,
            },
        ]
    }
    balance = {row["source_collection"]: row for row in balance_rows(rows)}
    assert balance["a"]["mean_episode_length"] == 10
    assert balance["b"]["mean_episode_length"] == 30
