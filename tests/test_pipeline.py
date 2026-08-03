import json

from steerable_bridge.pipeline import _write_taxonomy


def test_released_candidate_counts_exclude_the_canonical_surface(tmp_path) -> None:
    common = {
        "steering_trajectory_id": 7,
        "source_collection": "bridge_data_v2",
        "pool_id": "pool_1",
        "pool_index": 0,
        "canonical_task": "pick up the cup",
        "canonical_subtask": "Reach for the cup",
        "coordinate_count": 0,
        "has_unresolved_placeholder": False,
        "has_numeric_angle_coordinate": False,
        "has_unparsed_square_bracket": False,
        "needs_semantic_review": False,
    }
    rows = [
        {
            **common,
            "slot_index": 0,
            "command": "Reach for the cup",
            "normalized_command": "reach for the cup",
            "automatic_class": "subtask_same_intent_paraphrase_candidate",
            "rule_reason": "canonical_subtask_exact_normalized_match",
            "same_intent_candidate": True,
        },
        {
            **common,
            "slot_index": 1,
            "command": "Approach the cup",
            "normalized_command": "approach the cup",
            "automatic_class": "subtask_same_intent_paraphrase_candidate",
            "rule_reason": "matching_action_family_and_no_added_entities",
            "same_intent_candidate": True,
        },
    ]

    _, pools, _ = _write_taxonomy(tmp_path / "taxonomy.csv", rows)
    assert len(pools) == 1
    pool = pools[0]
    assert pool["canonical_present"] is True
    assert pool["paraphrase_count_excluding_canonical"] == 1
    assert pool["automatic_candidate_count"] == 1
    assert json.loads(pool["automatic_candidate_surfaces"]) == ["Approach the cup"]
