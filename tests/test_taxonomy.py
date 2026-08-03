from steerable_bridge.taxonomy import classify_command, generated_wrapper_paraphrases


def classify(command: object):
    return classify_command(
        command,
        canonical_task="put the cup in the sink",
        canonical_subtask="Reach for the cup",
    )


def test_taxonomy_precedence_separates_modalities() -> None:
    assert classify("put the cup in the sink")["automatic_class"] == "task_level_language"
    assert classify("Move to [12, 30]")["automatic_class"] == "point_coordinate_grounding"
    assert classify("Move via [12, 30], [20, 40]")["automatic_class"] == "multi_point_gripper_trace"
    assert classify("Move to [12, 30] and grasp")["automatic_class"] == "hybrid_or_added_behavioral_semantics"
    assert classify("Move left")["automatic_class"] == "atomic_motion_or_gripper_command"
    assert classify(17)["automatic_class"] == "malformed_or_unclear"


def test_generated_wrappers_preserve_verbatim_canonical() -> None:
    canonical = "put the cup in the sink"
    train, heldout = generated_wrapper_paraphrases(canonical)
    assert len(set(train)) == 4
    assert len(set(heldout)) == 2
    assert not set(train) & set(heldout)
    assert all(canonical in value for value in train + heldout)


def test_grounded_action_and_constraint_contradictions_are_not_paraphrases() -> None:
    grounded = classify_command(
        "Grasp cloth at [97, 108]",
        canonical_task="fold the cloth",
        canonical_subtask="Grasp the cloth",
    )
    assert grounded["automatic_class"] == "point_coordinate_grounding"

    relation_change = classify_command(
        "Place the pan on the cloth",
        canonical_task="put the pan in the yellow cloth",
        canonical_subtask="Place pan in yellow cloth",
    )
    assert relation_change["same_intent_candidate"] is False

    direction_change = classify_command(
        "Move can forward and to the left",
        canonical_task="move the can",
        canonical_subtask="Move can to front right corner",
    )
    assert direction_change["same_intent_candidate"] is False


def test_coordinate_aliases_and_action_composition_are_distinct() -> None:
    point = classify_command(
        "Open gripper above container at [145,122]",
        canonical_task="put the cup in the container",
        canonical_subtask="Open gripper above container",
    )
    assert point["automatic_class"] == "point_coordinate_grounding"

    hybrid = classify_command(
        "Move to [12,30] and grasp",
        canonical_task="pick up the cup",
        canonical_subtask="Reach for and grasp the cup",
    )
    assert hybrid["automatic_class"] == "hybrid_or_added_behavioral_semantics"

    trace = classify_command(
        "Move via <12,30> <40,50>",
        canonical_task="move the cup",
        canonical_subtask="Move toward the cup",
    )
    assert trace["automatic_class"] == "multi_point_gripper_trace"
    assert trace["coordinate_count"] == 2

    for command in (
        "Release cup and open drawer at [10,20]",
        "Grasp cup and close drawer at [10,20]",
        "Open gripper and open drawer at [10,20]",
    ):
        result = classify_command(
            command,
            canonical_task="rearrange the scene",
            canonical_subtask="Manipulate the object",
        )
        assert result["automatic_class"] == "hybrid_or_added_behavioral_semantics"


def test_role_and_order_swaps_are_not_paraphrase_candidates() -> None:
    object_role_swap = classify_command(
        "Place plate on cup",
        canonical_task="stack the objects",
        canonical_subtask="Place cup on plate",
    )
    assert object_role_swap["same_intent_candidate"] is False

    relation_role_swap = classify_command(
        "Move bowl to the left of cup",
        canonical_task="rearrange the table",
        canonical_subtask="Move cup to the left of bowl",
    )
    assert relation_role_swap["same_intent_candidate"] is False

    direction_order_swap = classify_command(
        "Move right then left",
        canonical_task="move the gripper",
        canonical_subtask="Move left then right",
    )
    assert direction_order_swap["same_intent_candidate"] is False
