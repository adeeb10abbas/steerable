from steerable_bridge.text import (
    coordinate_pairs,
    normalize_semantic_text,
    normalize_text,
    source_collection,
)


def test_normalize_text_changes_form_only() -> None:
    assert normalize_text("  Place_the Cup, in Sink! ") == "place the cup in sink"
    assert normalize_semantic_text("微調花盆位置") == "微調花盆位置"


def test_coordinate_parser_handles_bracketed_and_contextual_pairs() -> None:
    assert coordinate_pairs("Move from [10, 20] to [30, 40]") == [
        (10.0, 20.0),
        (30.0, 40.0),
    ]
    assert coordinate_pairs("Grasp the toy at 158, 193") == [(158.0, 193.0)]


def test_source_collection_is_explicit() -> None:
    path = "/root/numpy_256/bridge_data_v2/site/task/train/out.npy"
    assert source_collection(path) == "bridge_data_v2"
