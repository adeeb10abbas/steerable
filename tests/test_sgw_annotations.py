import pytest

from experiments.workshops.spatial_grounding_v1.prediction_annotations import (
    make_annotation,
    validate_annotation_panel,
    validate_prediction_alignment,
)


def _prediction(**overrides):
    value = {
        "request_id": "req-1",
        "camera_name": "front",
        "reset_id": "reset-1",
        "target_action_step": 12,
        "decoded": True,
    }
    value.update(overrides)
    return value


def test_alignment_requires_same_request_camera_reset_and_executed_prefix():
    assert validate_prediction_alignment(
        _prediction(), request_id="req-1", camera_name="front",
        reset_id="reset-1", executed_action_count=32,
    ) == 12
    with pytest.raises(ValueError):
        validate_prediction_alignment(
            _prediction(request_id="other"), request_id="req-1", camera_name="front",
            reset_id="reset-1", executed_action_count=32,
        )
    with pytest.raises(ValueError):
        validate_prediction_alignment(
            _prediction(target_action_step=32), request_id="req-1", camera_name="front",
            reset_id="reset-1", executed_action_count=32,
        )


def test_unobservable_future_cannot_receive_geometry_label():
    with pytest.raises(ValueError):
        make_annotation(
            _prediction(), rater_id="r1", label="positive", observable=False,
            request_id="req-1", camera_name="front", reset_id="reset-1",
            executed_action_count=32,
        )
    annotation = make_annotation(
        _prediction(), rater_id="r1", label="unknown", observable=False,
        request_id="req-1", camera_name="front", reset_id="reset-1",
        executed_action_count=32,
    )
    assert annotation.blind is True


def test_annotation_panel_stays_pending_until_two_raters_and_adjudication():
    one = make_annotation(
        _prediction(), rater_id="r1", label="positive", observable=True,
        request_id="req-1", camera_name="front", reset_id="reset-1",
        executed_action_count=32,
    )
    assert validate_annotation_panel([one])["status"] == "pending"
    two = make_annotation(
        _prediction(), rater_id="r2", label="unknown", observable=True,
        request_id="req-1", camera_name="front", reset_id="reset-1",
        executed_action_count=32,
    )
    assert validate_annotation_panel([one, two])["status"] == "pending"
    assert validate_annotation_panel([one, two], adjudicated_label="positive")["status"] == "complete"
