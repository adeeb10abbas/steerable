"""Blind, provenance-aware annotation records for decoded SGW-01 futures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

AnnotationLabel = Literal["positive", "negative", "moving", "stationary", "unknown"]


@dataclass(frozen=True)
class PredictionAnnotation:
    request_id: str
    camera_name: str
    reset_id: str
    target_action_step: int
    rater_id: str
    label: AnnotationLabel
    observable: bool
    blind: bool = True
    note: str | None = None


def validate_prediction_alignment(
    prediction: Mapping[str, object],
    *,
    request_id: str,
    camera_name: str,
    reset_id: str,
    executed_action_count: int,
) -> int:
    """Validate physical-time alignment inside the executed action prefix."""
    for key, expected in (("request_id", request_id), ("camera_name", camera_name), ("reset_id", reset_id)):
        if prediction.get(key) != expected:
            raise ValueError(f"prediction provenance mismatch: {key}")
    step = prediction.get("target_action_step")
    if not isinstance(step, int) or step < 0 or step >= executed_action_count:
        raise ValueError("prediction target is outside the executed action prefix")
    if not prediction.get("decoded", True):
        raise ValueError("prediction is not decodable")
    return step


def make_annotation(
    prediction: Mapping[str, object],
    *,
    rater_id: str,
    label: AnnotationLabel,
    observable: bool,
    request_id: str,
    camera_name: str,
    reset_id: str,
    executed_action_count: int,
) -> PredictionAnnotation:
    step = validate_prediction_alignment(
        prediction,
        request_id=request_id,
        camera_name=camera_name,
        reset_id=reset_id,
        executed_action_count=executed_action_count,
    )
    if not observable and label != "unknown":
        raise ValueError("unobservable predicted images must be labelled unknown")
    return PredictionAnnotation(
        request_id=request_id, camera_name=camera_name, reset_id=reset_id,
        target_action_step=step, rater_id=rater_id, label=label,
        observable=observable, blind=True,
    )
