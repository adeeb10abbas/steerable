"""Evaluator-only observation audit fields for changed-observation visibility."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ObservationAuditEvidence:
    """Audit payload fields sampled alongside policy observations; never policy input."""

    reference_displacement_m: float
    camera_ids: tuple[str, ...] = ()
    moved_object_mask_pixels_by_camera: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ChangedObservationVisibilityResult:
    qualified: bool
    reason: str | None = None
    qualifying_camera_id: str | None = None
    qualifying_mask_pixels: int = 0


def build_observation_audit_payload(
    *,
    reference_displacement_m: float,
    camera_ids: Sequence[str],
    moved_object_mask_pixels_by_camera: Mapping[str, int] | None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "reference_displacement_m": float(reference_displacement_m),
        "camera_ids": list(camera_ids),
    }
    if moved_object_mask_pixels_by_camera is not None:
        payload["moved_object_mask_pixels_by_camera"] = {
            str(camera_id): int(pixels)
            for camera_id, pixels in moved_object_mask_pixels_by_camera.items()
        }
    if extra:
        payload.update(dict(extra))
    return payload


def parse_observation_audit(payload: bytes | str | Mapping[str, Any]) -> ObservationAuditEvidence | None:
    if isinstance(payload, Mapping):
        audit = payload
    else:
        if isinstance(payload, bytes):
            text = payload.decode("utf-8")
        else:
            text = payload
        try:
            parsed = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
            return None
        if not isinstance(parsed, dict):
            return None
        audit = parsed

    try:
        reference_displacement_m = float(audit.get("reference_displacement_m", 0.0))
    except (TypeError, ValueError):
        return None

    camera_ids_raw = audit.get("camera_ids")
    if isinstance(camera_ids_raw, list):
        camera_ids = tuple(str(item) for item in camera_ids_raw)
    else:
        camera_ids = ()

    mask_by_camera: dict[str, int] = {}
    raw_masks = audit.get("moved_object_mask_pixels_by_camera")
    if isinstance(raw_masks, Mapping):
        for camera_id, pixels in raw_masks.items():
            try:
                mask_by_camera[str(camera_id)] = int(pixels)
            except (TypeError, ValueError):
                continue

    return ObservationAuditEvidence(
        reference_displacement_m=reference_displacement_m,
        camera_ids=camera_ids,
        moved_object_mask_pixels_by_camera=mask_by_camera,
    )


def evaluate_changed_observation_visibility(
    audit: ObservationAuditEvidence,
    *,
    displacement_threshold_m: float,
    policy_camera_ids: Sequence[str],
) -> ChangedObservationVisibilityResult:
    """Require >= threshold displacement and a policy-camera mask; no displacement-only fallback."""
    if audit.reference_displacement_m + 1e-12 < displacement_threshold_m:
        return ChangedObservationVisibilityResult(
            qualified=False,
            reason="insufficient_reference_displacement",
        )

    expected = tuple(str(camera_id) for camera_id in policy_camera_ids)
    if not expected:
        return ChangedObservationVisibilityResult(
            qualified=False,
            reason="missing_policy_camera_ids",
        )

    if audit.camera_ids and tuple(audit.camera_ids) != expected:
        return ChangedObservationVisibilityResult(
            qualified=False,
            reason="audit_camera_ids_mismatch",
        )

    if not audit.moved_object_mask_pixels_by_camera:
        return ChangedObservationVisibilityResult(
            qualified=False,
            reason="missing_visibility_evidence",
        )

    for camera_id in expected:
        pixels = audit.moved_object_mask_pixels_by_camera.get(camera_id, 0)
        if pixels >= 1:
            return ChangedObservationVisibilityResult(
                qualified=True,
                qualifying_camera_id=camera_id,
                qualifying_mask_pixels=int(pixels),
            )

    return ChangedObservationVisibilityResult(
        qualified=False,
        reason="no_nonempty_mask_in_policy_cameras",
    )
