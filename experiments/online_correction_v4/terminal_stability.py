"""Registered passive-settling stability and support predicates for horizontal fixtures."""

from __future__ import annotations

import math
from typing import Sequence

from experiments.online_correction_v4.adapters import TerminalPhysicalPredicates
from experiments.online_correction_v4.droid_reset import (
    ANGULAR_SPEED_TOLERANCE_RAD_S,
    LINEAR_SPEED_TOLERANCE_M_S,
)
from experiments.online_correction_v4.droid_reset_verify import POSITION_TOLERANCE_M
from experiments.online_correction_v4.droid_task_files.constants import TARGET_OBJECT

# Geodesic orientation drift bound paired with reset stability evidence (2 deg).
ORIENTATION_POSE_TOLERANCE_RAD = math.radians(2.0)

HORIZONTAL_SUPPORT_CONTACTS = frozenset({"table"})


def geodesic_orientation_delta_rad(baseline_wxyz: Sequence[float], current_wxyz: Sequence[float]) -> float:
    """Return the geodesic angle between two unit quaternions in wxyz order."""
    bw, bx, by, bz = (float(v) for v in baseline_wxyz[:4])
    cw, cx, cy, cz = (float(v) for v in current_wxyz[:4])
    b_norm = math.sqrt(bw * bw + bx * bx + by * by + bz * bz)
    c_norm = math.sqrt(cw * cw + cx * cx + cy * cy + cz * cz)
    if b_norm <= 0.0 or c_norm <= 0.0:
        return float("inf")
    bw, bx, by, bz = bw / b_norm, bx / b_norm, by / b_norm, bz / b_norm
    cw, cx, cy, cz = cw / c_norm, cx / c_norm, cy / c_norm, cz / c_norm
    dot = abs(bw * cw + bx * cx + by * cy + bz * cz)
    dot = min(1.0, max(-1.0, dot))
    return 2.0 * math.acos(dot)


def position_drift_m(
    baseline_xyz: Sequence[float],
    current_xyz: Sequence[float],
) -> float:
    return float(
        math.sqrt(
            sum((float(a) - float(b)) ** 2 for a, b in zip(baseline_xyz[:3], current_xyz[:3]))
        )
    )


def evaluate_horizontal_terminal_sample(
    *,
    detached: bool,
    linear_speed_m_s: float,
    angular_speed_rad_s: float,
    support_contacts: tuple[str, ...] | None,
    position_drift_m: float = 0.0,
    orientation_drift_rad: float = 0.0,
    predicates_available: bool = True,
    missing_fields: tuple[str, ...] = (),
) -> TerminalPhysicalPredicates:
    """Evaluate one passive-settling sample using registered velocity and pose tolerances."""
    if not predicates_available:
        return TerminalPhysicalPredicates(
            available=False,
            missing_fields=tuple(missing_fields or ("terminal_predicates",)),
        )

    velocity_stable = (
        linear_speed_m_s <= LINEAR_SPEED_TOLERANCE_M_S
        and angular_speed_rad_s <= ANGULAR_SPEED_TOLERANCE_RAD_S
    )
    pose_stable = (
        position_drift_m <= POSITION_TOLERANCE_M
        and orientation_drift_rad <= ORIENTATION_POSE_TOLERANCE_RAD
    )
    stable_for_dwell = detached and velocity_stable and pose_stable

    if support_contacts is None:
        allowed_support = False
        missing = tuple(sorted(set(missing_fields) | {"support_contact_evidence"}))
        support_evidence_available = False
    else:
        support_evidence_available = True
        missing = tuple(missing_fields)
        allowed_support = detached and any(
            contact in HORIZONTAL_SUPPORT_CONTACTS for contact in support_contacts
        )

    return TerminalPhysicalPredicates(
        available=True,
        allowed_support=allowed_support,
        stable_for_dwell=stable_for_dwell,
        linear_speed_m_s=linear_speed_m_s,
        angular_speed_rad_s=angular_speed_rad_s,
        position_drift_m=position_drift_m,
        orientation_drift_rad=orientation_drift_rad,
        support_contacts=support_contacts or (),
        support_evidence_available=support_evidence_available,
        missing_fields=missing,
    )


def target_object_contact_names(
    contact_names: Sequence[str],
    *,
    target_object: str = TARGET_OBJECT,
) -> tuple[str, ...]:
    """Reduce raw contact sensor names to registered support contact labels."""
    labels: set[str] = set()
    for name in contact_names:
        lowered = str(name).lower()
        if target_object.lower() not in lowered:
            continue
        if "table" in lowered:
            labels.add("table")
    return tuple(sorted(labels))
