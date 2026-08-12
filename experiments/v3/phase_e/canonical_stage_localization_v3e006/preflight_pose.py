"""Pure pose normalization used by the V3-E006 zero-model health probe."""

from __future__ import annotations

from typing import Any


def scalar_z_from_world_pose(pose: tuple[Any, Any]) -> float:
    """Return scalar world-frame z from Isaac's single-prim pose tuple."""
    position, _orientation = pose
    value = position[2]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)
