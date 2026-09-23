"""Camera/proprioception-only boundary shared by the two native policy clients."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .adapters import AdapterError


CAMERAS = ("wrist_cam", "over_shoulder_left_camera", "over_shoulder_right_camera")


def native_policy_observation(value: Mapping[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {"image_obs": {}, "proprio_obs": {}}
    for group, names in (("image_obs", CAMERAS), ("proprio_obs", ("arm_joint_pos", "gripper_pos"))):
        fields = value.get(group)
        if not isinstance(fields, Mapping):
            raise AdapterError(f"native observation lacks {group}")
        for name in names:
            array = fields.get(name)
            if hasattr(array, "detach"):
                array = array.detach().cpu().numpy()
            array = np.asarray(array)
            if group == "image_obs":
                if (array.dtype != np.uint8 or array.ndim != 4 or array.shape[0] != 1
                        or array.shape[-1] != 3 or not np.ptp(array)):
                    raise AdapterError(f"{name} must be a nonblank one-environment uint8 RGB batch")
            elif (array.shape != (1, 7 if name == "arm_joint_pos" else 1)
                  or not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all()):
                raise AdapterError(f"{name} has invalid native proprioception shape or values")
            result[group][name] = array.copy()
    return result


def nano_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    """Leave resizing/composition and gripper conversion to the pinned service."""
    native = native_policy_observation(value)
    cameras, proprio = native["image_obs"], native["proprio_obs"]
    return {
        "observation/wrist_image_left": cameras["wrist_cam"][0],
        "observation/exterior_image_1_left": cameras["over_shoulder_left_camera"][0],
        "observation/exterior_image_2_left": cameras["over_shoulder_right_camera"][0],
        "observation/joint_position": proprio["arm_joint_pos"][0],
        "observation/gripper_position": proprio["gripper_pos"][0],
    }


def dreamzero_observation(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Restore transported arrays to tensors before the unmodified official extractor."""
    native = native_policy_observation(value)
    import torch

    return {
        group: {name: torch.from_numpy(array) for name, array in fields.items()}
        for group, fields in native.items()
    }
